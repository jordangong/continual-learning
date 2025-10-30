import os
import random
import time
from typing import Optional

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data.data_module import DataModule
from src.models.model_factory import create_model, get_pretrained_normalization_params
from src.trainers.trainer import ContinualTrainer
from src.utils.logging import plot_accuracy_curve, plot_forgetting_curve, setup_logger
from src.utils.metrics import forgetting


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_distributed(rank: int, world_size: int) -> None:
    """
    Initialize the distributed environment.

    Args:
        rank: Rank of the current process
        world_size: Number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Set the device
    torch.cuda.set_device(rank)


def run_training(
    rank: Optional[int], world_size: Optional[int], cfg: DictConfig
) -> None:
    """
    Run the training process on a single process.

    Args:
        rank: Rank of the current process (None for single GPU)
        world_size: Number of processes (None for single GPU)
        cfg: Configuration
    """
    # Convert config to dictionary
    config = OmegaConf.to_container(cfg, resolve=True)

    # Check if eval_only mode is enabled
    eval_only = config.get("eval_only", False)

    # Set random seed
    set_seed(config["seed"])

    # Initialize distributed environment if using multiple GPUs
    distributed = rank is not None and world_size is not None
    if distributed:
        setup_distributed(rank, world_size)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    # Only print on main process if distributed
    if not distributed or (distributed and rank == 0):
        print(f"Using device: {device}")

    # Setup logger (only on main process if distributed)
    if not distributed or (distributed and rank == 0):
        # Create experiment-specific log directory
        experiment_name = config["experiment"]["name"]
        log_dir = os.path.join(config["paths"]["log_dir"], experiment_name)
        logger = setup_logger(log_dir)
        logger.info(f"Configuration: {OmegaConf.to_yaml(cfg)}")
    else:
        logger = None

    # Get pretrained model's normalization parameters if needed
    cache_dir = config["paths"].get("cache_dir", None)
    pretrained_model_mean, pretrained_model_std = None, None
    if config["dataset"].get("use_pretrained_norm", True):
        pretrained_model_mean, pretrained_model_std = (
            get_pretrained_normalization_params(
                model_config=config["model"], cache_dir=cache_dir
            )
        )
        print(
            "Using pretrained model's normalization:",
            f"mean={pretrained_model_mean}, std={pretrained_model_std}",
        )

    # Setup model
    model = create_model(
        model_config=config["model"],
        num_classes=config["dataset"]["num_classes"],
        device=device,
        cache_dir=cache_dir,
        continual_config=config["continual"],
    )

    # Setup data module
    data_module = DataModule(
        config,
        model_normalization_mean=pretrained_model_mean,
        model_normalization_std=pretrained_model_std,
    )

    # Setup trainer with distributed info if applicable
    trainer = ContinualTrainer(
        model=model,
        config=config,
        device=device,
        local_rank=rank if distributed else -1,
        data_module=data_module,
    )

    # Train or evaluate on each step
    num_steps = config["continual"]["num_steps"]
    accuracies = []
    forgetting_measures = []

    # If eval_only is True, we'll only evaluate the model without training
    if eval_only and not distributed or (eval_only and distributed and rank == 0):
        print("Running in evaluation-only mode. Loading model from checkpoint...")

    all_step_classes = []
    for step in range(num_steps):
        # Get data loaders for current step
        memory_data = data_module.get_memory_samples(step) if step > 0 else None
        step_classes, train_loader, test_loader = data_module.get_data_loaders(
            step,
            memory_data,
            distributed_sampler=distributed,
            rank=rank,
            world_size=world_size,
        )
        all_step_classes.append(step_classes)

        # For eval_only mode with calibration, populate historical checkpoint paths
        if eval_only and trainer.calibration_enabled and step > 0:
            # Populate checkpoint paths for all previous steps
            experiment_name = config["experiment"]["name"]
            experiment_checkpoint_dir = os.path.join(trainer.checkpoint_dir, experiment_name)
            # Add checkpoint paths for all previous steps (0 to step)
            for prev_step in range(step + 1):
                if prev_step not in trainer.historical_checkpoint_paths:
                    checkpoint_path = os.path.join(experiment_checkpoint_dir, f"step{prev_step}_best.pth")
                    trainer.historical_checkpoint_paths[prev_step] = checkpoint_path

                    if not distributed or (distributed and rank == 0):
                        # Verify checkpoint exists for offline calibration
                        if not os.path.exists(checkpoint_path):
                            print(f"Warning: Checkpoint not found for offline calibration: {checkpoint_path}")

        # Apply debug settings to limit data loaders if enabled
        train_loader, test_loader = data_module.limit_data_loaders(
            train_loader,
            test_loader,
            config.get("debug", {"enabled": False}),
            distributed=distributed,
            local_rank=rank if distributed else -1,
        )

        # Initialize prototypes if using prototypical classifier
        # This includes:
        # 1. Direct prototypical classifier (classifier.type == "prototypical")
        # 2. CLIP hybrid mode with prototypical learned classifier (classifier.type == "clip_text", mode == "hybrid", learned_classifier_type == "prototypical")
        is_prototypical = config["model"]["classifier"]["type"] == "prototypical"
        is_clip_hybrid_prototypical = (
            config["model"]["classifier"]["type"] == "clip_text"
            and config["model"]["classifier"].get("mode", "text") == "hybrid"
            and config["model"]["classifier"].get("learned_classifier_type", "linear") == "prototypical"
        )

        if (not (eval_only and config["continual"]["strategy"] != "zeroshot")) and (
            is_prototypical or is_clip_hybrid_prototypical
        ):
            # Only print on main process if distributed
            if not distributed or (distributed and rank == 0):
                print(
                    f"\n=== Initializing prototypes for step {step + 1}/{num_steps} ==="
                )

            # Check if we should use pretrained model for prototype initialization
            use_pretrained_for_prototypes = step != 0 and (
                config["continual"]
                .get("prototypical", {})
                .get("init_with_pretrained", False)
            )

            if use_pretrained_for_prototypes:
                # Only print on main process if distributed
                if not distributed or (distributed and rank == 0):
                    print("Using pretrained model for prototype initialization")

                # Create a fresh pretrained model for prototype initialization
                pretrained_model = create_model(
                    model_config=config["model"],
                    num_classes=config["dataset"]["num_classes"],
                    device=device,
                    cache_dir=cache_dir,
                    continual_config=config["continual"],
                )

                # Wrap with DDP if distributed
                if distributed:
                    pretrained_model = torch.nn.parallel.DistributedDataParallel(
                        pretrained_model, device_ids=[rank]
                    )

                model_to_use = (
                    pretrained_model.module if distributed else pretrained_model
                )
            else:
                # Use the current step model (existing behavior)
                model_to_use = model.module if distributed else model

            # For incremental learning, only reset prototypes on the first step
            # This ensures we maintain prototypes for previous classes
            reset_prototypes = step == 0

            # Initialize prototypes from training data but with test transforms
            # Create a temporary data loader with test transforms for prototype initialization
            # This ensures consistent feature extraction without augmentations
            temp_dataset = []
            trainset = train_loader.dataset
            # Handle nested datasets (e.g., Subsets, Concatenated datasets)
            while True:
                if hasattr(trainset, "dataset"):
                    trainset = trainset.dataset
                elif hasattr(trainset, "datasets"):
                    trainset = trainset.datasets
                else:
                    if isinstance(trainset, list):
                        temp_dataset.extend(trainset)
                    else:
                        temp_dataset.append(trainset)
                    break

            # Store original transform and temporarily replace with test transform
            original_transforms = []
            for dataset in temp_dataset:
                assert hasattr(dataset, "transform"), (
                    f"Dataset {dataset} does not have a transform attribute"
                )
                original_transforms.append(dataset.transform)
                dataset.transform = data_module.test_transform

            # Create a temporary data loader with evaluation batch size for faster processing
            temp_loader = torch.utils.data.DataLoader(
                train_loader.dataset,
                batch_size=test_loader.batch_size,  # Use larger eval batch size
                shuffle=False,  # No need to shuffle for prototype extraction
                num_workers=train_loader.num_workers,
                pin_memory=train_loader.pin_memory,
            )

            # Initialize prototypes using the temporary loader with test transforms
            if use_pretrained_for_prototypes:
                # Extract features using pretrained model and transfer to main model
                pretrained_features_list = []
                pretrained_labels_list = []

                model_to_use.eval()

                data_iter = tqdm(temp_loader, desc="Extracting features for prototypes")
                with torch.no_grad():
                    for batch in data_iter:
                        # Handle both 2-tuple and 3-tuple batches (with captions)
                        if len(batch) == 3:
                            inputs, targets, _ = batch  # Ignore captions for prototype extraction
                        else:
                            inputs, targets = batch
                        inputs = inputs.to(device)
                        targets = targets.to(device)

                        # Extract features using pretrained model
                        features = model_to_use.forward_features(inputs)
                        pretrained_features_list.append(features.cpu())
                        pretrained_labels_list.append(targets.cpu())

                # Transfer prototypes to the main model
                main_model = model.module if distributed else model

                # Move features back to device and compute prototypes
                features_device_list = []
                labels_device_list = []

                for features, labels in zip(
                    pretrained_features_list, pretrained_labels_list
                ):
                    features_device = features.to(device)
                    labels_device = labels.to(device)
                    features_device_list.append(features_device)
                    labels_device_list.append(labels_device)

                # Compute prototypes on main model using pretrained features
                main_model.compute_prototypes(
                    features_device_list, labels_device_list, reset=reset_prototypes
                )

                # Clean up pretrained model to free memory
                del pretrained_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                # Use existing behavior with current step model
                model_to_use.init_prototypes_from_data(
                    temp_loader, reset=reset_prototypes
                )

            # Restore original transform
            if original_transforms:
                for dataset, original in zip(temp_dataset, original_transforms):
                    assert hasattr(dataset, "transform"), (
                        f"Dataset {dataset} does not have a transform attribute"
                    )
                    dataset.transform = original

        # Initialize CLIP text classifier with class names
        if config["model"]["classifier"]["type"] == "clip_text":
            # Get class names for current step
            step_class_names = data_module.dataset.get_class_names(all_step_classes[step])

            # Get the actual model (unwrap DDP if needed)
            model_to_init = model.module if distributed else model

            # Check if text encoder is frozen or trainable
            freeze_text_encoder = config["model"]["classifier"].get("freeze_text_encoder", True)

            # Only print on main process if distributed
            if not distributed or (distributed and rank == 0):
                if freeze_text_encoder:
                    print(f"\n=== Initializing CLIP text embeddings for step {step + 1}/{num_steps} ===")
                else:
                    print(f"\n=== Setting up CLIP class names for step {step + 1}/{num_steps} ===")

            # Set class names for current step (incremental update)
            # If text encoder is frozen: precomputes and caches embeddings
            # If text encoder is trainable: only stores class names (embeddings computed in forward pass)
            model_to_init.set_class_names(class_names=step_class_names, class_indices=all_step_classes[step])

            if not distributed or (distributed and rank == 0):
                if freeze_text_encoder:
                    print(f"Cached text embeddings for {len(step_class_names)} classes: {step_class_names[:5]}...")
                else:
                    print(f"Stored class names for {len(step_class_names)} classes: {step_class_names[:5]}...")
                    print("Text embeddings will be computed dynamically during forward pass")

        # Train or evaluate on current step
        start_time = time.time()

        if eval_only:
            if config["continual"]["strategy"] != "zeroshot":
                # Load model from checkpoint and evaluate
                trainer._load_checkpoint(step, best=True)

            # Set prototypes initialized flag for prototypical classifier
            if config["model"]["classifier"]["type"] == "prototypical":
                trainer.model.classifier.classifier.prototypes_initialized = True

            # Handle evaluation with teacher model
            if trainer.ema_enabled and trainer.ema_eval_with_teacher:
                # Replace student with saved teacher AFTER loading checkpoint
                # This uses the teacher from the same training iteration as the best checkpoint
                trainer._replace_student_with_teacher()
                print("Using EMA teacher parameters from best checkpoint for evaluation")

            # Replace classifier with prototypes if using prototypical classifier
            if trainer.prototypical_enabled and trainer.prototypical_replace_classifiers:
                trainer._replace_classifier_with_prototypes(train_loader, all_step_classes[step])

            # Apply offline calibration if enabled
            if trainer.calibration_enabled:
                trainer.calibrate_previous_classifiers(step, train_loader, all_step_classes)

            # Evaluate model on test data
            _, test_acc = trainer._evaluate(test_loader)
            metrics = {"accuracy": test_acc}
            if step > 0:
                # Calculate forgetting measure if not the first step
                # This assumes previous steps have been evaluated already
                prev_accuracies = accuracies.copy()
                metrics["forgetting"] = forgetting(prev_accuracies + [test_acc], step)
            eval_time = time.time() - start_time
            if not distributed or (distributed and rank == 0):
                print(
                    f"Step {step + 1}/{num_steps} evaluation time: {eval_time:.2f} seconds"
                )
        else:
            # Train as usual
            metrics = trainer.train_step(step, all_step_classes, train_loader, test_loader)
            train_time = time.time() - start_time
            if not distributed or (distributed and rank == 0):
                print(
                    f"Step {step + 1}/{num_steps} training time: {train_time:.2f} seconds"
                )

        # Record metrics
        accuracies.append(metrics["accuracy"])
        if step > 0:
            forgetting_measures.append(metrics["forgetting"])

        # Log metrics (only on main process if distributed)
        if not distributed or (distributed and rank == 0) and logger is not None:
            logger.info(
                f"Step {step + 1}/{num_steps} - Accuracy: {metrics['accuracy']:.2f}%"
            )
            if step > 0:
                logger.info(
                    f"Step {step + 1}/{num_steps} - Forgetting: {metrics['forgetting']:.4f}"
                )

            # Calculate and log average metrics up to the current step
            current_avg_acc = np.mean(accuracies)
            logger.info(
                f"Step {step + 1}/{num_steps} - Average accuracy so far: {current_avg_acc:.2f}%"
            )

            if len(forgetting_measures) > 0:
                current_avg_fgt = np.mean(forgetting_measures)
                logger.info(
                    f"Step {step + 1}/{num_steps} - Average forgetting so far: {current_avg_fgt:.4f}"
                )

            # Log step-level metrics to TensorBoard
            global_epoch = (step + 1) * config["training"]["num_epochs"] - 1
            if config["logging"]["tensorboard"] and trainer.writer is not None:
                trainer.writer.add_scalar(
                    "global/accuracy", metrics["accuracy"], global_epoch
                )
                if step > 0:
                    trainer.writer.add_scalar(
                        "global/forgetting", metrics["forgetting"], global_epoch
                    )
                trainer.writer.add_scalar(
                    "global/avg_accuracy", current_avg_acc, global_epoch
                )
                if len(forgetting_measures) > 0:
                    trainer.writer.add_scalar(
                        "global/avg_forgetting", current_avg_fgt, global_epoch
                    )

            # Log step-level metrics to Weights & Biases
            if config["logging"]["wandb"]:
                log_data = {
                    "global/accuracy": metrics["accuracy"],
                    "global/avg_accuracy": current_avg_acc,
                }
                if step > 0:
                    log_data["global/forgetting"] = metrics["forgetting"]
                if len(forgetting_measures) > 0:
                    log_data["global/avg_forgetting"] = current_avg_fgt

                wandb.log(log_data, step=global_epoch)

    # Only create plots and log final metrics on main process if distributed
    if not distributed or (distributed and rank == 0):
        # Use the experiment name from the config
        experiment_name = config["experiment"]["name"]

        # Create experiment-specific plots directory
        plots_dir = os.path.join(config["paths"]["plots_dir"], experiment_name)
        os.makedirs(plots_dir, exist_ok=True)

        # Plot and save accuracy curve
        accuracy_plot_path = os.path.join(plots_dir, "accuracy_curve.png")
        plot_accuracy_curve(accuracies, save_path=accuracy_plot_path)

        # Plot and save forgetting curve
        if len(forgetting_measures) > 0:
            forgetting_plot_path = os.path.join(plots_dir, "forgetting_curve.png")
            plot_forgetting_curve(forgetting_measures, save_path=forgetting_plot_path)

        # Log final metrics
        if logger:
            logger.info(f"Final average accuracy: {np.mean(accuracies):.2f}%")
            if len(forgetting_measures) > 0:
                logger.info(
                    f"Final average forgetting: {np.mean(forgetting_measures):.4f}"
                )

        # Print completion message
        print("Continual learning completed!")
        print(f"Results saved to {config['paths']['output_root']}")

    # Clean up distributed environment
    if distributed:
        dist.destroy_process_group()


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main function for continual learning.

    Args:
        cfg: Configuration
    """
    # Get eval_only setting from config
    eval_only = cfg.get("eval_only", False)
    # Get distributed training configuration from Hydra config
    distributed_enabled = cfg.get("distributed", {}).get("enabled", False)
    world_size = cfg.get("distributed", {}).get("world_size", torch.cuda.device_count())

    # Print debug mode status if enabled
    debug_enabled = cfg.get("debug", {}).get("enabled", False)
    if debug_enabled:
        print("\n[DEBUG MODE ENABLED]")
        print(f"Debug settings:\n{OmegaConf.to_yaml(cfg.debug)}")

    # Run distributed training if enabled and multiple GPUs are available
    if distributed_enabled and world_size > 1:
        print(
            f"Running distributed {'evaluation' if eval_only else 'training'} on {world_size} GPUs"
        )
        mp.spawn(run_training, args=(world_size, cfg), nprocs=world_size, join=True)
    else:
        # Run on a single GPU or CPU
        print("Running on a single device")
        run_training(None, None, cfg)


if __name__ == "__main__":
    # Set multiprocessing start method to spawn for distributed training
    mp.set_start_method("spawn", force=True)
    main()
