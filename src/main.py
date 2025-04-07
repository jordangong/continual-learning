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

from src.data.data_module import DataModule
from src.models.model_factory import create_model
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

    # Setup data module
    data_module = DataModule(config)

    # Setup model
    cache_dir = config["paths"].get("cache_dir", None)
    model = create_model(
        model_config=config["model"],
        num_classes=config["dataset"]["num_classes"],
        device=device,
        cache_dir=cache_dir,
    )

    # Setup trainer with distributed info if applicable
    trainer = ContinualTrainer(
        model=model,
        config=config,
        device=device,
        local_rank=rank if distributed else -1,
    )

    # Train or evaluate on each step
    num_steps = config["continual"]["num_steps"]
    accuracies = []
    forgetting_measures = []

    # If eval_only is True, we'll only evaluate the model without training
    if eval_only and not distributed or (eval_only and distributed and rank == 0):
        print("Running in evaluation-only mode. Loading model from checkpoint...")

    for step in range(num_steps):
        # Get data loaders for current step
        memory_data = data_module.get_memory_samples(step) if step > 0 else None
        train_loader, test_loader = data_module.get_data_loaders(
            step,
            memory_data,
            distributed_sampler=distributed,
            rank=rank,
            world_size=world_size,
        )

        # Initialize prototypes if using prototypical classifier
        if config["model"]["classifier"]["type"] == "prototypical":
            # Only print on main process if distributed
            if not distributed or (distributed and rank == 0):
                print(f"\n=== Initializing prototypes for step {step} ===")

            # Get the correct model reference
            model_to_use = model.module if distributed else model

            # For incremental learning, only reset prototypes on the first step
            # This ensures we maintain prototypes for previous classes
            reset_prototypes = step == 0

            # Initialize prototypes from training data
            model_to_use.init_prototypes_from_data(train_loader, reset=reset_prototypes)

        # Train or evaluate on current step
        start_time = time.time()

        if eval_only:
            if config["continual"]["strategy"] != "zeroshot":
                # Load model from checkpoint and evaluate
                trainer._load_checkpoint(step, best=True)

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
            metrics = trainer.train_step(step, train_loader, test_loader)
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
