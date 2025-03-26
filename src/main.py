import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import random
from typing import Optional

from src.data.data_module import DataModule
from src.models.model_factory import create_model
from src.trainers.trainer import ContinualTrainer
from src.utils.logging import setup_logger, plot_accuracy_curve, plot_forgetting_curve


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
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set the device
    torch.cuda.set_device(rank)


def run_training(rank: Optional[int], world_size: Optional[int], cfg: DictConfig) -> None:
    """
    Run the training process on a single process.
    
    Args:
        rank: Rank of the current process (None for single GPU)
        world_size: Number of processes (None for single GPU)
        cfg: Configuration
    """
    # Convert config to dictionary
    config = OmegaConf.to_container(cfg, resolve=True)
    
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
        logger = setup_logger(config["logging"]["log_dir"])
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
        cache_dir=cache_dir
    )
    
    # Setup trainer with distributed info if applicable
    trainer = ContinualTrainer(
        model=model,
        config=config,
        device=device,
        local_rank=rank if distributed else -1
    )
    
    # Train on each step
    num_steps = config["continual"]["num_steps"]
    accuracies = []
    forgetting_measures = []
    
    for step in range(num_steps):
        # Get data loaders for current step
        memory_data = data_module.get_memory_samples(step) if step > 0 else None
        train_loader, test_loader = data_module.get_data_loaders(
            step, 
            memory_data, 
            distributed_sampler=distributed,
            rank=rank,
            world_size=world_size
        )
        
        # Train on current step
        metrics = trainer.train_step(step, train_loader, test_loader)
        
        # Record metrics
        accuracies.append(metrics["accuracy"])
        if step > 0:
            forgetting_measures.append(metrics["forgetting"])
        
        # Log metrics (only on main process if distributed)
        if not distributed or (distributed and rank == 0):
            if logger:
                logger.info(f"Step {step+1}/{num_steps} - Accuracy: {metrics['accuracy']:.2f}%")
                if step > 0:
                    logger.info(f"Step {step+1}/{num_steps} - Forgetting: {metrics['forgetting']:.4f}")
    
    # Only create plots and log final metrics on main process if distributed
    if not distributed or (distributed and rank == 0):
        # Create plots directory
        plots_dir = os.path.join(config["paths"]["output_root"], "plots")
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
                logger.info(f"Final average forgetting: {np.mean(forgetting_measures):.4f}")
        
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
        print(f"Running distributed training on {world_size} GPUs")
        mp.spawn(
            run_training,
            args=(world_size, cfg),
            nprocs=world_size,
            join=True
        )
    else:
        # Run on a single GPU or CPU
        print("Running on a single device")
        run_training(None, None, cfg)


if __name__ == "__main__":
    # Set multiprocessing start method to spawn for distributed training
    mp.set_start_method('spawn', force=True)
    main()
