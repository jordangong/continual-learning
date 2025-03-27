import os
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

def setup_logger(log_dir: str, name: str = "continual_learning") -> logging.Logger:
    """
    Setup logger for the project.
    
    Args:
        log_dir: Directory to save logs
        name: Logger name
        
    Returns:
        Logger instance
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Clear any existing handlers to avoid duplication
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Prevent propagation to the root logger to avoid duplicate logs
    logger.propagate = False
    
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create file handler
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def plot_accuracy_curve(
    accuracies: List[float],
    save_path: Optional[str] = None,
    title: str = "Accuracy Curve"
) -> None:
    """
    Plot accuracy curve.
    
    Args:
        accuracies: List of accuracies for each step
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o')
    plt.xlabel("Step")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.grid(True)
    
    if save_path is not None:
        plt.savefig(save_path)
    
    plt.close()

def plot_forgetting_curve(
    forgetting: List[float],
    save_path: Optional[str] = None,
    title: str = "Forgetting Curve"
) -> None:
    """
    Plot forgetting curve.
    
    Args:
        forgetting: List of forgetting measures for each step
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, len(forgetting) + 2), forgetting, marker='o')
    plt.xlabel("Step")
    plt.ylabel("Forgetting")
    plt.title(title)
    plt.grid(True)
    
    if save_path is not None:
        plt.savefig(save_path)
    
    plt.close()

# Note: The following W&B logging functions were removed as they were unused in the codebase:
# - log_metrics_to_wandb
# - log_model_to_wandb
# - log_confusion_matrix
# - log_images_to_wandb
