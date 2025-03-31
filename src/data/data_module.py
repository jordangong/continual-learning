import os
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from src.data.datasets import (
    CIFAR100CL,
    CUB200CL,
    VTABCL,
    ImageNetACL,
    ImageNetRCL,
    ObjectNetCL,
    OmniBenchCL,
)
from src.data.transforms import get_transforms


class DataModule:
    """Data module for continual learning."""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.dataset_config = config["dataset"]
        self.continual_config = config["continual"]
        self.training_config = config["training"]
        self.data_dir = config["paths"]["data_dir"]

        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)

        # Setup transforms
        self.train_transform, self.test_transform = get_transforms(
            input_size=self.dataset_config["input_size"],
            mean=self.dataset_config["mean"],
            std=self.dataset_config["std"],
            augmentation=self.dataset_config.get("augmentation", None),
        )

        # Initialize dataset
        self.setup_dataset()

    def setup_dataset(self):
        """Setup the dataset based on the configuration."""
        dataset_name = self.dataset_config["name"].lower()

        if dataset_name == "cifar100":
            self.dataset = CIFAR100CL(
                root=self.data_dir,
                num_steps=self.continual_config["num_steps"],
                classes_per_step=self.continual_config["classes_per_step"],
                transform=self.train_transform,
                test_transform=self.test_transform,
                target_transform=None,
                download=True,
                seed=self.config["seed"],
            )
        elif dataset_name == "cub200":
            self.dataset = CUB200CL(
                root=self.data_dir,
                num_steps=self.continual_config["num_steps"],
                classes_per_step=self.continual_config["classes_per_step"],
                transform=self.train_transform,
                test_transform=self.test_transform,
                target_transform=None,
                download=True,
                seed=self.config["seed"],
            )
        elif dataset_name == "imagenet-r":
            self.dataset = ImageNetRCL(
                root=self.data_dir,
                num_steps=self.continual_config["num_steps"],
                classes_per_step=self.continual_config["classes_per_step"],
                transform=self.train_transform,
                test_transform=self.test_transform,
                target_transform=None,
                download=True,
                seed=self.config["seed"],
                train_ratio=self.dataset_config.get("train_ratio", 0.8),
            )
        elif dataset_name == "imagenet-a":
            self.dataset = ImageNetACL(
                root=self.data_dir,
                num_steps=self.continual_config["num_steps"],
                classes_per_step=self.continual_config["classes_per_step"],
                transform=self.train_transform,
                test_transform=self.test_transform,
                target_transform=None,
                download=True,
                seed=self.config["seed"],
                train_ratio=self.dataset_config.get("train_ratio", 0.8),
            )
        elif dataset_name == "objectnet":
            self.dataset = ObjectNetCL(
                root=self.data_dir,
                num_steps=self.continual_config["num_steps"],
                classes_per_step=self.continual_config["classes_per_step"],
                transform=self.train_transform,
                test_transform=self.test_transform,
                target_transform=None,
                download=True,
                seed=self.config["seed"],
                train_ratio=self.dataset_config.get("train_ratio", 0.8),
            )
        elif dataset_name == "omnibench":
            self.dataset = OmniBenchCL(
                root=self.data_dir,
                num_steps=self.continual_config["num_steps"],
                classes_per_step=self.continual_config["classes_per_step"],
                transform=self.train_transform,
                test_transform=self.test_transform,
                target_transform=None,
                download=True,
                seed=self.config["seed"],
                category=self.dataset_config.get("category", "all"),
                max_classes=self.dataset_config.get("max_classes", 300),
            )
        elif dataset_name == "vtab":
            self.dataset = VTABCL(
                root=self.data_dir,
                num_steps=self.continual_config["num_steps"],
                classes_per_step=self.continual_config["classes_per_step"],
                transform=self.train_transform,
                test_transform=self.test_transform,
                target_transform=None,
                download=True,
                seed=self.config["seed"],
                task=self.dataset_config.get("task", "cifar100"),
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    def get_data_loaders(
        self,
        step: int,
        memory_data: Optional[torch.utils.data.Dataset] = None,
        distributed_sampler: bool = False,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Get data loaders for a specific step.

        Args:
            step: Current continual learning step
            memory_data: Optional memory data to include in training
            distributed_sampler: Whether to use distributed sampler for multi-GPU training
            rank: Rank of the current process (for distributed training)
            world_size: Number of processes (for distributed training)

        Returns:
            Tuple of (train_loader, test_loader)
        """
        return self.dataset.get_data_loaders(
            step=step,
            batch_size=self.training_config["batch_size"],
            eval_batch_size=self.training_config["eval_batch_size"],
            num_workers=self.config["num_workers"],
            memory_data=memory_data,
            distributed_sampler=distributed_sampler,
            rank=rank,
            world_size=world_size,
        )

    def get_memory_samples(self, step: int) -> Optional[torch.utils.data.Dataset]:
        """
        Get memory samples for rehearsal.

        Args:
            step: Current step

        Returns:
            Dataset containing memory samples
        """
        if self.continual_config["strategy"] in ["replay", "er"]:
            return self.dataset.get_memory_samples(
                step=step, memory_size=self.continual_config["memory_size"]
            )
        return None
