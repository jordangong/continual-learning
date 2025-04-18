import os
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset

from src.data.datasets import (
    CIFAR100CL,
    CUB200CL,
    VTABCL,
    Caltech256CL,
    DomainNetCL,
    ImageNetACL,
    ImageNetRCL,
    MergedTaskDataset,
    ObjectNet200CL,
    ObjectNetCL,
    OmniBench300CL,
    OmniBenchCL,
    StanfordCarsCL,
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
        elif dataset_name == "caltech256":
            self.dataset = Caltech256CL(
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
        elif dataset_name == "stanford_cars":
            self.dataset = StanfordCarsCL(
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
        elif dataset_name == "domainnet":
            # For DomainNet, we need to determine if it's class-incremental or domain-incremental
            mode = self.dataset_config.get("mode", "class")
            domains = self.dataset_config.get("domains", None)

            self.dataset = DomainNetCL(
                root=self.data_dir,
                num_steps=self.continual_config["num_steps"],
                classes_per_step=self.continual_config["classes_per_step"],
                transform=self.train_transform,
                test_transform=self.test_transform,
                target_transform=None,
                download=True,
                seed=self.config["seed"],
                mode=mode,
                domains=domains,
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
        elif dataset_name == "objectnet200":
            self.dataset = ObjectNet200CL(
                root=self.data_dir,
                num_steps=self.continual_config["num_steps"],
                classes_per_step=self.continual_config["classes_per_step"],
                transform=self.train_transform,
                test_transform=self.test_transform,
                target_transform=None,
                download=True,
                seed=self.config["seed"],
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
        elif dataset_name == "omnibench300":
            self.dataset = OmniBench300CL(
                root=self.data_dir,
                num_steps=self.continual_config["num_steps"],
                classes_per_step=self.continual_config["classes_per_step"],
                transform=self.train_transform,
                test_transform=self.test_transform,
                target_transform=None,
                download=True,
                seed=self.config["seed"],
                class_order=None,  # Use random class order by default
            )
        elif dataset_name == "vtab":
            # Check if we're using the merged dataset option
            merged_task_subsets = self.dataset_config.get("merged_task_subsets", {})

            if merged_task_subsets and len(merged_task_subsets) > 0:
                # Using merged dataset with multiple tasks
                self.dataset = MergedTaskDataset(
                    root=self.data_dir,
                    num_steps=self.continual_config["num_steps"],
                    classes_per_step=self.continual_config["classes_per_step"],
                    task_subsets=merged_task_subsets,
                    transform=self.train_transform,
                    test_transform=self.test_transform,
                    target_transform=None,
                    download=True,
                    seed=self.config["seed"],
                )
            else:
                # Using single task VTAB dataset
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
            Tuple of (step_classes, train_loader, test_loader)
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

    def limit_data_loaders(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        debug_config: Dict[str, Any],
        distributed: bool = False,
        local_rank: int = -1,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Limit the number of batches in data loaders based on debug configuration.
        This function can be used in both training and evaluation modes.

        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            debug_config: Debug configuration dictionary
            distributed: Whether distributed training is enabled
            local_rank: Local rank for distributed training

        Returns:
            Tuple of (limited_train_loader, limited_test_loader)
        """
        debug_enabled = debug_config.get("enabled", False)

        if not debug_enabled:
            return train_loader, test_loader

        # Print debug information
        if not distributed or (distributed and local_rank == 0):
            print("\n[DEBUG MODE ENABLED]")
            print(f"Debug settings: {debug_config}")

        # Limit the number of batches if fast_dev_run is enabled
        if debug_config.get("fast_dev_run", False):
            # Limit the number of batches
            train_loader = self._limit_batches(train_loader, 3)  # Just 3 batches
            test_loader = self._limit_batches(test_loader, 3)  # Just 3 batches
            if not distributed or (distributed and local_rank == 0):
                print("[DEBUG] Fast dev run: Running only 1 epoch with 3 batches")
        else:
            # Apply batch limits if specified
            train_limit = debug_config.get("limit_train_batches", 1.0)
            val_limit = debug_config.get("limit_val_batches", 1.0)

            if train_limit < 1.0:
                train_loader = self._limit_batches(
                    train_loader, int(len(train_loader) * train_limit)
                )
                if not distributed or (distributed and local_rank == 0):
                    print(
                        f"[DEBUG] Limited training to {train_limit * 100}% of batches ({len(train_loader)} batches)"
                    )

            if val_limit < 1.0:
                test_loader = self._limit_batches(
                    test_loader, int(len(test_loader) * val_limit)
                )
                if not distributed or (distributed and local_rank == 0):
                    print(
                        f"[DEBUG] Limited validation to {val_limit * 100}% of batches ({len(test_loader)} batches)"
                    )

        return train_loader, test_loader

    def _limit_batches(self, data_loader: DataLoader, num_batches: int) -> DataLoader:
        """
        Limit the number of batches in a data loader.

        Args:
            data_loader: Data loader to limit
            num_batches: Number of batches to keep

        Returns:
            Limited data loader
        """
        # Create a subset of the dataset with only the first num_batches * batch_size samples
        batch_size = data_loader.batch_size
        subset_size = min(num_batches * batch_size, len(data_loader.dataset))

        # Create a subset of the dataset
        indices = list(range(subset_size))
        subset = Subset(data_loader.dataset, indices)

        # Create a new data loader with the subset
        limited_loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=data_loader.shuffle if hasattr(data_loader, "shuffle") else False,
            num_workers=(
                data_loader.num_workers if hasattr(data_loader, "num_workers") else 0
            ),
            persistent_workers=(
                data_loader.persistent_workers
                if hasattr(data_loader, "persistent_workers")
                else False
            ),
            pin_memory=(
                data_loader.pin_memory if hasattr(data_loader, "pin_memory") else False
            ),
            sampler=None,  # We're using a subset, so we don't need a sampler
            drop_last=(
                data_loader.drop_last if hasattr(data_loader, "drop_last") else False
            ),
        )

        return limited_loader
