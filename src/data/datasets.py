import glob
import json
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.datasets as tv_datasets
from PIL import Image
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    DistributedSampler,
    Subset,
)
from torchvision.datasets import ImageFolder


class ContinualDataset:
    """Base class for continual learning datasets."""

    def __init__(
        self,
        root: str,
        num_steps: int,
        classes_per_step: int,
        transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            root: Root directory for dataset
            num_steps: Number of continual learning steps
            classes_per_step: Number of classes per step
            transform: Transform to apply to training data
            test_transform: Transform to apply to test data (if None, uses transform)
            target_transform: Transform to apply to targets
            download: Whether to download the dataset
            seed: Random seed for reproducibility
        """
        self.root = root
        self.num_steps = num_steps
        self.classes_per_step = classes_per_step
        self.transform = transform
        self.test_transform = (
            test_transform if test_transform is not None else transform
        )
        self.target_transform = target_transform
        self.download = download
        self.seed = seed

        # To be set by child classes
        self.dataset_train = None
        self.dataset_test = None
        self.num_classes = None
        self.class_order = None

        # Cache for dataset indices by class
        self._class_indices_cache = {}
        self._cache_lock = threading.Lock()
        self._cache_initialized = {}

        # Set random seed for reproducibility
        np.random.seed(self.seed)

    def setup(self):
        """Setup the dataset. Should be implemented by child classes."""
        raise NotImplementedError

    def get_data_loaders(
        self,
        step: int,
        batch_size: int,
        eval_batch_size: Optional[int] = None,
        num_workers: int = 4,
        memory_data: Optional[Dataset] = None,
        distributed_sampler: bool = False,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ) -> Tuple[DataLoader, DataLoader]:
        """Get data loaders for a specific step.

        Args:
            step: Current continual learning step
            batch_size: Batch size for data loaders
            eval_batch_size: Batch size for evaluation
            num_workers: Number of workers for data loaders
            memory_data: Optional memory data to include in training
            distributed_sampler: Whether to use distributed sampler for multi-GPU training
            rank: Rank of the current process (for distributed training)
            world_size: Number of processes (for distributed training)

        Returns:
            Tuple of (step_classes, train_loader, test_loader)
        """
        if step >= self.num_steps:
            raise ValueError(
                f"Step {step} exceeds the number of steps {self.num_steps}"
            )

        # Get class indices for current step
        step_classes = self._get_step_classes(step)

        # Get train indices for current step
        train_indices = self._get_indices_for_classes(self.dataset_train, step_classes)
        train_dataset = Subset(self.dataset_train, train_indices)

        # Combine with memory data if provided
        if memory_data is not None:
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, memory_data])

        # Get test indices for all classes seen so far
        seen_classes = []
        for i in range(step + 1):
            seen_classes.extend(self._get_step_classes(i))
        test_indices = self._get_indices_for_classes(self.dataset_test, seen_classes)
        test_dataset = Subset(self.dataset_test, test_indices)

        # Create data loaders with distributed sampler if needed
        if distributed_sampler and rank is not None and world_size is not None:
            # Create distributed samplers
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=False,
            )

            test_sampler = DistributedSampler(
                test_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,  # Sampler handles shuffling
                num_workers=num_workers,
                persistent_workers=True,
                sampler=train_sampler,
                pin_memory=True,
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=(batch_size if eval_batch_size is None else eval_batch_size),
                shuffle=False,
                num_workers=num_workers,
                persistent_workers=True,
                sampler=test_sampler,
                pin_memory=True,
            )
        else:
            # Regular data loaders for single GPU
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                persistent_workers=True,
                pin_memory=True,
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=(batch_size if eval_batch_size is None else eval_batch_size),
                shuffle=False,
                num_workers=num_workers,
                persistent_workers=True,
                pin_memory=True,
            )

        return step_classes, train_loader, test_loader

    def _get_step_classes(self, step: int) -> List[int]:
        """Get classes for a specific step."""
        start_idx = step * self.classes_per_step
        end_idx = min((step + 1) * self.classes_per_step, self.num_classes)
        return self.class_order[start_idx:end_idx]

    def _initialize_class_indices_cache(
        self, dataset: Dataset, num_workers: int = 4
    ) -> None:
        """Initialize the class indices cache for a dataset using multithreading.

        Args:
            dataset: The dataset to scan
            num_workers: Number of threads to use for scanning
        """
        dataset_id = id(dataset)
        if (
            dataset_id in self._cache_initialized
            and self._cache_initialized[dataset_id]
        ):
            return

        with self._cache_lock:
            # Check again in case another thread initialized the cache while we were waiting
            if (
                dataset_id in self._cache_initialized
                and self._cache_initialized[dataset_id]
            ):
                return

            print(f"Initializing class indices cache for dataset {dataset_id}...")

            # Create a defaultdict to store indices by class
            class_indices = defaultdict(list)
            dataset_size = len(dataset)

            # Function to process a chunk of the dataset
            def process_chunk(start_idx, end_idx):
                chunk_indices = defaultdict(list)
                for i in range(start_idx, min(end_idx, dataset_size)):
                    _, target = dataset[i]
                    chunk_indices[target].append(i)
                return chunk_indices

            # Split the dataset into chunks for parallel processing
            chunk_size = max(1, dataset_size // num_workers)
            chunks = [
                (i, min(i + chunk_size, dataset_size))
                for i in range(0, dataset_size, chunk_size)
            ]

            # Process chunks in parallel
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                chunk_results = list(executor.map(lambda x: process_chunk(*x), chunks))

            # Merge results from all chunks
            for chunk_result in chunk_results:
                for target, indices in chunk_result.items():
                    class_indices[target].extend(indices)

            # Store the cache
            self._class_indices_cache[dataset_id] = dict(class_indices)
            self._cache_initialized[dataset_id] = True
            print(
                f"Class indices cache initialized for dataset {dataset_id} with {len(class_indices)} classes"
            )

    def _get_indices_for_classes(
        self, dataset: Dataset, classes: List[int]
    ) -> List[int]:
        """Get indices of samples belonging to specified classes.

        Uses a cached mapping of classes to indices for efficiency with large datasets.
        The cache is initialized on the first call using multithreading.
        """
        dataset_id = id(dataset)

        # Initialize cache if not already done
        if not hasattr(self, "_class_indices_cache"):
            self._class_indices_cache = {}
            self._cache_lock = threading.Lock()
            self._cache_initialized = {}

        if (
            dataset_id not in self._cache_initialized
            or not self._cache_initialized[dataset_id]
        ):
            self._initialize_class_indices_cache(dataset)

        # Get indices from cache
        indices = []
        for cls in classes:
            if cls in self._class_indices_cache[dataset_id]:
                indices.extend(self._class_indices_cache[dataset_id][cls])

        return indices

    def get_memory_samples(self, step: int, memory_size: int) -> Optional[Dataset]:
        """Get memory samples for rehearsal.

        Args:
            step: Current step (we get memory from previous steps)
            memory_size: Total size of memory buffer

        Returns:
            Dataset containing memory samples
        """
        if step == 0:
            return None

        # Get classes from previous steps
        prev_classes = []
        for i in range(step):
            prev_classes.extend(self._get_step_classes(i))

        # Get indices for previous classes
        indices = self._get_indices_for_classes(self.dataset_train, prev_classes)

        # Calculate samples per class
        samples_per_class = memory_size // len(prev_classes)

        # Group indices by class
        class_indices = {c: [] for c in prev_classes}
        for idx in indices:
            _, target = self.dataset_train[idx]
            class_indices[target].append(idx)

        # Sample indices for memory
        memory_indices = []
        for c in prev_classes:
            c_indices = class_indices[c]
            # Randomly sample indices for this class
            if len(c_indices) > samples_per_class:
                c_indices = np.random.choice(
                    c_indices, samples_per_class, replace=False
                ).tolist()
            memory_indices.extend(c_indices)

        # Create memory dataset
        memory_dataset = Subset(self.dataset_train, memory_indices)
        return memory_dataset


class CIFAR100CL(ContinualDataset):
    """CIFAR-100 dataset for continual learning."""

    def __init__(
        self,
        root: str,
        num_steps: int,
        classes_per_step: int,
        transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        seed: int = 42,
        class_order: Optional[List[int]] = None,
    ):
        super().__init__(
            root=root,
            num_steps=num_steps,
            classes_per_step=classes_per_step,
            transform=transform,
            test_transform=test_transform,
            target_transform=target_transform,
            download=download,
            seed=seed,
        )

        self.num_classes = 100

        # Set class order (either provided or random)
        if class_order is None:
            self.class_order = list(range(self.num_classes))
            np.random.shuffle(self.class_order)
        else:
            assert len(class_order) == self.num_classes, (
                "Class order must contain all classes"
            )
            self.class_order = class_order

        self.setup()

    def setup(self):
        """Setup CIFAR-100 dataset."""
        self.dataset_train = tv_datasets.CIFAR100(
            root=self.root,
            train=True,
            transform=self.transform,
            target_transform=self.target_transform,
            download=self.download,
        )

        self.dataset_test = tv_datasets.CIFAR100(
            root=self.root,
            train=False,
            transform=self.test_transform,
            target_transform=self.target_transform,
            download=self.download,
        )


class CUB200CL(ContinualDataset):
    """CUB-200-2011 dataset for continual learning."""

    def __init__(
        self,
        root: str,
        num_steps: int,
        classes_per_step: int,
        transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        seed: int = 42,
        class_order: Optional[List[int]] = None,
    ):
        super().__init__(
            root=root,
            num_steps=num_steps,
            classes_per_step=classes_per_step,
            transform=transform,
            test_transform=test_transform,
            target_transform=target_transform,
            download=download,
            seed=seed,
        )

        self.num_classes = 200

        # Set class order (either provided or random)
        if class_order is None:
            self.class_order = list(range(self.num_classes))
            np.random.shuffle(self.class_order)
        else:
            assert len(class_order) == self.num_classes, (
                "Class order must contain all classes"
            )
            self.class_order = class_order

        self.setup()

    def setup(self):
        """Setup CUB-200-2011 dataset."""
        # CUB-200-2011 dataset structure:
        # root/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg
        # We need to create a dataset that follows the ImageFolder structure

        # Check if dataset exists
        dataset_path = os.path.join(self.root, "CUB_200_2011")
        if not os.path.exists(dataset_path) and self.download:
            raise ValueError(
                "CUB-200-2011 dataset must be downloaded manually. "
                "Please download from https://www.vision.caltech.edu/datasets/cub_200_2011/ "
                "and extract to {}".format(dataset_path)
            )

        # Setup train/test split (use the standard split from the dataset)
        images_path = os.path.join(dataset_path, "images")
        split_file = os.path.join(dataset_path, "train_test_split.txt")
        image_id_file = os.path.join(dataset_path, "images.txt")
        class_file = os.path.join(dataset_path, "image_class_labels.txt")

        # Read split information
        with open(split_file, "r") as f:
            split_lines = f.readlines()
            split_dict = {
                int(line.strip().split()[0]): int(line.strip().split()[1])
                for line in split_lines
            }

        # Read image paths
        with open(image_id_file, "r") as f:
            image_lines = f.readlines()
            image_dict = {
                int(line.strip().split()[0]): line.strip().split()[1]
                for line in image_lines
            }

        # Read class labels
        with open(class_file, "r") as f:
            class_lines = f.readlines()
            class_dict = {
                int(line.strip().split()[0]): int(line.strip().split()[1]) - 1
                for line in class_lines
            }

        # Create train and test datasets
        train_images = []
        train_targets = []
        test_images = []
        test_targets = []

        for img_id, is_train in split_dict.items():
            img_path = os.path.join(images_path, image_dict[img_id])
            target = class_dict[img_id]

            if is_train == 1:
                train_images.append(img_path)
                train_targets.append(target)
            else:
                test_images.append(img_path)
                test_targets.append(target)

        self.dataset_train = ImageListDataset(
            train_images, train_targets, transform=self.transform
        )
        self.dataset_test = ImageListDataset(
            test_images, test_targets, transform=self.test_transform
        )


class Caltech256CL(ContinualDataset):
    """Caltech256 dataset for continual learning."""

    def __init__(
        self,
        root: str,
        num_steps: int,
        classes_per_step: int,
        transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        seed: int = 42,
        class_order: Optional[List[int]] = None,
        train_ratio: float = 0.8,  # Caltech256 doesn't have a train/test split, so we create one
    ):
        super().__init__(
            root=root,
            num_steps=num_steps,
            classes_per_step=classes_per_step,
            transform=transform,
            test_transform=test_transform,
            target_transform=target_transform,
            download=download,
            seed=seed,
        )

        self.train_ratio = train_ratio

        # Caltech256 has 256 classes (excluding clutter class)
        self.num_classes = 256

        # Set class order (either provided or random)
        if class_order is None:
            self.class_order = list(range(self.num_classes))
            np.random.shuffle(self.class_order)
        else:
            assert len(class_order) == self.num_classes, (
                "Class order must contain all classes"
            )
            self.class_order = class_order

        self.setup()

    def setup(self):
        """Setup Caltech256 dataset excluding the clutter class (class 256)."""
        # Load the dataset
        dataset = tv_datasets.Caltech256(
            root=self.root,
            transform=None,  # We'll apply transforms later
            target_transform=self.target_transform,
            download=self.download,
        )

        # Filter out the clutter class (class 256)
        filtered_indices = [i for i, (_, target) in enumerate(dataset) if target != 256]

        # Create train/test split from filtered indices
        np.random.shuffle(filtered_indices)
        split_idx = int(len(filtered_indices) * self.train_ratio)

        train_indices = filtered_indices[:split_idx]
        test_indices = filtered_indices[split_idx:]

        # Create subsets with appropriate transforms
        self.dataset_train = TransformSubset(dataset, train_indices, self.transform)
        self.dataset_test = TransformSubset(dataset, test_indices, self.test_transform)


class TransformSubset(Dataset):
    """Subset of a dataset with a different transform."""

    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, target = self.dataset[self.indices[idx]]

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class StanfordCarsCL(ContinualDataset):
    """Stanford Cars dataset for continual learning."""

    def __init__(
        self,
        root: str,
        num_steps: int,
        classes_per_step: int,
        transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        seed: int = 42,
        class_order: Optional[List[int]] = None,
    ):
        super().__init__(
            root=root,
            num_steps=num_steps,
            classes_per_step=classes_per_step,
            transform=transform,
            test_transform=test_transform,
            target_transform=target_transform,
            download=download,
            seed=seed,
        )

        # Stanford Cars has 196 classes
        self.num_classes = 196

        # Set class order (either provided or random)
        if class_order is None:
            self.class_order = list(range(self.num_classes))
            np.random.shuffle(self.class_order)
        else:
            assert len(class_order) == self.num_classes, (
                "Class order must contain all classes"
            )
            self.class_order = class_order

        self.setup()

    def setup(self):
        """Setup Stanford Cars dataset."""
        # Load the train and test datasets
        self.dataset_train = tv_datasets.StanfordCars(
            root=self.root,
            split="train",
            transform=self.transform,
            target_transform=self.target_transform,
            download=self.download,
        )

        self.dataset_test = tv_datasets.StanfordCars(
            root=self.root,
            split="test",
            transform=self.test_transform,
            target_transform=self.target_transform,
            download=self.download,
        )


class ImageNetRCL(ContinualDataset):
    """ImageNet-R dataset for continual learning."""

    def __init__(
        self,
        root: str,
        num_steps: int,
        classes_per_step: int,
        transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        seed: int = 42,
        class_order: Optional[List[int]] = None,
        train_ratio: float = 0.8,  # ImageNet-R doesn't have a train/test split, so we create one
    ):
        super().__init__(
            root=root,
            num_steps=num_steps,
            classes_per_step=classes_per_step,
            transform=transform,
            test_transform=test_transform,
            target_transform=target_transform,
            download=download,
            seed=seed,
        )

        self.num_classes = 200  # ImageNet-R has 200 classes
        self.train_ratio = train_ratio

        # Set class order (either provided or random)
        if class_order is None:
            self.class_order = list(range(self.num_classes))
            np.random.shuffle(self.class_order)
        else:
            assert len(class_order) == self.num_classes, (
                "Class order must contain all classes"
            )
            self.class_order = class_order

        self.setup()

    def setup(self):
        """Setup ImageNet-R dataset."""
        # Check if dataset exists
        dataset_path = os.path.join(self.root, "imagenet-r")
        if not os.path.exists(dataset_path) and self.download:
            raise ValueError(
                "ImageNet-R dataset must be downloaded manually. "
                "Please download from https://github.com/hendrycks/imagenet-r "
                "and extract to {}".format(dataset_path)
            )

        # ImageNet-R follows the ImageFolder structure, but doesn't have a train/test split
        # We'll create our own split
        train_dataset = ImageFolder(dataset_path, transform=self.transform)
        test_dataset = ImageFolder(dataset_path, transform=self.test_transform)

        # Create train/test split
        class_indices = defaultdict(list)
        for idx, (_, target) in enumerate(train_dataset.samples):
            class_indices[target].append(idx)

        train_indices = []
        test_indices = []

        # For each class, split into train and test
        for target, indices in class_indices.items():
            np.random.shuffle(indices)
            split_idx = int(len(indices) * self.train_ratio)
            train_indices.extend(indices[:split_idx])
            test_indices.extend(indices[split_idx:])

        # Create train and test datasets
        self.dataset_train = Subset(train_dataset, train_indices)
        self.dataset_test = Subset(test_dataset, test_indices)


class ImageNetACL(ContinualDataset):
    """ImageNet-A dataset for continual learning."""

    def __init__(
        self,
        root: str,
        num_steps: int,
        classes_per_step: int,
        transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        seed: int = 42,
        class_order: Optional[List[int]] = None,
        train_ratio: float = 0.8,  # ImageNet-A doesn't have a train/test split, so we create one
    ):
        super().__init__(
            root=root,
            num_steps=num_steps,
            classes_per_step=classes_per_step,
            transform=transform,
            test_transform=test_transform,
            target_transform=target_transform,
            download=download,
            seed=seed,
        )

        self.num_classes = 200  # ImageNet-A has 200 adversarial classes
        self.train_ratio = train_ratio

        # Set class order (either provided or random)
        if class_order is None:
            self.class_order = list(range(self.num_classes))
            np.random.shuffle(self.class_order)
        else:
            assert len(class_order) == self.num_classes, (
                "Class order must contain all classes"
            )
            self.class_order = class_order

        self.setup()

    def setup(self):
        """Setup ImageNet-A dataset."""
        # Check if dataset exists
        dataset_path = os.path.join(self.root, "imagenet-a")
        if not os.path.exists(dataset_path) and self.download:
            raise ValueError(
                "ImageNet-A dataset must be downloaded manually. "
                "Please download from https://github.com/hendrycks/natural-adv-examples "
                "and extract to {}".format(dataset_path)
            )

        # ImageNet-A follows the ImageFolder structure, but doesn't have a train/test split
        # We'll create our own split
        train_dataset = ImageFolder(dataset_path, transform=self.transform)
        test_dataset = ImageFolder(dataset_path, transform=self.test_transform)

        # Create train/test split
        class_indices = defaultdict(list)
        for idx, (_, target) in enumerate(train_dataset.samples):
            class_indices[target].append(idx)

        train_indices = []
        test_indices = []

        # For each class, split into train and test
        for target, indices in class_indices.items():
            np.random.shuffle(indices)
            split_idx = int(len(indices) * self.train_ratio)
            train_indices.extend(indices[:split_idx])
            test_indices.extend(indices[split_idx:])

        # Create train and test datasets
        self.dataset_train = Subset(train_dataset, train_indices)
        self.dataset_test = Subset(test_dataset, test_indices)


class DomainNetCL(ContinualDataset):
    """DomainNet dataset for continual learning.

    Supports both class-incremental and domain-incremental learning.

    DomainNet has 6 domains:
    - clipart: Clipart images
    - infograph: Infographic images
    - painting: Painting images
    - quickdraw: Hand-drawn sketches
    - real: Real-world images (photos)
    - sketch: Sketch images

    For class-incremental learning, we use all domains but split classes.
    For domain-incremental learning, we use all classes but split domains.
    """

    def __init__(
        self,
        root: str,
        num_steps: int,
        classes_per_step: int,
        transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        seed: int = 42,
        class_order: Optional[List[int]] = None,
        mode: str = "class",  # "class" or "domain"
        domains: Optional[List[str]] = None,
    ):
        """
        Args:
            root: Root directory for dataset
            num_steps: Number of continual learning steps
            classes_per_step: Number of classes per step (for class-incremental)
                              or number of domains per step (for domain-incremental)
            transform: Transform to apply to training data
            test_transform: Transform to apply to test data (if None, uses transform)
            target_transform: Transform to apply to targets
            download: Whether to download the dataset (not implemented)
            seed: Random seed for reproducibility
            class_order: Specific order of classes (if None, random order is used)
            mode: Learning mode, either "class" for class-incremental or "domain" for domain-incremental
            domains: List of domains to use (if None, uses all domains)
        """
        super().__init__(
            root=root,
            num_steps=num_steps,
            classes_per_step=classes_per_step,
            transform=transform,
            test_transform=test_transform,
            target_transform=target_transform,
            download=download,
            seed=seed,
        )

        # Available domains in DomainNet
        self.available_domains = [
            "clipart",
            "infograph",
            "painting",
            "quickdraw",
            "real",
            "sketch",
        ]

        # Validate and set domains to use
        if domains is None:
            self.domains = self.available_domains
        else:
            for domain in domains:
                if domain not in self.available_domains:
                    raise ValueError(
                        f"Invalid domain: {domain}. Available domains: {self.available_domains}"
                    )
            self.domains = domains

        # Set mode (class-incremental or domain-incremental)
        if mode not in ["class", "domain"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'class' or 'domain'")
        self.mode = mode

        # For domain-incremental learning, ensure num_steps is valid
        if self.mode == "domain":
            if num_steps > len(self.domains):
                raise ValueError(
                    f"For domain-incremental learning, num_steps ({num_steps}) cannot exceed the number of domains ({len(self.domains)})"
                )

        # DomainNet has 345 classes
        self.num_classes = 345 if self.mode == "class" else len(self.domains)

        # Set class order (either provided or random)
        if class_order is None:
            self.class_order = list(range(self.num_classes))
            np.random.shuffle(self.class_order)
        else:
            assert len(class_order) == self.num_classes, (
                f"Class order must contain all {self.num_classes} classes"
            )
            self.class_order = class_order

        # For domain-incremental learning, create domain order
        if self.mode == "domain":
            domain_indices = list(range(len(self.domains)))
            np.random.shuffle(domain_indices)
            self.domain_order = [self.domains[i] for i in domain_indices]

        self.setup()

    def setup(self):
        """Setup DomainNet dataset."""
        # Check if dataset exists
        dataset_path = os.path.join(self.root, "domainnet")
        if not os.path.exists(dataset_path):
            raise ValueError(
                f"DomainNet dataset not found at {dataset_path}. "
                "Please download the dataset manually."
            )

        # For class-incremental learning, we load all domains
        if self.mode == "class":
            self._setup_class_incremental(dataset_path)
        # For domain-incremental learning, we load all classes but split domains
        else:
            self._setup_domain_incremental(dataset_path)

    def _setup_class_incremental(self, dataset_path):
        """Setup for class-incremental learning (all domains, split classes)."""
        train_images, train_targets = [], []
        test_images, test_targets = [], []

        # Load data from all domains
        for domain in self.domains:
            domain_train_images, domain_train_targets = self._load_domain_data(
                dataset_path, domain, "train"
            )
            domain_test_images, domain_test_targets = self._load_domain_data(
                dataset_path, domain, "test"
            )

            train_images.extend(domain_train_images)
            train_targets.extend(domain_train_targets)
            test_images.extend(domain_test_images)
            test_targets.extend(domain_test_targets)

        # Create datasets
        self.dataset_train = ImageListDataset(
            train_images, train_targets, transform=self.transform
        )
        self.dataset_test = ImageListDataset(
            test_images, test_targets, transform=self.test_transform
        )

    def _setup_domain_incremental(self, dataset_path):
        """Setup for domain-incremental learning (all classes, split domains)."""
        # For domain-incremental learning, we create a separate dataset for each domain
        # and relabel classes to be domain IDs

        train_images, train_targets = [], []
        test_images, test_targets = [], []

        # Load data from all domains, but relabel targets to be domain IDs
        for domain_idx, domain in enumerate(self.domains):
            domain_train_images, _ = self._load_domain_data(
                dataset_path, domain, "train"
            )
            domain_test_images, _ = self._load_domain_data(dataset_path, domain, "test")

            # Assign domain index as the target
            domain_train_targets = [domain_idx] * len(domain_train_images)
            domain_test_targets = [domain_idx] * len(domain_test_images)

            train_images.extend(domain_train_images)
            train_targets.extend(domain_train_targets)
            test_images.extend(domain_test_images)
            test_targets.extend(domain_test_targets)

        # Create datasets
        self.dataset_train = ImageListDataset(
            train_images, train_targets, transform=self.transform
        )
        self.dataset_test = ImageListDataset(
            test_images, test_targets, transform=self.test_transform
        )

    def _load_domain_data(self, dataset_path, domain, split):
        """Load data for a specific domain and split.

        Args:
            dataset_path: Path to the DomainNet dataset
            domain: Domain name
            split: Either "train" or "test"

        Returns:
            Tuple of (images, targets)
        """
        split_file = os.path.join(dataset_path, f"{domain}_{split}.txt")

        if not os.path.exists(split_file):
            raise ValueError(f"Split file not found: {split_file}")

        images, targets = [], []

        with open(split_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 2:
                    continue

                image_path, target = parts
                image_path = os.path.join(dataset_path, image_path)
                target = int(target)

                # Skip if image doesn't exist
                if not os.path.exists(image_path):
                    continue

                images.append(image_path)
                targets.append(target)

        return images, targets

    def _get_step_classes(self, step):
        """Get classes for a specific step.

        For class-incremental learning, returns a subset of classes.
        For domain-incremental learning, returns domains for the step.
        """
        if self.mode == "class":
            # Class-incremental: return subset of classes
            start_idx = step * self.classes_per_step
            end_idx = min((step + 1) * self.classes_per_step, self.num_classes)
            return [self.class_order[i] for i in range(start_idx, end_idx)]
        else:
            # Domain-incremental: return domains for this step
            start_idx = step * self.classes_per_step
            end_idx = min((step + 1) * self.classes_per_step, len(self.domains))
            domain_indices = [self.class_order[i] for i in range(start_idx, end_idx)]
            return domain_indices


class ObjectNetCL(ContinualDataset):
    """ObjectNet dataset for continual learning."""

    def __init__(
        self,
        root: str,
        num_steps: int,
        classes_per_step: int,
        transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        seed: int = 42,
        class_order: Optional[List[int]] = None,
        train_ratio: float = 0.8,  # ObjectNet doesn't have a train/test split, so we create one
    ):
        super().__init__(
            root=root,
            num_steps=num_steps,
            classes_per_step=classes_per_step,
            transform=transform,
            test_transform=test_transform,
            target_transform=target_transform,
            download=download,
            seed=seed,
        )

        self.train_ratio = train_ratio

        # ObjectNet has 313 classes, but we'll use the 113 that overlap with ImageNet
        self.num_classes = 113

        # Set class order (either provided or random)
        if class_order is None:
            self.class_order = list(range(self.num_classes))
            np.random.shuffle(self.class_order)
        else:
            assert len(class_order) == self.num_classes, (
                "Class order must contain all classes"
            )
            self.class_order = class_order

        self.setup()

    def setup(self):
        """Setup ObjectNet dataset."""
        # Check if dataset exists
        dataset_path = os.path.join(self.root, "objectnet-1.0")
        if not os.path.exists(dataset_path) and self.download:
            raise ValueError(
                "ObjectNet dataset must be downloaded manually. "
                "Please download from https://objectnet.dev/ "
                "and extract to {}".format(dataset_path)
            )

        # ObjectNet has a different structure than ImageFolder
        # We need to create a mapping from folders to classes
        images_path = os.path.join(dataset_path, "images")
        mapping_file = os.path.join(
            dataset_path, "mappings", "folder_to_objectnet_label.json"
        )

        # Load the mapping
        with open(mapping_file, "r") as f:
            folder_to_label = json.load(f)

        # Create a mapping from label to index
        unique_labels = sorted(set(folder_to_label.values()))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

        # Collect all images and their targets
        all_images = []
        all_targets = []

        for folder, label in folder_to_label.items():
            folder_path = os.path.join(images_path, folder)
            if os.path.exists(folder_path):
                for img_path in glob.glob(os.path.join(folder_path, "*.png")):
                    all_images.append(img_path)
                    all_targets.append(label_to_idx[label])

        # Create train/test split
        indices = list(range(len(all_images)))
        np.random.shuffle(indices)
        split_idx = int(len(indices) * self.train_ratio)

        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        train_images = [all_images[i] for i in train_indices]
        train_targets = [all_targets[i] for i in train_indices]

        test_images = [all_images[i] for i in test_indices]
        test_targets = [all_targets[i] for i in test_indices]

        # Create custom datasets with remove_border=True to remove the 1-pixel red border
        # as mentioned in the ObjectNet documentation: https://objectnet.dev/download.html
        self.dataset_train = ImageListDataset(
            train_images, train_targets, transform=self.transform, remove_border=True
        )
        self.dataset_test = ImageListDataset(
            test_images, test_targets, transform=self.test_transform, remove_border=True
        )


class ObjectNet200CL(ContinualDataset):
    """ObjectNet-200 dataset for continual learning.

    This class implements a subset of ObjectNet with 200 classes, using provided split files.
    """

    def __init__(
        self,
        root: str,
        num_steps: int,
        classes_per_step: int,
        transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        seed: int = 42,
        class_order: Optional[List[int]] = None,
    ):
        super().__init__(
            root=root,
            num_steps=num_steps,
            classes_per_step=classes_per_step,
            transform=transform,
            test_transform=test_transform,
            target_transform=target_transform,
            download=download,
            seed=seed,
        )

        # ObjectNet-200 has 200 classes
        self.num_classes = 200

        # Set class order (either provided or random)
        if class_order is None:
            self.class_order = list(range(self.num_classes))
            np.random.shuffle(self.class_order)
        else:
            assert len(class_order) == self.num_classes, (
                "Class order must contain all classes"
            )
            self.class_order = class_order

        self.setup()

    def setup(self):
        """Setup ObjectNet-200 dataset using the provided split files."""
        # Check if dataset exists
        dataset_path = os.path.join(self.root, "objectnet-1.0")
        if not os.path.exists(dataset_path) and self.download:
            raise ValueError(
                "ObjectNet dataset must be downloaded manually. "
                "Please download from https://objectnet.dev/ "
                "and extract to {}".format(dataset_path)
            )

        # Check if split files exist
        split_dir = os.path.join(self.root, "objectnet-200")
        train_split_file = os.path.join(split_dir, "train.txt")
        test_split_file = os.path.join(split_dir, "test.txt")

        if not os.path.exists(train_split_file) or not os.path.exists(test_split_file):
            raise ValueError(
                "ObjectNet-200 split files not found. "
                "Please ensure the files are in {}".format(split_dir)
            )

        # Load train and test splits
        train_images, train_targets = self._load_split_data(
            train_split_file, dataset_path
        )
        test_images, test_targets = self._load_split_data(test_split_file, dataset_path)

        # Create custom datasets with remove_border=True to remove the 1-pixel red border
        self.dataset_train = ImageListDataset(
            train_images, train_targets, transform=self.transform, remove_border=True
        )
        self.dataset_test = ImageListDataset(
            test_images, test_targets, transform=self.test_transform, remove_border=True
        )

    def _load_split_data(self, split_file, dataset_path):
        """Load data from a split file.

        Args:
            split_file: Path to the split file
            dataset_path: Path to the ObjectNet dataset

        Returns:
            Tuple of (images, targets)
        """
        # Read split file
        with open(split_file, "r") as f:
            lines = f.read().strip().split("\n")

        # Create a mapping from folder names to class indices
        folders = sorted(list(set([line.split("/")[2] for line in lines])))
        folder_to_idx = {folder: idx for idx, folder in enumerate(folders)}

        # Process each line in the split file
        images = []
        targets = []

        for line in lines:
            # Split the path components
            parts = line.split("/")
            folder = parts[2]  # Class folder name
            filename = parts[3]  # Image filename

            # In split files, paths are like ./train/mouse_pad/filename.png
            # In the original dataset, paths are like data/objectnet-1.0/images/mouse_pad/filename.png
            img_path = os.path.join(dataset_path, "images", folder, filename)

            if os.path.exists(img_path):
                images.append(img_path)
                targets.append(folder_to_idx[folder])
            else:
                print(f"Warning: Image not found: {img_path}")

        return images, targets


class OmniBenchCL(ContinualDataset):
    """OmniBench dataset for continual learning.

    Based on the OmniBenchmark dataset: https://github.com/ZhangYuanhan-AI/OmniBenchmark
    """

    def __init__(
        self,
        root: str,
        num_steps: int,
        classes_per_step: int,
        transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        seed: int = 42,
        class_order: Optional[List[int]] = None,
        category: str = "activity",  # Which OmniBenchmark category to use
        max_classes: int = 300,  # Maximum number of classes to use (for 'all' category)
    ):
        super().__init__(
            root=root,
            num_steps=num_steps,
            classes_per_step=classes_per_step,
            transform=transform,
            test_transform=test_transform,
            target_transform=target_transform,
            download=download,
            seed=seed,
        )

        self.category = category.lower()
        self.dataset_path = os.path.join(self.root, "omnibenchmark_v2")
        self.max_classes = max_classes

        # Set random seed for reproducibility
        np.random.seed(seed)

        # Check if dataset exists
        if not os.path.exists(self.dataset_path) and self.download:
            raise ValueError(
                f"OmniBenchmark dataset must be downloaded manually. "
                f"Please download from https://github.com/ZhangYuanhan-AI/OmniBenchmark "
                f"and extract to {self.dataset_path}"
            )

        # Special handling for 'all' category
        if self.category == "all":
            # Get all available categories
            categories = [
                d
                for d in os.listdir(os.path.join(self.dataset_path, "data"))
                if os.path.isdir(os.path.join(self.dataset_path, "data", d))
            ]

            # Get class counts for each category
            category_class_counts = {}
            for category in categories:
                max_label = self._get_max_label_for_category(category)
                if max_label >= 0:
                    category_class_counts[category] = max_label + 1

            # Calculate total number of classes across all categories
            total_classes = sum(category_class_counts.values())

            # If total classes exceed max_classes, we need to subset
            if total_classes > self.max_classes and self.max_classes > 0:
                # Sort categories by class count (ascending)
                sorted_categories = sorted(
                    category_class_counts.items(), key=lambda x: x[1]
                )

                # First, include categories with fewer classes until we reach max_classes
                selected_class_count = 0
                for category, class_count in sorted_categories:
                    if selected_class_count + class_count <= self.max_classes:
                        selected_class_count += class_count
                    else:
                        # If adding this category would exceed max_classes, skip it
                        continue

                self.num_classes = selected_class_count
            else:
                self.num_classes = total_classes
        else:
            # Check if the requested category exists
            category_path = os.path.join(self.dataset_path, "data", self.category)
            if not os.path.exists(category_path):
                raise ValueError(
                    f"Category '{self.category}' not found in OmniBenchmark dataset. "
                    f"Available categories are in {os.path.join(self.dataset_path, 'data')}"
                )

            # Get the number of classes for a single category
            max_label = self._get_max_label_for_category(self.category)

            # Number of classes is max label + 1 (since labels are 0-indexed)
            self.num_classes = max_label + 1

        # Set class order (either provided or random)
        if class_order is None:
            self.class_order = list(range(self.num_classes))
            np.random.shuffle(self.class_order)
        else:
            assert len(class_order) == self.num_classes, (
                "Class order must contain all classes"
            )
            self.class_order = class_order

        self.setup()

    def _get_max_label_for_category(self, category):
        """Get the maximum label ID for a specific category.

        Args:
            category: The category to check

        Returns:
            The maximum label ID found in the category's annotation files
        """
        train_file = os.path.join(
            self.dataset_path, "annotation", category, "meta", "train.txt"
        )
        val_file = os.path.join(
            self.dataset_path, "annotation", category, "meta", "val.txt"
        )
        test_file = os.path.join(
            self.dataset_path, "annotation", category, "meta", "test.txt"
        )

        max_label = -1
        for file_path in [train_file, val_file, test_file]:
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            label = int(parts[1])
                            max_label = max(max_label, label)

        return max_label

    def setup(self):
        """Setup OmniBenchmark dataset."""
        train_images = []
        train_targets = []
        test_images = []
        test_targets = []

        if self.category == "all":
            # Get all available categories
            categories = [
                d
                for d in os.listdir(os.path.join(self.dataset_path, "data"))
                if os.path.isdir(os.path.join(self.dataset_path, "data", d))
            ]

            # Get class counts for each category to help with subsetting
            category_class_counts = {}
            for category in categories:
                max_label = self._get_max_label_for_category(category)
                if max_label >= 0:
                    category_class_counts[category] = max_label + 1

            # Calculate total number of classes across all categories
            total_classes = sum(category_class_counts.values())

            # If total classes exceed max_classes, we need to subset
            if total_classes > self.max_classes and self.max_classes > 0:
                # Randomly select categories to include, prioritizing those with fewer classes
                selected_categories = []
                selected_class_count = 0

                # Sort categories by class count (ascending)
                sorted_categories = sorted(
                    category_class_counts.items(), key=lambda x: x[1]
                )

                # First, include categories with fewer classes until we reach max_classes
                for category, class_count in sorted_categories:
                    if selected_class_count + class_count <= self.max_classes:
                        selected_categories.append(category)
                        selected_class_count += class_count
                    else:
                        # If adding this category would exceed max_classes, skip it
                        continue

                # If we still have room, randomly select from remaining categories
                remaining_categories = [
                    c for c in categories if c not in selected_categories
                ]
                np.random.shuffle(remaining_categories)

                for category in remaining_categories:
                    class_count = category_class_counts[category]
                    if selected_class_count + class_count <= self.max_classes:
                        selected_categories.append(category)
                        selected_class_count += class_count

                # Use only the selected categories
                categories = selected_categories
                print(
                    f"Using {len(categories)} categories with {selected_class_count} classes in total (max: {self.max_classes})"
                )

            # Track label offset for each category
            label_offset = 0

            # Process each category
            for category in categories:
                (
                    category_train_images,
                    category_train_targets,
                    category_test_images,
                    category_test_targets,
                ) = self._load_category_data(category, label_offset)

                train_images.extend(category_train_images)
                train_targets.extend(category_train_targets)
                test_images.extend(category_test_images)
                test_targets.extend(category_test_targets)

                # Update label offset for the next category
                if category_train_targets or category_test_targets:
                    max_label = max(
                        max(category_train_targets) if category_train_targets else -1,
                        max(category_test_targets) if category_test_targets else -1,
                    )
                    label_offset = max_label + 1
        else:
            # Load a single category
            train_images, train_targets, test_images, test_targets = (
                self._load_category_data(self.category)
            )

        # Create custom datasets
        self.dataset_train = ImageListDataset(
            train_images, train_targets, transform=self.transform
        )
        self.dataset_test = ImageListDataset(
            test_images, test_targets, transform=self.test_transform
        )

    def _load_category_data(self, category, label_offset=0):
        """Load data for a specific category.

        Args:
            category: The category to load data for
            label_offset: Offset to add to all labels (for 'all' mode)

        Returns:
            train_images, train_targets, test_images, test_targets
        """
        # Paths to annotation files
        train_file = os.path.join(
            self.dataset_path, "annotation", category, "meta", "train.txt"
        )
        val_file = os.path.join(
            self.dataset_path, "annotation", category, "meta", "val.txt"
        )
        test_file = os.path.join(
            self.dataset_path, "annotation", category, "meta", "test.txt"
        )

        # Path to images
        images_path = os.path.join(self.dataset_path, "data", category, "images")

        # Load train and validation data
        train_images = []
        train_targets = []

        # Load training data
        if os.path.exists(train_file):
            with open(train_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        img_file = parts[0]
                        label = int(parts[1]) + label_offset
                        img_path = os.path.join(images_path, img_file)
                        if os.path.exists(img_path):
                            train_images.append(img_path)
                            train_targets.append(label)

        # Add validation data to training set
        if os.path.exists(val_file):
            with open(val_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        img_file = parts[0]
                        label = int(parts[1]) + label_offset
                        img_path = os.path.join(images_path, img_file)
                        if os.path.exists(img_path):
                            train_images.append(img_path)
                            train_targets.append(label)

        # Load test data
        test_images = []
        test_targets = []

        if os.path.exists(test_file):
            with open(test_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        img_file = parts[0]
                        label = int(parts[1]) + label_offset
                        img_path = os.path.join(images_path, img_file)
                        if os.path.exists(img_path):
                            test_images.append(img_path)
                            test_targets.append(label)

        return train_images, train_targets, test_images, test_targets


class OmniBench300CL(ContinualDataset):
    """OmniBench 300-class subset for continual learning.

    This class implements a subset of OmniBenchmark with 300 classes,
    using provided split files in data/omnibenchmark-300.
    """

    def __init__(
        self,
        root: str,
        num_steps: int,
        classes_per_step: int,
        transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        seed: int = 42,
        class_order: Optional[List[int]] = None,
    ):
        super().__init__(
            root=root,
            num_steps=num_steps,
            classes_per_step=classes_per_step,
            transform=transform,
            test_transform=test_transform,
            target_transform=target_transform,
            download=download,
            seed=seed,
        )

        self.dataset_path = os.path.join(self.root, "omnibenchmark_v2")
        self.split_path = os.path.join(self.root, "omnibenchmark-300")

        # Set number of classes
        self.num_classes = 300

        # Set random seed for reproducibility
        np.random.seed(seed)

        # Check if dataset exists
        if not os.path.exists(self.dataset_path) and self.download:
            raise ValueError(
                f"OmniBenchmark dataset must be downloaded manually. "
                f"Please download from https://github.com/ZhangYuanhan-AI/OmniBenchmark "
                f"and extract to {self.dataset_path}"
            )

        # Check if split files exist
        if not os.path.exists(self.split_path):
            raise ValueError(
                f"OmniBench 300-class split files not found at {self.split_path}. "
                f"Please ensure the split files are placed in the correct location."
            )

        # Set class order (either provided or random)
        if class_order is None:
            self.class_order = list(range(self.num_classes))
            np.random.shuffle(self.class_order)
        else:
            assert len(class_order) == self.num_classes, (
                "Class order must contain all classes"
            )
            self.class_order = class_order

        self.setup()

    def _build_image_lookup_table(self):
        """Build a single lookup table mapping image filenames to their categories.

        Includes images from all splits (train, val, test) in the original dataset.
        """
        print("Building image lookup table...")
        image_to_category = {}

        # Read all meta files once and build lookup table
        for category in os.listdir(os.path.join(self.dataset_path, "annotation")):
            # Process all split files (train, val, test)
            for split in ["train", "val", "test"]:
                meta_file = os.path.join(
                    self.dataset_path, "annotation", category, "meta", f"{split}.txt"
                )
                if os.path.exists(meta_file):
                    with open(meta_file, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                img_filename = parts[0]
                                image_to_category[img_filename] = category

        print(f"Built lookup table with {len(image_to_category)} total images")
        return image_to_category

    def _collect_unique_class_ids(self):
        """Collect all unique class IDs from the split files and create a mapping."""
        unique_class_ids = set()

        # Process both train and test split files
        for split in ["train", "test"]:
            split_file = os.path.join(self.split_path, f"{split}.txt")
            if os.path.exists(split_file):
                with open(split_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split("/")
                            if len(parts) >= 3:
                                class_id = int(parts[2])
                                unique_class_ids.add(class_id)

        # Create a mapping from original class IDs to sequential IDs starting from 0
        class_id_mapping = {
            original_id: idx for idx, original_id in enumerate(sorted(unique_class_ids))
        }
        print(
            f"Remapped {len(class_id_mapping)} unique class IDs to sequential IDs starting from 0"
        )
        return class_id_mapping

    def _process_split_file(self, split, image_to_category, class_id_mapping):
        """Process a split file and return images and targets.

        Args:
            split: The split name ('train' or 'test')
            image_to_category: Lookup table mapping image filenames to categories
            class_id_mapping: Mapping from original class IDs to sequential IDs

        Returns:
            images: List of image paths
            targets: List of class labels
        """
        images = []
        targets = []
        split_file = os.path.join(self.split_path, f"{split}.txt")

        if os.path.exists(split_file):
            with open(split_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # Parse the line from the split file
                        # Format: ./train/54/5219044416_f1f7bdeec6_c.jpg or ./test/54/5219044416_f1f7bdeec6_c.jpg
                        parts = line.split("/")
                        if len(parts) >= 3:
                            original_class_id = int(
                                parts[2]
                            )  # Extract original class ID (e.g., 54)
                            remapped_class_id = class_id_mapping[
                                original_class_id
                            ]  # Remap to sequential ID
                            img_filename = parts[3]  # Extract image filename

                            # Use the lookup table to find the category
                            if img_filename in image_to_category:
                                category = image_to_category[img_filename]
                                img_path = os.path.join(
                                    self.dataset_path,
                                    "data",
                                    category,
                                    "images",
                                    img_filename,
                                )
                                if os.path.exists(img_path):
                                    images.append(img_path)
                                    targets.append(remapped_class_id)
                                else:
                                    print(
                                        f"Warning: {split.capitalize()} image file not found: {img_path}"
                                    )
                            else:
                                print(
                                    f"Warning: {split.capitalize()} image {img_filename} not found in meta files"
                                )

        return images, targets

    def setup(self):
        """Setup OmniBench 300-class subset dataset."""
        # Build a single lookup table for all image filenames to categories
        image_to_category = self._build_image_lookup_table()

        # Collect unique class IDs and create mapping
        class_id_mapping = self._collect_unique_class_ids()

        # Process train and test split files
        train_images, train_targets = self._process_split_file(
            "train", image_to_category, class_id_mapping
        )
        test_images, test_targets = self._process_split_file(
            "test", image_to_category, class_id_mapping
        )

        # Create custom datasets
        self.dataset_train = ImageListDataset(
            train_images, train_targets, transform=self.transform
        )
        self.dataset_test = ImageListDataset(
            test_images, test_targets, transform=self.test_transform
        )

        print(
            f"Loaded OmniBench300 dataset with {len(train_images)} training images and {len(test_images)} test images"
        )


class VTABCL(ContinualDataset):
    """Visual Task Adaptation Benchmark (VTAB) for continual learning."""

    def __init__(
        self,
        root: str,
        num_steps: int,
        classes_per_step: int,
        transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        seed: int = 42,
        class_order: Optional[List[int]] = None,
        task: str = "cifar100",  # Which VTAB task to use
    ):
        super().__init__(
            root=root,
            num_steps=num_steps,
            classes_per_step=classes_per_step,
            transform=transform,
            test_transform=test_transform,
            target_transform=target_transform,
            download=download,
            seed=seed,
        )

        self.task = task.lower()

        # Set number of classes based on the task
        task_classes = {
            "cifar100": 100,
            "caltech101": 102,
            "dtd": 47,
            "oxford_flowers102": 102,
            "oxford_iiit_pet": 37,
            "sun397": 397,
            "svhn": 10,
            "patch_camelyon": 2,
            "eurosat": 10,
            "resisc45": 45,
            "diabetic_retinopathy": 5,
            "clevr_count": 8,
            "clevr_distance": 6,
            "dmlab": 6,
            "kitti": 4,
            "smallnorb_azimuth": 18,
            "smallnorb_elevation": 9,
            "dsprites_orientation": 16,
            "dsprites_location": 16,
            # Torchvision datasets
            "cifar10": 10,
            "caltech256": 257,
            "celeba": 40,  # Attributes
            "country211": 211,
            "emnist": 62,  # Letters and digits
            "fashionmnist": 10,
            "fer2013": 7,
            "food101": 101,
            "gtsrb": 43,
            "imagenet": 1000,
            "kmnist": 10,
            "lfw": 5749,  # People
            "mnist": 10,
            "omniglot": 1623,
            "places365": 365,
            "qmnist": 10,
            "rendered_sst2": 2,
            "stanfordcars": 196,
            "stl10": 10,
            "usps": 10,
        }

        if self.task not in task_classes:
            raise ValueError(f"Unsupported VTAB task: {self.task}")

        self.num_classes = task_classes[self.task]
        self.is_torchvision_dataset = self.task in [
            "cifar10",
            "cifar100",
            "caltech101",
            "caltech256",
            "celeba",
            "country211",
            "dtd",
            "emnist",
            "eurosat",
            "fashionmnist",
            "fer2013",
            "food101",
            "gtsrb",
            "imagenet",
            "kmnist",
            "lfw",
            "mnist",
            "omniglot",
            "oxford_flowers102",
            "oxford_iiit_pet",
            "places365",
            "qmnist",
            "rendered_sst2",
            "stanfordcars",
            "stl10",
            "sun397",
            "svhn",
            "usps",
        ]

        # Set class order (either provided or random)
        if class_order is None:
            self.class_order = list(range(self.num_classes))
            np.random.shuffle(self.class_order)
        else:
            assert len(class_order) == self.num_classes, (
                "Class order must contain all classes"
            )
            self.class_order = class_order

        self.setup()

    def setup(self):
        """Setup VTAB dataset."""
        if self.is_torchvision_dataset:
            # Use torchvision dataset
            self._setup_torchvision_dataset()
        else:
            # Use custom VTAB dataset
            self._setup_custom_vtab_dataset()

    def _setup_torchvision_dataset(self):
        """Setup dataset using torchvision."""
        # Map task name to torchvision dataset class
        dataset_mapping = {
            "cifar10": tv_datasets.CIFAR10,
            "cifar100": tv_datasets.CIFAR100,
            "caltech101": tv_datasets.Caltech101,
            "caltech256": tv_datasets.Caltech256,
            "celeba": tv_datasets.CelebA,
            "country211": tv_datasets.Country211,
            "dtd": tv_datasets.DTD,
            "emnist": lambda root, train, download, transform: tv_datasets.EMNIST(
                root,
                split="byclass",
                train=train,
                download=download,
                transform=transform,
            ),
            "eurosat": tv_datasets.EuroSAT,
            "fashionmnist": tv_datasets.FashionMNIST,
            "fer2013": tv_datasets.FER2013,
            "food101": tv_datasets.Food101,
            "gtsrb": tv_datasets.GTSRB,
            "imagenet": tv_datasets.ImageNet,
            "kmnist": tv_datasets.KMNIST,
            "lfw": lambda root, download, transform: tv_datasets.LFWPeople(
                root, download=download, transform=transform
            ),
            "mnist": tv_datasets.MNIST,
            "omniglot": tv_datasets.Omniglot,
            "oxford_flowers102": tv_datasets.Flowers102,
            "oxford_iiit_pet": tv_datasets.OxfordIIITPet,
            "places365": tv_datasets.Places365,
            "qmnist": tv_datasets.QMNIST,
            "rendered_sst2": tv_datasets.RenderedSST2,
            "stanfordcars": tv_datasets.StanfordCars,
            "stl10": tv_datasets.STL10,
            "sun397": tv_datasets.SUN397,
            "svhn": tv_datasets.SVHN,
            "usps": tv_datasets.USPS,
        }

        dataset_class = dataset_mapping[self.task]

        # Handle datasets with different initialization parameters
        try:
            # Standard case: train/test split with transforms
            if self.task in [
                "cifar10",
                "cifar100",
                "fashionmnist",
                "kmnist",
                "mnist",
                "qmnist",
                "usps",
            ]:
                self.dataset_train = dataset_class(
                    root=self.root,
                    train=True,
                    download=self.download,
                    transform=self.transform,
                )
                self.dataset_test = dataset_class(
                    root=self.root,
                    train=False,
                    download=self.download,
                    transform=self.test_transform,
                )
            # SVHN has 'train' and 'test' splits
            elif self.task == "svhn":
                self.dataset_train = dataset_class(
                    root=self.root,
                    split="train",
                    download=self.download,
                    transform=self.transform,
                )
                self.dataset_test = dataset_class(
                    root=self.root,
                    split="test",
                    download=self.download,
                    transform=self.test_transform,
                )
            # STL10 has 'train' and 'test' splits
            elif self.task == "stl10":
                self.dataset_train = dataset_class(
                    root=self.root,
                    split="train",
                    download=self.download,
                    transform=self.transform,
                )
                self.dataset_test = dataset_class(
                    root=self.root,
                    split="test",
                    download=self.download,
                    transform=self.test_transform,
                )
            # EMNIST has special handling
            elif self.task == "emnist":
                self.dataset_train = dataset_class(
                    self.root, True, self.download, self.transform
                )
                self.dataset_test = dataset_class(
                    self.root, False, self.download, self.test_transform
                )
            # LFW has special handling
            elif self.task == "lfw":
                dataset = dataset_class(self.root, self.download, self.transform)
                # Create train/test split
                indices = list(range(len(dataset)))
                np.random.shuffle(indices)
                split_idx = int(len(indices) * 0.8)  # 80% for training
                train_indices = indices[:split_idx]
                test_indices = indices[split_idx:]
                self.dataset_train = Subset(dataset, train_indices)
                # Create a new dataset with test transform
                test_dataset = dataset_class(
                    self.root, self.download, self.test_transform
                )
                self.dataset_test = Subset(test_dataset, test_indices)
            # Datasets with 'train' and 'test' splits and additional parameters
            else:
                # Try standard initialization with split parameter
                try:
                    self.dataset_train = dataset_class(
                        root=self.root,
                        split="train",
                        download=self.download,
                        transform=self.transform,
                    )
                    self.dataset_test = dataset_class(
                        root=self.root,
                        split="test",
                        download=self.download,
                        transform=self.test_transform,
                    )
                except (TypeError, ValueError):
                    # If that fails, try with 'train' boolean parameter
                    try:
                        self.dataset_train = dataset_class(
                            root=self.root,
                            train=True,
                            download=self.download,
                            transform=self.transform,
                        )
                        self.dataset_test = dataset_class(
                            root=self.root,
                            train=False,
                            download=self.download,
                            transform=self.test_transform,
                        )
                    except (TypeError, ValueError):
                        # Last resort: create a dataset and split it manually
                        dataset = dataset_class(
                            root=self.root,
                            download=self.download,
                            transform=self.transform,
                        )
                        # Create train/test split
                        indices = list(range(len(dataset)))
                        np.random.shuffle(indices)
                        split_idx = int(len(indices) * 0.8)  # 80% for training
                        train_indices = indices[:split_idx]
                        test_indices = indices[split_idx:]
                        self.dataset_train = Subset(dataset, train_indices)
                        # Create a new dataset with test transform
                        test_dataset = dataset_class(
                            root=self.root,
                            download=self.download,
                            transform=self.test_transform,
                        )
                        self.dataset_test = Subset(test_dataset, test_indices)
        except Exception as e:
            raise ValueError(f"Failed to initialize {self.task} dataset: {str(e)}")

    def _setup_custom_vtab_dataset(self):
        """Setup custom VTAB dataset."""
        # Check if dataset exists
        dataset_path = os.path.join(self.root, self.task)
        if not os.path.exists(dataset_path) and self.download:
            raise ValueError(
                f"VTAB {self.task} dataset must be downloaded manually. "
                f"Please download and extract to {dataset_path}"
            )

        # VTAB datasets typically have a train and test folder
        train_path = os.path.join(dataset_path, "train")
        test_path = os.path.join(dataset_path, "test")

        if os.path.exists(train_path) and os.path.exists(test_path):
            # If the dataset follows the standard structure
            self.dataset_train = ImageFolder(train_path, transform=self.transform)
            self.dataset_test = ImageFolder(test_path, transform=self.test_transform)
        else:
            # If the dataset has a different structure, we need to manually create the datasets
            # For simplicity, we'll assume the dataset follows an ImageFolder-like structure
            # but without the train/test split

            # Create datasets from the root with appropriate transforms
            train_dataset = ImageFolder(dataset_path, transform=self.transform)
            test_dataset = ImageFolder(dataset_path, transform=self.test_transform)

            # Create train/test split
            indices = list(range(len(train_dataset)))
            np.random.shuffle(indices)
            split_idx = int(len(indices) * 0.8)  # 80% for training

            train_indices = indices[:split_idx]
            test_indices = indices[split_idx:]

            # Create train and test datasets
            self.dataset_train = Subset(train_dataset, train_indices)
            self.dataset_test = Subset(test_dataset, test_indices)


class ImageListDataset(Dataset):
    """Dataset from a list of images and targets."""

    def __init__(self, images, targets, transform=None, remove_border=False):
        self.images = images
        self.targets = targets
        self.transform = transform
        self.remove_border = remove_border

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        target = self.targets[idx]

        # Load image
        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")

        # Remove 1-pixel border if specified (for ObjectNet)
        if self.remove_border:
            width, height = img.size
            img = img.crop((1, 1, width - 1, height - 1))

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class MergedTaskDataset(ContinualDataset):
    """Dataset that merges multiple task datasets with class subsetting.

    This class allows you to:
    1. Subset multiple task datasets by specifying the number of classes to use from each
    2. Merge these subsets into one larger dataset with proper class indexing
    3. Use the merged dataset for continual learning

    Example usage:
        merged_dataset = MergedTaskDataset(
            root="/path/to/data",
            num_steps=3,
            classes_per_step=10,
            task_subsets={
                "dtd": 10,                # Use 10 classes from DTD
                "oxford_flowers102": 10, # Use 10 classes from Oxford Flowers
                "oxford_iiit_pet": 10,   # Use 10 classes from Oxford Pets
            },
            transform=transform,
            test_transform=test_transform,
            download=True,
            seed=42,
        )
    """

    def __init__(
        self,
        root: str,
        num_steps: int,
        classes_per_step: int,
        task_subsets: Dict[
            str, int
        ],  # Dict mapping task names to number of classes to use
        transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        seed: int = 42,
        class_order: Optional[List[int]] = None,
    ):
        super().__init__(
            root=root,
            num_steps=num_steps,
            classes_per_step=classes_per_step,
            transform=transform,
            test_transform=test_transform,
            target_transform=target_transform,
            download=download,
            seed=seed,
        )

        self.task_subsets = task_subsets

        # Validate tasks
        valid_tasks = {
            "cifar100": 100,
            "caltech101": 102,
            "dtd": 47,
            "oxford_flowers102": 102,
            "oxford_iiit_pet": 37,
            "sun397": 397,
            "svhn": 10,
            "patch_camelyon": 2,
            "eurosat": 10,
            "resisc45": 45,
            "diabetic_retinopathy": 5,
            "clevr_count": 8,
            "clevr_distance": 6,
            "dmlab": 6,
            "kitti": 4,
            "smallnorb_azimuth": 18,
            "smallnorb_elevation": 9,
            "dsprites_orientation": 16,
            "dsprites_location": 16,
        }

        # Check if all tasks are valid
        for task, num_classes in self.task_subsets.items():
            if task not in valid_tasks:
                raise ValueError(f"Unsupported task: {task}")
            if num_classes > valid_tasks[task]:
                raise ValueError(
                    f"Requested {num_classes} classes from {task}, but it only has {valid_tasks[task]} classes"
                )

        # Calculate total number of classes
        self.num_classes = sum(self.task_subsets.values())

        # Set class order (either provided or random)
        if class_order is None:
            self.class_order = list(range(self.num_classes))
            np.random.shuffle(self.class_order)
        else:
            assert len(class_order) == self.num_classes, (
                "Class order must contain all classes"
            )
            self.class_order = class_order

        self.setup()

    def setup(self):
        """Setup merged dataset by loading and merging task subsets."""
        train_datasets = []
        test_datasets = []

        # Keep track of class remapping for each task
        self.class_remapping = {}
        current_class_offset = 0

        # Process each task
        for task_name, num_classes in self.task_subsets.items():
            # Create a temporary VTAB dataset for this task
            task_dataset = VTABCL(
                root=self.root,
                num_steps=1,  # Not relevant for our purpose
                classes_per_step=1,  # Not relevant for our purpose
                transform=self.transform,
                test_transform=self.test_transform,
                target_transform=self.target_transform,
                download=self.download,
                seed=self.seed,
                task=task_name,
            )

            # Randomly select classes from this task
            available_classes = list(range(task_dataset.num_classes))
            np.random.shuffle(available_classes)
            selected_classes = available_classes[:num_classes]

            # Create class remapping for this task
            # Maps original class IDs to new sequential IDs in the merged dataset
            task_class_map = {}
            for i, class_id in enumerate(selected_classes):
                task_class_map[class_id] = current_class_offset + i

            self.class_remapping[task_name] = task_class_map

            # Get indices for selected classes from train dataset
            train_indices = task_dataset._get_indices_for_classes(
                task_dataset.dataset_train, selected_classes
            )

            # Get indices for selected classes from test dataset
            test_indices = task_dataset._get_indices_for_classes(
                task_dataset.dataset_test, selected_classes
            )

            # Create remapped datasets with new class IDs
            train_subset = self._create_remapped_subset(
                task_dataset.dataset_train, train_indices, task_class_map
            )

            test_subset = self._create_remapped_subset(
                task_dataset.dataset_test, test_indices, task_class_map
            )

            # Add to our list of datasets
            train_datasets.append(train_subset)
            test_datasets.append(test_subset)

            # Update class offset for the next task
            current_class_offset += num_classes

        # Merge all datasets
        self.dataset_train = ConcatDataset(train_datasets)
        self.dataset_test = ConcatDataset(test_datasets)

    def _create_remapped_subset(self, dataset, indices, class_map):
        """Create a subset with remapped class IDs.

        Args:
            dataset: The original dataset
            indices: Indices to include in the subset
            class_map: Mapping from original class IDs to new class IDs

        Returns:
            A dataset with remapped class IDs
        """
        return RemappedSubset(dataset, indices, class_map)


class RemappedSubset(Dataset):
    """Subset of a dataset with remapped class IDs."""

    def __init__(self, dataset, indices, class_map):
        self.dataset = dataset
        self.indices = indices
        self.class_map = class_map

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, target = self.dataset[self.indices[idx]]
        # Remap the target to the new class ID
        remapped_target = self.class_map[target]
        return image, remapped_target
