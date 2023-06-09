import os
from abc import abstractmethod
from typing import Any, Callable, List, Optional, Union

import torch
from pytorch_lightning import LightningDataModule
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms


class VisionDataModule(LightningDataModule):
    EXTRA_ARGS: dict = {}
    name: str = ""
    #: Dataset class to use
    dataset_cls: type
    #: A tuple describing the shape of the data
    dims: tuple

    def __init__(
            self,
            data_dir: Optional[str] = None,
            val_split: Union[int, float] = 0.2,
            num_workers: int = 0,
            normalize: bool = False,
            batch_size: int = 32,
            seed: int = 42,
            shuffle: bool = True,
            pin_memory: bool = True,
            drop_last: bool = False,
            train_transforms: Optional[Callable] = None,
            val_transforms: Optional[Callable] = None,
            test_transforms: Optional[Callable] = None,
            *args: Any,
            **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: Where to save/load the data
            val_split: Percent (float) or number (int) of samples
                       to use for the validation split
            num_workers: How many workers to use for loading data
            normalize: If true applies image normalize
            batch_size: How many samples per batch to load
            seed: Random seed to be used for train/val/test splits
            shuffle: If true shuffles the train data every epoch
            pin_memory: If true, the data loader will copy Tensors
                        into CUDA pinned memory before returning them
            drop_last: If true drops the last incomplete batch
            train_transforms: transformations you can apply to train dataset
            val_transforms: transformations you can apply to validation dataset
            test_transforms: transformations you can apply to test dataset
        """

        super().__init__(*args, **kwargs)

        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.val_split = val_split
        self.num_workers = num_workers
        self.normalize = normalize
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self._train_transforms = train_transforms
        self._val_transforms = val_transforms
        self._test_transforms = test_transforms

    @property
    def train_transforms(self) -> Optional[Callable[..., Any]]:
        """
        Optional transforms (or collection of transforms)
        you can apply to train dataset.
        """
        return self._train_transforms

    @train_transforms.setter
    def train_transforms(self, t: Callable) -> None:
        self._train_transforms = t

    @property
    def val_transforms(self) -> Optional[Callable[..., Any]]:
        """
        Optional transforms (or collection of transforms)
        you can apply to validation dataset.
        """
        return self._val_transforms

    @val_transforms.setter
    def val_transforms(self, t: Callable) -> None:
        self._val_transforms = t

    @property
    def test_transforms(self) -> Optional[Callable[..., Any]]:
        """
        Optional transforms (or collection of transforms)
        you can apply to test dataset.
        """
        return self._test_transforms

    @test_transforms.setter
    def test_transforms(self, t: Callable) -> None:
        self._test_transforms = t

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """Saves files to data_dir."""
        self.dataset_cls(self.data_dir, train=True, download=True)
        self.dataset_cls(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """Creates train, val, and test dataset."""
        if stage == "fit" or stage is None:
            train_transforms = self.default_transforms() \
                if self.train_transforms is None else self.train_transforms
            val_transforms = self.default_transforms() \
                if self.val_transforms is None else self.val_transforms

            dataset_train = self.dataset_cls(self.data_dir, train=True,
                                             transform=train_transforms,
                                             **self.EXTRA_ARGS)
            dataset_val = self.dataset_cls(self.data_dir, train=True,
                                           transform=val_transforms,
                                           **self.EXTRA_ARGS)

            # Split
            self.dataset_train = self._split_dataset(dataset_train)
            self.dataset_val = self._split_dataset(dataset_val, train=False)

        if stage == "test" or stage is None:
            test_transforms = self.default_transforms() \
                if self.test_transforms is None else self.test_transforms
            self.dataset_test = self.dataset_cls(self.data_dir, train=False,
                                                 transform=test_transforms,
                                                 **self.EXTRA_ARGS)

    def _split_dataset(self, dataset: Dataset, train: bool = True) -> Dataset:
        """Splits the dataset into train and validation set."""
        len_dataset = len(dataset)
        splits = self._get_splits(len_dataset)
        dataset_train, dataset_val = random_split(
            dataset, splits,
            generator=torch.Generator().manual_seed(self.seed)
        )

        if train:
            return dataset_train
        return dataset_val

    def _get_splits(self, len_dataset: int) -> List[int]:
        """Computes split lengths for train and validation set."""
        if isinstance(self.val_split, int):
            train_len = len_dataset - self.val_split
            splits = [train_len, self.val_split]
        elif isinstance(self.val_split, float):
            val_len = int(self.val_split * len_dataset)
            train_len = len_dataset - val_len
            splits = [train_len, val_len]
        else:
            raise ValueError(f"Unsupported type {type(self.val_split)}")

        return splits

    @abstractmethod
    def default_transforms(self) -> Callable:
        """Default transform for the dataset."""

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """The train dataloader."""
        return self._data_loader(self.dataset_train, shuffle=self.shuffle)

    def val_dataloader(
            self,
            *args: Any,
            **kwargs: Any
    ) -> Union[DataLoader, List[DataLoader]]:
        """The val dataloader."""
        return self._data_loader(self.dataset_val)

    def test_dataloader(
            self,
            *args: Any,
            **kwargs: Any
    ) -> Union[DataLoader, List[DataLoader]]:
        """The test dataloader."""
        return self._data_loader(self.dataset_test)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )


class CIFAR10DataModule(VisionDataModule):
    """
    .. figure:: https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2019/01/
        Plot-of-a-Subset-of-Images-from-the-CIFAR-10-Dataset.png
        :width: 400
        :alt: CIFAR-10

    Specs:
        - 10 classes (1 per class)
        - Each image is (3 x 32 x 32)

    Standard CIFAR10, train, val, test splits and transforms

    Transforms::

        transforms = transform_lib.Compose([
            transform_lib.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
            )
        ])

    Example::

        from pl_bolts.datamodules import CIFAR10DataModule

        dm = CIFAR10DataModule(PATH)
        model = LitModel()

        Trainer().fit(model, datamodule=dm)

    Or you can set your own transforms

    Example::

        dm.train_transforms = ...
        dm.test_transforms = ...
        dm.val_transforms  = ...
    """

    name = "cifar10"
    dataset_cls = CIFAR10
    dims = (3, 32, 32)

    def __init__(
            self,
            data_dir: Optional[str] = None,
            val_split: Union[int, float] = 0.2,
            num_workers: int = 0,
            normalize: bool = False,
            batch_size: int = 32,
            seed: int = 42,
            shuffle: bool = True,
            pin_memory: bool = True,
            drop_last: bool = False,
            *args: Any,
            **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: Where to save/load the data
            val_split: Percent (float) or number (int) of samples
                       to use for the validation split
            num_workers: How many workers to use for loading data
            normalize: If true applies image normalize
            batch_size: How many samples per batch to load
            seed: Random seed to be used for train/val/test splits
            shuffle: If true shuffles the train data every epoch
            pin_memory: If true, the data loader will copy Tensors
                        into CUDA pinned memory before returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__(  # type: ignore[misc]
            data_dir=data_dir,
            val_split=val_split,
            num_workers=num_workers,
            normalize=normalize,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            *args,
            **kwargs,
        )

    @property
    def num_samples(self) -> int:
        train_len, _ = self._get_splits(len_dataset=50_000)
        return train_len

    @property
    def num_classes(self) -> int:
        """
        Return:
            10
        """
        return 10


def cifar10aug(args):
    dm = CIFAR10DataModule(
        data_dir=args.data_dir,
        val_split=args.val_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    args.num_samples = dm.num_samples
    args.num_classes = dm.num_classes

    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )

    dm.train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.ToTensor(),
        torch.jit.script(nn.Sequential(
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.2,
            ),
            transforms.RandomGrayscale(p=0.2),
            normalize,
        )),
    ])
    dm.val_transforms = dm.test_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        torch.jit.script(normalize),
    ])

    return dm
