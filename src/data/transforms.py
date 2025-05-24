from typing import Any, Dict, List, Tuple

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.data.moment_matching import MomentMatchingTransform


def get_transforms(
    input_size: int,
    mean: List[float],
    std: List[float],
    augmentation: Dict[str, Any] = None,
    apply_moment_matching: bool = False,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get train and test transforms for image data.

    Args:
        input_size: Input size for images
        mean: Mean values for normalization
        std: Standard deviation values for normalization
        augmentation: Dictionary of augmentation options, including moment matching parameters if needed
        apply_moment_matching: Whether to apply moment matching to align data statistics

    Returns:
        Tuple of (train_transform, test_transform)
    """
    if augmentation is None:
        augmentation = {
            "random_resized_crop": True,
            "random_resized_crop_scale": (0.08, 1.0),
            "random_resized_crop_ratio": (3 / 4, 4 / 3),
            "random_crop": False,
            "random_horizontal_flip": True,
            "color_jitter": False,
            "auto_augment": False,
            "auto_augment_policy": "imagenet",
            "test_center_crop": True,
            "normalize": True,
        }

    # Test transform
    if augmentation.get("test_center_crop", True):
        resize_size = int((256 / 224) * input_size)
        test_transforms = [
            transforms.Resize(
                resize_size,
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(input_size),
        ]
    else:
        test_transforms = [
            transforms.Resize(
                (input_size, input_size),
                interpolation=InterpolationMode.BICUBIC,
            )
        ]

    test_transforms.append(transforms.ToTensor())

    # Apply moment matching if requested
    if apply_moment_matching and "moment_matching" in augmentation:
        mm_params = augmentation["moment_matching"]
        test_transforms.append(
            MomentMatchingTransform(
                current_data_mean=mm_params["current_data_mean"],
                current_data_std=mm_params["current_data_std"],
                pretraining_data_mean=mm_params["pretraining_data_mean"],
                pretraining_data_std=mm_params["pretraining_data_std"],
            )
        )

    if augmentation.get("normalize", True):
        test_transforms.append(transforms.Normalize(mean=mean, std=std))

    test_transform = transforms.Compose(test_transforms)

    # Start with basic transforms
    train_transforms = []

    # Add augmentations based on config
    if augmentation.get("random_resized_crop", True):
        scale = augmentation.get("random_resized_crop_scale", (0.08, 1.0))
        ratio = augmentation.get("random_resized_crop_ratio", (3 / 4, 4 / 3))
        train_transforms.append(
            transforms.RandomResizedCrop(
                input_size,
                scale=scale,
                ratio=ratio,
                interpolation=InterpolationMode.BICUBIC,
            )
        )
    else:
        # Fall back to traditional resize if random_resized_crop is disabled
        train_transforms.append(
            transforms.Resize(
                (input_size, input_size),
                interpolation=InterpolationMode.BICUBIC,
            )
        )
        # Add random crop if enabled
        if augmentation.get("random_crop", False):
            train_transforms.append(transforms.RandomCrop(input_size, padding=4))

    if augmentation.get("random_horizontal_flip", True):
        train_transforms.append(transforms.RandomHorizontalFlip())

    if augmentation.get("color_jitter", False):
        train_transforms.append(
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            )
        )

    if augmentation.get("auto_augment", False):
        policy = augmentation.get("auto_augment_policy", "imagenet")
        if policy == "cifar10":
            policy = transforms.AutoAugmentPolicy.CIFAR10
        elif policy == "imagenet":
            policy = transforms.AutoAugmentPolicy.IMAGENET
        elif policy == "svhn":
            policy = transforms.AutoAugmentPolicy.SVHN
        else:
            raise ValueError(f"Unknown auto_augment_policy: {policy}")

        train_transforms.append(transforms.AutoAugment(policy))

    # Add final transforms
    train_transforms.append(transforms.ToTensor())

    # Apply moment matching if requested
    if apply_moment_matching and "moment_matching" in augmentation:
        mm_params = augmentation["moment_matching"]
        train_transforms.append(
            MomentMatchingTransform(
                current_data_mean=mm_params["current_data_mean"],
                current_data_std=mm_params["current_data_std"],
                pretraining_data_mean=mm_params["pretraining_data_mean"],
                pretraining_data_std=mm_params["pretraining_data_std"],
            )
        )

    if augmentation.get("normalize", True):
        train_transforms.append(transforms.Normalize(mean=mean, std=std))

    train_transform = transforms.Compose(train_transforms)

    return train_transform, test_transform
