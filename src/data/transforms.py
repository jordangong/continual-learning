from typing import Dict, List, Tuple

from torchvision import transforms


def get_transforms(
    input_size: int,
    mean: List[float],
    std: List[float],
    augmentation: Dict[str, bool] = None,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get train and test transforms for image data.

    Args:
        input_size: Input size for images
        mean: Mean values for normalization
        std: Standard deviation values for normalization
        augmentation: Dictionary of augmentation options

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
        }

    # Test transform
    if augmentation.get("test_center_crop", True):
        resize_size = int((256 / 224) * input_size)
        test_transforms = [
            transforms.Resize(resize_size),
            transforms.CenterCrop(input_size),
        ]
    else:
        test_transforms = [transforms.Resize((input_size, input_size))]

    test_transforms.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )

    test_transform = transforms.Compose(test_transforms)

    # Start with basic transforms
    train_transforms = []

    # Add augmentations based on config
    if augmentation.get("random_resized_crop", True):
        scale = augmentation.get("random_resized_crop_scale", (0.08, 1.0))
        ratio = augmentation.get("random_resized_crop_ratio", (3 / 4, 4 / 3))
        train_transforms.append(transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio))
    else:
        # Fall back to traditional resize if random_resized_crop is disabled
        train_transforms.append(transforms.Resize((input_size, input_size)))

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
    train_transforms.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )

    train_transform = transforms.Compose(train_transforms)

    return train_transform, test_transform
