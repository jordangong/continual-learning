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
            "random_crop": True,
            "random_horizontal_flip": True,
            "color_jitter": False,
            "auto_augment": False,
        }

    # Base test transform (always applied)
    test_transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    # Start with basic transforms
    train_transforms = [transforms.Resize((input_size, input_size))]

    # Add augmentations based on config
    if augmentation.get("random_crop", False):
        train_transforms.append(transforms.RandomCrop(input_size, padding=4))

    if augmentation.get("random_horizontal_flip", False):
        train_transforms.append(transforms.RandomHorizontalFlip())

    if augmentation.get("color_jitter", False):
        train_transforms.append(
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            )
        )

    if augmentation.get("auto_augment", False):
        train_transforms.append(
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10)
        )

    # Add final transforms
    train_transforms.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )

    train_transform = transforms.Compose(train_transforms)

    return train_transform, test_transform
