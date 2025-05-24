from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def calculate_dataset_statistics(
    dataset: Dataset,
    batch_size: int = 1,
    num_workers: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the mean and standard deviation of a dataset.

    Args:
        dataset: The dataset to calculate statistics for
        batch_size: Batch size for processing
        num_workers: Number of workers for data loading
        device: Device to use for computation

    Returns:
        Tuple of (mean, std) tensors with shape [C]
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    n_pixels = 0
    mean_sum = None
    var_sum = None

    print("Calculating dataset statistics...")
    # Calculate mean
    for batch in tqdm(dataloader, desc="Calculating mean"):
        if isinstance(batch, (list, tuple)):
            images = batch[0]  # Assume first element is the image
        else:
            images = batch
        images = images.to(device)

        b, c, h, w = images.size()
        n_pixels += b * h * w
        batch_mean_sum = torch.sum(images, dim=(0, 2, 3))  # [C]

        if mean_sum is None:
            mean_sum = batch_mean_sum
        else:
            mean_sum += batch_mean_sum
    mean = mean_sum / n_pixels  # [C]
    mean_rounded = mean.round().long()  # [C]

    # Calculate std deviation in a second pass
    for batch in tqdm(dataloader, desc="Calculating std"):
        if isinstance(batch, (list, tuple)):
            images = batch[0]  # Assume first element is the image
        else:
            images = batch
        images = images.to(device)

        b, c, h, w = images.size()
        centered = images - mean_rounded.view(1, c, 1, 1)  # [B, C, H, W]
        batch_var_sum = torch.sum(centered**2, dim=(0, 2, 3))  # [C]

        if var_sum is None:
            var_sum = batch_var_sum
        else:
            var_sum += batch_var_sum
    std = torch.sqrt(var_sum / n_pixels)  # [C]

    return mean.cpu(), std.cpu()


class MomentMatchingTransform(torch.nn.Module):
    """
    Transform that applies moment matching to align data statistics with target statistics.

    This transform adjusts the input data to match the first two moments (mean and standard deviation)
    of the target distribution (pretraining data statistics).

    Formula: X_adj = (X - μ_current) * (σ_pretrain/σ_current) + μ_pretrain

    Where:
    - X is the input data with statistics μ_current, σ_current
    - X_adj is the adjusted data with statistics μ_pretrain, σ_pretrain
    - μ_pretrain, σ_pretrain are the statistics of the pretraining data (e.g., ImageNet)
    - μ_current, σ_current are the statistics of the current dataset
    """

    def __init__(
        self,
        current_data_mean: List[float],
        current_data_std: List[float],
        pretraining_data_mean: List[float],
        pretraining_data_std: List[float],
    ):
        """
        Initialize the moment matching transform.

        Args:
            current_data_mean: Mean of the current dataset
            current_data_std: Standard deviation of the current dataset
            pretraining_data_mean: Mean of the pretraining data
            pretraining_data_std: Standard deviation of the pretraining data
        """
        super().__init__()

        # Convert to tensors
        self.current_data_mean = torch.tensor(current_data_mean).view(-1, 1, 1)
        self.current_data_std = torch.tensor(current_data_std).view(-1, 1, 1)
        self.pretraining_data_mean = torch.tensor(pretraining_data_mean).view(-1, 1, 1)
        self.pretraining_data_std = torch.tensor(pretraining_data_std).view(-1, 1, 1)

        # Pre-compute scaling factor
        self.scale = self.pretraining_data_std / self.current_data_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply moment matching to the input tensor.

        Args:
            x: Input tensor of shape [C, H, W]

        Returns:
            Transformed tensor with matched moments
        """
        # Apply the transformation: X_adj = (X - μ_current) * (σ_pretrain/σ_current) + μ_pretrain
        x_centered = x - self.current_data_mean
        x_scaled = x_centered * self.scale
        x_shifted = x_scaled + self.pretraining_data_mean

        return x_shifted
