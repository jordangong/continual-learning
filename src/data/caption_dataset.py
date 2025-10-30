"""
Dataset wrapper for loading recaptioned data.

This module provides a wrapper that adds recaptioned text to existing datasets,
enabling vision-language pretraining and text encoder training.
"""

import json
import random
from pathlib import Path
from typing import Any, Optional, Tuple

from torch.utils.data import Dataset


class CaptionDataset(Dataset):
    """
    Wrapper dataset that adds captions to images.
    
    Args:
        base_dataset: The underlying image dataset to add captions to
        caption_file: Path to JSON file containing captions
        sample_strategy: How to select captions ("random", "first", or "all")
        seed: Random seed for caption sampling
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        caption_file: str,
        sample_strategy: str = "random",
        seed: int = 42,
    ):
        self.base_dataset = base_dataset
        self.sample_strategy = sample_strategy
        self.rng = random.Random(seed)
        
        # Load captions from JSON
        print(f"Loading captions from {caption_file}...")
        with open(caption_file, "r") as f:
            self.caption_data = json.load(f)
        
        # Build index-based caption lookup
        self._build_index_lookup()
        
    def _build_index_lookup(self):
        """Build lookup table mapping dataset indices to captions."""
        self.caption_lookup = {}
        
        for key, entry in self.caption_data.items():
            img_idx = entry["image_idx"]
            self.caption_lookup[img_idx] = entry["captions"]
        
        print(f"Built index-based lookup with {len(self.caption_lookup)} entries")
    
    def _get_caption(self, idx: int) -> Optional[str]:
        """Get caption for a given index."""
        # Use index-based matching
        # Captions are generated on the subset (train/test split), so indices
        # in the JSON file match the subset indices directly (0, 1, 2, ...)
        if idx not in self.caption_lookup:
            return None
        
        captions = self.caption_lookup[idx]
        
        # Select caption based on strategy
        if self.sample_strategy == "random":
            return self.rng.choice(captions)
        elif self.sample_strategy == "first":
            return captions[0]
        elif self.sample_strategy == "all":
            return captions  # Return all captions as a list
        else:
            return self.rng.choice(captions)
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[Any, int, Optional[str]]:
        """
        Get item with caption.
        
        Returns:
            image: The image (PIL Image or Tensor)
            label: The class label
            caption: The caption text (or None if not available)
        """
        image, label = self.base_dataset[idx]
        caption = self._get_caption(idx)
        
        return image, label, caption
    
    # Forward dataset attributes for compatibility
    def __getattr__(self, name: str):
        """Forward attribute access to base dataset."""
        # __getattr__ is only called when attribute not found in instance
        # Forward to base_dataset (avoid infinite recursion by checking __dict__)
        if 'base_dataset' in self.__dict__:
            return getattr(self.base_dataset, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def create_caption_dataset(
    base_dataset: Dataset,
    caption_dir: str,
    dataset_name: str,
    split: str = "train",
    sample_strategy: str = "random",
    seed: int = 42,
) -> Optional[CaptionDataset]:
    """
    Create a caption dataset if caption file exists.
    
    Args:
        base_dataset: The underlying image dataset
        caption_dir: Directory containing caption JSON files
        dataset_name: Name of the dataset (e.g., "cifar100", "imagenet-r")
        split: Dataset split ("train" or "test")
        sample_strategy: How to select captions ("random", "first", or "all")
        seed: Random seed for caption sampling
    
    Returns:
        CaptionDataset if caption file exists, None otherwise
    """
    # Construct caption file path
    caption_file = Path(caption_dir) / f"{dataset_name}_{split}_captions.json"
    
    if not caption_file.exists():
        print(f"Caption file not found: {caption_file}")
        return None
    
    # Captions are generated on subset, so indices match directly
    return CaptionDataset(
        base_dataset=base_dataset,
        caption_file=str(caption_file),
        sample_strategy=sample_strategy,
        seed=seed,
    )
