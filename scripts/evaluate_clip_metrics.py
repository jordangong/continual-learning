#!/usr/bin/env python
"""
Evaluate CLIP metrics on various datasets to measure image-text alignment quality.

This script calculates:
- Paired similarity: Cosine similarity between matched image-caption pairs
- Unpaired similarity: Cosine similarity with random negative pairs
- CLIP contrastive loss: Standard CLIP InfoNCE loss
- Supervised contrastive loss: Modified CLIP loss using class labels

Datasets supported:
- COCO Captions (reference)
- CIFAR-100, CUB-200, ImageNet-R, Stanford Cars, Aircraft, Food-101, etc.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.datasets import (
    CIFAR100CL,
    Caltech256CL,
    CUB200CL,
    FGVCAircraftCL,
    Food101CL,
    ImageNet100CL,
    ImageNetACL,
    ImageNetRCL,
    OxfordIIITPetCL,
    StanfordCarsCL,
)
from src.data.caption_dataset import create_caption_dataset
from src.trainers.supervised_contrastive_loss import SupervisedClipLoss

try:
    import open_clip
    from open_clip.loss import ClipLoss
    CLIP_AVAILABLE = True
except ImportError:
    print("Warning: open_clip not available. Install with: pip install open-clip-torch")
    CLIP_AVAILABLE = False
    open_clip = None
    ClipLoss = None

try:
    from torchvision.datasets import CocoCaptions
    COCO_AVAILABLE = True
except ImportError:
    print("Warning: torchvision not available for COCO loading. Install with: pip install torchvision")
    COCO_AVAILABLE = False
    CocoCaptions = None


class ClassNameDataset(Dataset):
    """Wrapper to add class name captions to datasets."""
    
    def __init__(self, base_dataset: Dataset, class_names: List[str]):
        """
        Args:
            base_dataset: Base dataset (images and labels)
            class_names: List of class names
        """
        self.base_dataset = base_dataset
        self.class_names = class_names
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        caption = f"a photo of a {self.class_names[label].replace("_", " ")}"
        return image, label, caption


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_clip_model(model_name: str, pretrained: str, cache_dir: str, device: str):
    """Load CLIP model from open_clip."""
    if not CLIP_AVAILABLE:
        raise ImportError("open_clip not available. Install with: pip install open-clip-torch")
    
    print(f"Loading CLIP model: {model_name} (pretrained: {pretrained})")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        cache_dir=cache_dir,
        device=device,
    )
    model.eval()
    
    tokenizer = open_clip.get_tokenizer(model_name)
    
    return model, preprocess, tokenizer


def load_target_dataset(
    dataset_name: str,
    data_dir: str,
    split: str,
    caption_dir: Optional[str],
    transform,
    seed: int = 42,
):
    """Load target dataset with optional captions."""
    # Use same logic as recaptioning.py
    num_steps = 1
    classes_per_step = 1000  # Large number to include all classes
    
    if dataset_name == "cifar100":
        dataset = CIFAR100CL(
            root=data_dir,
            num_steps=num_steps,
            classes_per_step=classes_per_step,
            transform=transform,
            test_transform=transform,
            target_transform=None,
            download=True,
            seed=seed,
        )
    elif dataset_name == "cub200":
        dataset = CUB200CL(
            root=data_dir,
            num_steps=num_steps,
            classes_per_step=classes_per_step,
            transform=transform,
            test_transform=transform,
            target_transform=None,
            download=True,
            seed=seed,
        )
    elif dataset_name == "imagenet-r":
        dataset = ImageNetRCL(
            root=data_dir,
            num_steps=num_steps,
            classes_per_step=classes_per_step,
            transform=transform,
            test_transform=transform,
            target_transform=None,
            download=True,
            seed=seed,
            train_ratio=0.8,
        )
    elif dataset_name == "stanford_cars":
        dataset = StanfordCarsCL(
            root=data_dir,
            num_steps=num_steps,
            classes_per_step=classes_per_step,
            transform=transform,
            test_transform=transform,
            target_transform=None,
            download=False,
            seed=seed,
        )
    elif dataset_name == "fgvc_aircraft":
        dataset = FGVCAircraftCL(
            root=data_dir,
            num_steps=num_steps,
            classes_per_step=classes_per_step,
            transform=transform,
            test_transform=transform,
            target_transform=None,
            download=True,
            seed=seed,
            annotation_level="variant",
        )
    elif dataset_name == "caltech256":
        dataset = Caltech256CL(
            root=data_dir,
            num_steps=num_steps,
            classes_per_step=classes_per_step,
            transform=transform,
            test_transform=transform,
            target_transform=None,
            download=True,
            seed=seed,
            train_ratio=0.8,
        )
    elif dataset_name == "food101":
        dataset = Food101CL(
            root=data_dir,
            num_steps=num_steps,
            classes_per_step=classes_per_step,
            transform=transform,
            test_transform=transform,
            target_transform=None,
            download=True,
            seed=seed,
        )
    elif dataset_name == "oxford_pet":
        dataset = OxfordIIITPetCL(
            root=data_dir,
            num_steps=num_steps,
            classes_per_step=classes_per_step,
            transform=transform,
            test_transform=transform,
            target_transform=None,
            download=True,
            seed=seed,
        )
    elif dataset_name == "imagenet100":
        dataset = ImageNet100CL(
            root=data_dir,
            num_steps=num_steps,
            classes_per_step=classes_per_step,
            transform=transform,
            test_transform=transform,
            target_transform=None,
            download=False,
            seed=seed,
        )
    elif dataset_name == "imagenet-a":
        dataset = ImageNetACL(
            root=data_dir,
            num_steps=num_steps,
            classes_per_step=classes_per_step,
            transform=transform,
            test_transform=transform,
            target_transform=None,
            download=True,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Get base dataset (train or test)
    if split == "train":
        base_dataset = dataset.dataset_train
    else:
        base_dataset = dataset.dataset_test
    
    # Get class names
    all_class_indices = list(range(dataset.num_classes))
    class_names = dataset.get_class_names(all_class_indices)
    
    # Wrap with captions or class names
    if caption_dir is not None:
        caption_file = Path(caption_dir) / f"{dataset_name}_{split}_captions.json"
        if caption_file.exists():
            # Use create_caption_dataset to load captions
            captioned_dataset = create_caption_dataset(
                base_dataset=base_dataset,
                caption_dir=caption_dir,
                dataset_name=dataset_name,
                split=split,
                sample_strategy="first",  # Use first caption for consistency
                seed=seed,
            )
            if captioned_dataset is not None:
                return captioned_dataset
            print(f"Warning: Failed to create caption dataset from {caption_file}, using class names")
        else:
            print(f"Warning: Caption file not found: {caption_file}, using class names")
    
    # Fallback: use class names
    return ClassNameDataset(base_dataset, class_names)


@torch.no_grad()
def extract_features(
    model,
    dataloader: DataLoader,
    tokenizer,
    device: str,
    max_samples: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract image and text features from a dataset.
    
    Returns:
        image_features: [N, D] normalized image features
        text_features: [N, D] normalized text features  
        labels: [N] class labels
    """
    image_features_list = []
    text_features_list = []
    labels_list = []
    
    num_samples = 0
    for images, labels, captions in tqdm(dataloader, desc="Extracting features"):
        if max_samples is not None and num_samples >= max_samples:
            break
        
        images = images.to(device)
        
        # Encode images
        image_features = model.encode_image(images)
        image_features = F.normalize(image_features, p=2, dim=1)
        
        # Tokenize and encode text
        if isinstance(captions, (list, tuple)):
            text_tokens = tokenizer(captions).to(device)
        else:
            text_tokens = tokenizer([captions]).to(device)
        
        text_features = model.encode_text(text_tokens)
        text_features = F.normalize(text_features, p=2, dim=1)
        
        image_features_list.append(image_features.cpu())
        text_features_list.append(text_features.cpu())
        labels_list.append(labels.cpu())
        
        num_samples += len(images)
    
    image_features = torch.cat(image_features_list, dim=0)
    text_features = torch.cat(text_features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    return image_features, text_features, labels


def compute_paired_similarity(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
) -> float:
    """Compute mean cosine similarity for paired samples."""
    # Diagonal elements are paired
    similarities = (image_features * text_features).sum(dim=1)
    return similarities.mean().item()


def compute_unpaired_similarity(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    num_samples: int = 1000,
) -> float:
    """Compute mean cosine similarity for randomly sampled unpaired samples."""
    n = len(image_features)
    
    # Random sampling without replacement
    num_samples = min(num_samples, n)
    indices_i = torch.randperm(n)[:num_samples]
    indices_j = torch.randperm(n)[:num_samples]
    
    # Ensure unpaired (i != j)
    mask = indices_i == indices_j
    while mask.any():
        indices_j[mask] = torch.randint(0, n, (mask.sum().item(),))
        mask = indices_i == indices_j
    
    similarities = (image_features[indices_i] * text_features[indices_j]).sum(dim=1)
    return similarities.mean().item()


def compute_clip_loss_batchwise(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
    batch_size: int = 64,
) -> float:
    """Compute CLIP contrastive loss batch-wise from extracted features."""
    if not CLIP_AVAILABLE or ClipLoss is None:
        raise ImportError("open_clip required for CLIP loss. Install with: pip install open-clip-torch")
    
    # Create ClipLoss instance
    clip_loss = ClipLoss()
    
    losses = []
    n = len(image_features)
    
    # Process in batches
    for i in range(0, n, batch_size):
        batch_img_feats = image_features[i:i+batch_size]
        batch_txt_feats = text_features[i:i+batch_size]
        
        # Compute batch loss
        batch_loss = clip_loss(batch_img_feats, batch_txt_feats, logit_scale, output_dict=False)
        losses.append(batch_loss.item())
    
    # Return average loss across batches
    return sum(losses) / len(losses) if losses else 0.0


def compute_supervised_contrastive_loss_batchwise(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    labels: torch.Tensor,
    logit_scale: torch.Tensor,
    batch_size: int = 64,
) -> float:
    """
    Compute supervised contrastive loss batch-wise using SupervisedClipLoss.
    Treats same-class samples as positives.
    """
    # Create SupervisedClipLoss instance
    supervised_loss = SupervisedClipLoss()
    
    losses = []
    n = len(image_features)
    
    # Process in batches
    for i in range(0, n, batch_size):
        batch_img_feats = image_features[i:i+batch_size]
        batch_txt_feats = text_features[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        
        # Compute batch loss
        batch_loss = supervised_loss(
            batch_img_feats,
            batch_txt_feats,
            logit_scale,
            labels=batch_labels,
            output_dict=False
        )
        losses.append(batch_loss.item())
    
    # Return average loss across batches
    return sum(losses) / len(losses) if losses else 0.0


def evaluate_dataset(
    dataset_name: str,
    dataloader: DataLoader,
    model,
    tokenizer,
    device: str,
    use_supervised: bool,
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate CLIP metrics on a dataset."""
    print("\n" + "="*60)
    print("Evaluating: " + dataset_name)
    print("="*60)
    
    # Extract logit_scale from pretrained model
    logit_scale = model.logit_scale.exp()
    print(f"Using pretrained logit_scale (temperature): {logit_scale.item():.4f}")
    
    # Extract features
    image_features, text_features, labels = extract_features(
        model, dataloader, tokenizer, device, max_samples
    )
    
    print(f"Extracted features: {len(image_features)} samples")
    
    # Move to device for metric computation
    image_features = image_features.to(device)
    text_features = text_features.to(device)
    labels = labels.to(device)
    
    # Compute metrics
    metrics = {}
    metrics["logit_scale"] = logit_scale.item()  # Store for reference
    
    print("Computing paired similarity...")
    metrics["paired_similarity"] = compute_paired_similarity(image_features, text_features)
    
    print("Computing unpaired similarity...")
    metrics["unpaired_similarity"] = compute_unpaired_similarity(
        image_features, text_features, num_samples=min(1000, len(image_features))
    )
    
    # Compute similarity gap
    metrics["similarity_gap"] = metrics["paired_similarity"] - metrics["unpaired_similarity"]
    
    print("Computing CLIP loss...")
    metrics["clip_loss"] = compute_clip_loss_batchwise(
        image_features, text_features, logit_scale, batch_size=dataloader.batch_size
    )
    
    if use_supervised:
        print("Computing supervised contrastive loss...")
        metrics["supervised_loss"] = compute_supervised_contrastive_loss_batchwise(
            image_features, text_features, labels, logit_scale, batch_size=dataloader.batch_size
        )
    
    # Print results
    print("\nResults:")
    print(f"  Paired similarity:     {metrics['paired_similarity']:.4f}")
    print(f"  Unpaired similarity:   {metrics['unpaired_similarity']:.4f}")
    print(f"  Similarity gap:        {metrics['similarity_gap']:.4f}")
    print(f"  CLIP loss:             {metrics['clip_loss']:.4f}")
    if use_supervised:
        print(f"  Supervised loss:       {metrics['supervised_loss']:.4f}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CLIP metrics on various datasets"
    )
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-16-quickgelu",
        help="CLIP model name (e.g., ViT-B-16, ViT-L-14)",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="openai",
        help="Pretrained weights (e.g., openai, laion400m_e32, laion2b_s34b_b88k)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Cache directory for OpenCLIP model",
    )
    
    # Dataset arguments
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["coco", "cifar100", "cub200"],
        help="Datasets to evaluate (coco for reference, plus target datasets)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Root directory for datasets",
    )
    parser.add_argument(
        "--caption_dir",
        type=str,
        default=None,
        help="Directory containing caption JSON files (optional)",
    )
    parser.add_argument(
        "--coco_dir",
        type=str,
        default="./data/coco",
        help="COCO dataset directory (for reference metrics)",
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for feature extraction",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples per dataset (None = all)",
    )
    parser.add_argument(
        "--use_supervised",
        action="store_true",
        help="Compute supervised contrastive loss (requires class labels)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results (optional)",
    )
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load CLIP model
    model, preprocess, tokenizer = load_clip_model(
        args.model, args.pretrained, args.cache_dir, device
    )
    
    # Evaluate each dataset
    all_results = {}
    
    for dataset_name in args.datasets:
        try:
            if dataset_name == "coco":
                # COCO Captions (reference)
                if not COCO_AVAILABLE:
                    print("Skipping COCO (pycocotools not available)")
                    continue
                
                # Load COCO Captions using torchvision
                coco_dir = Path(args.coco_dir)
                ann_file = coco_dir / "annotations" / "captions_val2017.json"
                img_dir = coco_dir / "val2017"
                
                # Create base COCO dataset
                base_coco = CocoCaptions(
                    root=str(img_dir),
                    annFile=str(ann_file),
                    transform=preprocess,
                )
                
                # Wrapper to return (image, label, caption) format
                class COCOWrapper(Dataset):
                    def __init__(self, coco_dataset, max_samples=None):
                        self.coco_dataset = coco_dataset
                        self.indices = list(range(len(coco_dataset)))
                        if max_samples is not None and max_samples < len(self.indices):
                            random.shuffle(self.indices)
                            self.indices = self.indices[:max_samples]
                    
                    def __len__(self):
                        return len(self.indices)
                    
                    def __getitem__(self, idx):
                        real_idx = self.indices[idx]
                        image, captions = self.coco_dataset[real_idx]
                        # Use first caption
                        caption = captions[0] if captions else ""
                        return image, 0, caption  # label=0 (no class labels for COCO)
                
                dataset = COCOWrapper(base_coco, max_samples=args.max_samples)
                
                # Create dataloader with shuffling for proper metric computation
                dataloader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=True,  # Shuffle to prevent same-class clustering
                    num_workers=args.num_workers,
                    pin_memory=True,
                )
                
                # Evaluate COCO
                metrics = evaluate_dataset(
                    dataset_name=dataset_name,
                    dataloader=dataloader,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    use_supervised=False,  # COCO doesn't have class labels
                    max_samples=args.max_samples,
                )
                
                all_results[dataset_name] = metrics
                
            else:
                # Target datasets - evaluate with and without captions if caption_dir provided
                caption_configs = []
                
                if args.caption_dir is not None:
                    # Evaluate both with and without captions for comparison
                    caption_configs = [
                        ("with_captions", args.caption_dir),
                        ("without_captions", None),
                    ]
                else:
                    # Only evaluate without captions (class names only)
                    caption_configs = [("", None)]
                
                for suffix, caption_dir in caption_configs:
                    dataset = load_target_dataset(
                        dataset_name=dataset_name,
                        data_dir=args.data_dir,
                        split="test",  # Always use test split
                        caption_dir=caption_dir,
                        transform=preprocess,
                        seed=args.seed,
                    )
                    
                    # Create dataloader with shuffling for proper metric computation
                    dataloader = DataLoader(
                        dataset,
                        batch_size=args.batch_size,
                        shuffle=True,  # Shuffle to prevent same-class clustering
                        num_workers=args.num_workers,
                        pin_memory=True,
                    )
                    
                    # Determine result key
                    if suffix:
                        result_key = f"{dataset_name}_{suffix}"
                    else:
                        result_key = dataset_name
                    
                    # Evaluate
                    metrics = evaluate_dataset(
                        dataset_name=result_key,
                        dataloader=dataloader,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        use_supervised=args.use_supervised,
                        max_samples=args.max_samples,
                    )
                    
                    all_results[result_key] = metrics
            
        except Exception as e:
            print(f"Error evaluating {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nModel: {args.model} (pretrained: {args.pretrained})")
    # Get logit_scale from first result (should be same for all)
    if all_results:
        first_result = next(iter(all_results.values()))
        if 'logit_scale' in first_result:
            print(f"Logit scale (temperature): {first_result['logit_scale']:.4f}")
    print(f"Batch size: {args.batch_size}")
    print(f"Random seed: {args.seed}")
    
    print("\nDataset comparison:")
    print(f"{'Dataset':<20} {'Captions':<12} {'Paired Sim':<12} {'Unpaired Sim':<12} {'Gap':<10} {'CLIP Loss':<12}", end="")
    if args.use_supervised:
        print(f"{'Sup Loss':<12}", end="")
    print()
    separator_length = 82 if not args.use_supervised else 95
    print("-" * separator_length)
    
    for dataset_name, metrics in all_results.items():
        # Parse dataset name to extract caption info
        if dataset_name.endswith("_with_captions"):
            base_name = dataset_name[:-14]  # Remove "_with_captions"
            caption_status = "Yes"
        elif dataset_name.endswith("_without_captions"):
            base_name = dataset_name[:-17]  # Remove "_without_captions"
            caption_status = "No"
        else:
            base_name = dataset_name
            caption_status = "N/A" if dataset_name == "coco" else "No"
        
        print(f"{base_name:<20} {caption_status:<12} {metrics['paired_similarity']:<12.4f} "
              f"{metrics['unpaired_similarity']:<12.4f} {metrics['similarity_gap']:<10.4f} "
              f"{metrics['clip_loss']:<12.4f}", end="")
        if args.use_supervised and 'supervised_loss' in metrics:
            print(f"{metrics['supervised_loss']:<12.4f}", end="")
        print()
    
    # Save results to JSON
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get logit_scale from results (should be same for all datasets)
        logit_scale = None
        if all_results:
            first_result = next(iter(all_results.values()))
            logit_scale = first_result.get('logit_scale')
        
        output_data = {
            "model": args.model,
            "pretrained": args.pretrained,
            "logit_scale": logit_scale,
            "batch_size": args.batch_size,
            "random_seed": args.seed,
            "use_supervised": args.use_supervised,
            "results": all_results,
        }
        
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
