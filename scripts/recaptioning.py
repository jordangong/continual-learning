#!/usr/bin/env env python
"""
Recaptioning script for target datasets using OpenAI API.

This script recaptions images from target datasets (CIFAR-100, ImageNet-R, Stanford Cars, 
Aircraft, CUB-200) to reduce the distribution shift between pretraining data (with rich 
captions) and target data (with only class labels).
"""

import argparse
import base64
import json
import os
import random
from io import BytesIO
from pathlib import Path
from typing import List

import numpy as np
import torch
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel
from tqdm import tqdm

# Import dataset classes
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


# Pydantic model for structured outputs
class ImageCaptions(BaseModel):
    """Structured output model for image captions."""
    captions: List[str]


# COCO Captions examples (fallback if real COCO data not available)
COCO_EXAMPLES_FALLBACK = [
    "A woman wearing a net on her head cutting a cake.",
    "A large bus sitting on the side of a street.",
    "A black Honda motorcycle parked in front of a garage.",
    "A room with blue walls and a white sink and door.",
    "A car that seems to be parked illegally behind a legally parked car.",
    "A bathroom with a toilet, sink, and a large mirror.",
    "A giraffe standing in a grassy field with trees in the background.",
    "A group of people sitting around a table eating food.",
    "A pizza sitting on top of a pan covered in toppings.",
    "A young boy holding a baseball bat on a field.",
]


def load_coco_captions(coco_annotations_dir: str, num_examples: int = 10, seed: int = 42) -> List[str]:
    """
    Load real COCO captions from annotation files.
    
    Args:
        coco_annotations_dir: Path to COCO annotations directory
        num_examples: Number of caption examples to sample
        seed: Random seed for sampling
    
    Returns:
        List of caption strings
    """
    # Try to load captions from COCO annotation files
    caption_files = [
        "captions_train2017.json",
        "captions_val2017.json",
        "captions_train2014.json",
        "captions_val2014.json",
    ]
    
    all_captions = []
    for caption_file in caption_files:
        caption_path = Path(coco_annotations_dir) / caption_file
        if caption_path.exists():
            try:
                with open(caption_path, "r") as f:
                    data = json.load(f)
                    # Extract caption texts from annotations
                    captions = [ann["caption"] for ann in data.get("annotations", [])]
                    all_captions.extend(captions)
                    print(f"Loaded {len(captions)} captions from {caption_file}")
            except Exception as e:
                print(f"Warning: Failed to load {caption_file}: {e}")
    
    if len(all_captions) == 0:
        print("Warning: No COCO captions loaded. Using fallback examples.")
        return COCO_EXAMPLES_FALLBACK
    
    # Randomly sample captions
    random.seed(seed)
    sampled_captions = random.sample(all_captions, min(num_examples, len(all_captions)))
    
    print(f"Sampled {len(sampled_captions)} caption examples from {len(all_captions)} total COCO captions")
    return sampled_captions


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    Encode PIL Image to base64 string.
    
    Args:
        image: PIL Image object
        format: Image format (PNG, JPEG, etc.)
    
    Returns:
        Base64-encoded image string
    """
    buffered = BytesIO()
    # Convert RGBA to RGB if needed (for JPEG)
    if image.mode == "RGBA" and format.upper() == "JPEG":
        image = image.convert("RGB")
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def create_system_prompt(num_captions: int, coco_examples: List[str]) -> str:
    """
    Create system prompt with COCO Captions examples.
    
    Args:
        num_captions: Number of captions to generate
        coco_examples: List of COCO caption examples to use
    
    Returns:
        System prompt string
    """
    examples_str = "\n".join([f"- {ex}" for ex in coco_examples])
    
    prompt = f"""Write {num_captions} caption{"s" if num_captions > 1 else ""} for the given image and its class label.
Follow COCO Captions style:

{examples_str}

Guidelines:
- Be descriptive but concise (10-20 words per caption)
- Focus on visual attributes, context, and scene
- Incorporate the class label naturally
- Describe what you see: colors, positions, background, actions
- Vary the captions when generating multiple ones
- Use natural, simple language"""
    
    return prompt


def generate_captions(
    client: OpenAI,
    image: Image.Image,
    class_name: str,
    num_captions: int,
    coco_examples: List[str],
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_retries: int = 3,
) -> List[str]:
    """
    Generate captions for an image using OpenAI API with structured outputs.
    
    Args:
        client: OpenAI client
        image: PIL Image object
        class_name: Class label/name
        num_captions: Number of captions to generate
        model: OpenAI model to use
        temperature: Sampling temperature
        max_retries: Maximum number of retries on failure
    
    Returns:
        List of generated captions
    """
    # Keep track of image processing state
    current_image = image
    image_resized = False
    tried_jpeg = False
    
    # Encode image to base64
    image_base64 = encode_image_to_base64(current_image, format="PNG")
    
    # Create system and user prompts
    system_prompt = create_system_prompt(num_captions, coco_examples)
    user_prompt = f"Class label: {class_name}"
    
    # Retry loop for API calls
    for attempt in range(max_retries):
        try:
            # Vary seed across attempts to get different results if LLM doesn't follow instructions
            current_seed = 42 + attempt
            
            # Call OpenAI API with structured outputs
            completion = client.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                },
                            },
                        ],
                    },
                ],
                response_format=ImageCaptions,
                temperature=temperature,
                seed=current_seed,  # Vary seed across attempts
            )
            
            # Extract captions from structured response
            captions_obj = completion.choices[0].message.parsed
            captions = captions_obj.captions
            
            # Validate that we got the correct number of captions
            if len(captions) != num_captions:
                if attempt < max_retries - 1:
                    print(f"  Got {len(captions)}/{num_captions} captions, retrying with different seed (seed={42 + attempt + 1})...")
                    continue
                else:
                    print(f"  Warning: Got {len(captions)}/{num_captions} captions after {max_retries} attempts")
            
            return captions
            
        except Exception as e:
            error_msg = str(e)
            
            # Check if the error is due to image being too large
            if "Decompressed data too large" in error_msg:
                if not image_resized:
                    # First fallback: Resize to PNG
                    print("  Image too large, resizing to 512x512 PNG...")
                    max_size = 512
                    current_image = image.copy()
                    current_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    # Re-encode the resized image as PNG
                    image_base64 = encode_image_to_base64(current_image, format="PNG")
                    image_resized = True
                    # Retry immediately with resized PNG
                    continue
                elif not tried_jpeg:
                    # Second fallback: Convert to JPEG for smaller file size
                    print("  Resized PNG still too large, converting to JPEG...")
                    # Convert to RGB if necessary (JPEG doesn't support RGBA)
                    if current_image.mode in ("RGBA", "P", "LA"):
                        current_image = current_image.convert("RGB")
                    # Re-encode as JPEG (much smaller file size)
                    image_base64 = encode_image_to_base64(current_image, format="JPEG")
                    tried_jpeg = True
                    # Retry immediately with JPEG
                    continue
            
            if attempt < max_retries - 1:
                print(f"  Error on attempt {attempt + 1}: {e}. Retrying...")
                continue
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                # Return empty list on failure (keeps original caption unchanged)
                return []
    
    return []


def load_dataset(dataset_name: str, data_dir: str, seed: int):
    """
    Load a continual learning dataset.
    
    Args:
        dataset_name: Name of the dataset
        data_dir: Root directory for datasets
        seed: Random seed
    
    Returns:
        ContinualDataset instance
    """
    dataset_name = dataset_name.lower()
    
    # Dummy continual learning config (we'll access all classes)
    # No transform specified - datasets return raw PIL Images by default
    num_steps = 1
    classes_per_step = 1000  # Large number to include all classes
    
    if dataset_name == "cifar100":
        dataset = CIFAR100CL(
            root=data_dir,
            num_steps=num_steps,
            classes_per_step=classes_per_step,
            download=True,
            seed=seed,
        )
    elif dataset_name == "imagenet-r":
        dataset = ImageNetRCL(
            root=data_dir,
            num_steps=num_steps,
            classes_per_step=classes_per_step,
            download=True,
            seed=seed,
            train_ratio=0.8,
        )
    elif dataset_name == "stanford_cars":
        dataset = StanfordCarsCL(
            root=data_dir,
            num_steps=num_steps,
            classes_per_step=classes_per_step,
            download=False,
            seed=seed,
        )
    elif dataset_name == "fgvc_aircraft":
        dataset = FGVCAircraftCL(
            root=data_dir,
            num_steps=num_steps,
            classes_per_step=classes_per_step,
            download=True,
            seed=seed,
            annotation_level="variant",
        )
    elif dataset_name == "cub200":
        dataset = CUB200CL(
            root=data_dir,
            num_steps=num_steps,
            classes_per_step=classes_per_step,
            download=True,
            seed=seed,
        )
    elif dataset_name == "caltech256":
        dataset = Caltech256CL(
            root=data_dir,
            num_steps=num_steps,
            classes_per_step=classes_per_step,
            download=True,
            seed=seed,
            train_ratio=0.8,
        )
    elif dataset_name == "food101":
        dataset = Food101CL(
            root=data_dir,
            num_steps=num_steps,
            classes_per_step=classes_per_step,
            download=True,
            seed=seed,
        )
    elif dataset_name == "oxford_pet":
        dataset = OxfordIIITPetCL(
            root=data_dir,
            num_steps=num_steps,
            classes_per_step=classes_per_step,
            download=True,
            seed=seed,
        )
    elif dataset_name == "imagenet100":
        dataset = ImageNet100CL(
            root=data_dir,
            num_steps=num_steps,
            classes_per_step=classes_per_step,
            download=False,
            seed=seed,
        )
    elif dataset_name == "imagenet-a":
        dataset = ImageNetACL(
            root=data_dir,
            num_steps=num_steps,
            classes_per_step=classes_per_step,
            download=True,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Recaption target datasets using OpenAI API"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "cifar100",
            "cub200",
            "caltech256",
            "stanford_cars",
            "food101",
            "oxford_pet",
            "fgvc_aircraft",
            "imagenet100",
            "imagenet-r",
            "imagenet-a",
        ],
        help="Target dataset to recaption",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Root directory for datasets",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./recaptions",
        help="Output directory for generated captions",
    )
    
    # Captioning arguments
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use (e.g., gpt-4o, gpt-4o-mini)",
    )
    parser.add_argument(
        "--num_captions",
        type=int,
        default=5,
        help="Number of captions to generate per image",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for caption generation",
    )
    parser.add_argument(
        "--coco_annotations_dir",
        type=str,
        default="./data/coco/annotations",
        help="Path to COCO annotations directory (for loading real caption examples)",
    )
    parser.add_argument(
        "--num_coco_examples",
        type=int,
        default=10,
        help="Number of COCO caption examples to use in the prompt",
    )
    
    # Sampling arguments
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of images to caption per class (None = all images)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Number of classes to caption (None = all classes)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    # API arguments
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key (if not set, reads from OPENAI_API_KEY env variable)",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="Custom base URL for OpenAI-compatible API (e.g., http://localhost:8000/v1 for vLLM)",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum number of retries for API calls",
    )
    
    # Other arguments
    parser.add_argument(
        "--split",
        type=str,
        default="both",
        choices=["train", "test", "both"],
        help="Dataset split to use (default: both)",
    )
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Initialize OpenAI client
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    
    # For custom endpoints, API key might not be required
    if not api_key and not args.base_url:
        raise ValueError(
            "OpenAI API key not found. Set --api_key or OPENAI_API_KEY environment variable."
        )
    
    # Use dummy key for custom endpoints that don't require authentication
    if args.base_url and not api_key:
        api_key = "dummy-key"
        print(f"Using custom endpoint: {args.base_url} (no API key required)")
    
    # Initialize client with optional custom base URL
    client_kwargs = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
        print(f"Using custom API endpoint: {args.base_url}")
    
    client = OpenAI(**client_kwargs)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, args.data_dir, args.seed)
    
    # Determine which splits to process
    if args.split == "both":
        splits_to_process = ["train", "test"]
    else:
        splits_to_process = [args.split]
    
    # Load COCO caption examples (once for all splits)
    print(f"\nLoading COCO caption examples from {args.coco_annotations_dir}...")
    coco_examples = load_coco_captions(
        args.coco_annotations_dir,
        num_examples=args.num_coco_examples,
        seed=args.seed
    )
    print(f"Using {len(coco_examples)} COCO caption examples for prompting\n")
    
    # Get class names (same for all splits)
    all_class_indices = list(range(dataset.num_classes))
    class_names = dataset.get_class_names(all_class_indices)
    
    # Limit number of classes if specified (same for all splits)
    if args.num_classes is not None:
        num_classes = min(args.num_classes, dataset.num_classes)
        selected_classes = random.sample(range(dataset.num_classes), num_classes)
        print(f"Randomly selected {num_classes} classes out of {dataset.num_classes}")
    else:
        selected_classes = all_class_indices
        num_classes = dataset.num_classes
    
    # Process each split
    for split in splits_to_process:
        print(f"\n{'='*60}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*60}\n")
        
        # Get the appropriate split data
        if split == "train":
            data = dataset.dataset_train
        else:
            data = dataset.dataset_test
        
        print(f"Dataset loaded: {len(data)} images, {dataset.num_classes} classes")
        
        # Prepare output file
        output_file = output_dir / f"{args.dataset}_{split}_captions.json"
        
        # Load existing captions if resuming
        if output_file.exists():
            print(f"Loading existing captions from {output_file}")
            with open(output_file, "r") as f:
                all_captions = json.load(f)
        else:
            all_captions = {}
        
        # Process each class
        total_images_processed = 0
        total_images_skipped = 0
        
        # Pre-extract all targets for fast class indexing (avoid loading images)
        targets_list = None
        
        # Handle Subset wrapper (common in continual learning datasets)
        dataset_to_check = data
        subset_indices = None
        if hasattr(data, "dataset") and hasattr(data, "indices"):
            # This is a Subset - access underlying dataset
            dataset_to_check = data.dataset
            subset_indices = data.indices
        
        if hasattr(dataset_to_check, "targets"):
            # Many torchvision datasets have a targets attribute (CIFAR, MNIST, etc.)
            targets_list = dataset_to_check.targets
            if subset_indices is not None:
                targets_list = [targets_list[i] for i in subset_indices]
        elif hasattr(dataset_to_check, "samples"):
            # ImageFolder datasets have samples as (path, target) tuples
            targets_list = [s[1] for s in dataset_to_check.samples]
            if subset_indices is not None:
                targets_list = [targets_list[i] for i in subset_indices]
        elif hasattr(dataset_to_check, "_samples"):
            # Stanford Cars uses _samples (private attribute)
            targets_list = [s[1] for s in dataset_to_check._samples]
            if subset_indices is not None:
                targets_list = [targets_list[i] for i in subset_indices]
        elif hasattr(dataset_to_check, "_labels"):
            # FGVC Aircraft, Food101, Oxford Pet use _labels (private attribute)
            targets_list = dataset_to_check._labels
            if subset_indices is not None:
                targets_list = [targets_list[i] for i in subset_indices]
        else:
            # Fallback: load items one by one (slow)
            print("Warning: No fast target access found, loading dataset items (slower)...")
            targets_list = [data[idx][1] for idx in range(len(data))]
        
        for class_enum_idx, class_idx in enumerate(selected_classes, 1):
            # Replace underscores with spaces in class names
            class_name = class_names[class_idx].replace("_", " ")
            
            # Get indices for this class (fast - no image loading)
            class_indices = [idx for idx, label in enumerate(targets_list) if label == class_idx]
            
            if len(class_indices) == 0:
                print(f"  Warning: No samples found for class {class_idx} ({class_name})")
                continue
            
            # Sample images if specified
            if args.num_samples is not None:
                num_samples = min(args.num_samples, len(class_indices))
                sampled_indices = random.sample(class_indices, num_samples)
            else:
                sampled_indices = class_indices
            
            # Process each image with progress bar
            class_captions = {}
            progress_desc = f"Class {class_enum_idx}/{len(selected_classes)}: {class_name}"
            for img_idx in tqdm(sampled_indices, desc=progress_desc, unit="img"):
                # Skip if already processed with correct number of captions
                img_key = f"{class_idx}_{img_idx}"
                if img_key in all_captions:
                    existing_captions = all_captions[img_key]["captions"]
                    # Only skip if we have the expected number of captions
                    # Reprocess if empty or incomplete (e.g., from API errors/context limits)
                    if len(existing_captions) == args.num_captions:
                        total_images_skipped += 1
                        continue
                    else:
                        print(f"  Reprocessing {img_key}: has {len(existing_captions)}/{args.num_captions} captions")
                
                # Load image (should be PIL Image with transform=None)
                image, label = data[img_idx]
                
                # Try to get the image path if available
                image_path = None
                
                # Handle Subset wrapper for image paths
                dataset_for_path = data
                actual_idx = img_idx
                if hasattr(data, "dataset") and hasattr(data, "indices"):
                    # This is a Subset - need to map to underlying dataset index
                    dataset_for_path = data.dataset
                    actual_idx = data.indices[img_idx]
                
                if hasattr(dataset_for_path, 'samples'):
                    # ImageFolder-style datasets (ImageNet-R, CUB-200)
                    image_path = dataset_for_path.samples[actual_idx][0]
                elif hasattr(dataset_for_path, '_samples'):
                    # Stanford Cars uses _samples
                    image_path = dataset_for_path._samples[actual_idx][0]
                elif hasattr(dataset_for_path, 'imgs'):
                    # Alternative ImageFolder attribute
                    image_path = dataset_for_path.imgs[actual_idx][0]
                elif hasattr(dataset_for_path, 'images'):
                    # Custom ImageListDataset
                    image_path = dataset_for_path.images[actual_idx]
                elif hasattr(dataset_for_path, '_image_files'):
                    # Food101 uses _image_files
                    image_path = str(dataset_for_path._image_files[actual_idx])
                elif hasattr(dataset_for_path, 'data_dir') and hasattr(dataset_for_path, 'image_names'):
                    # Custom datasets with image_names attribute
                    image_path = str(Path(dataset_for_path.data_dir) / dataset_for_path.image_names[actual_idx])
                
                # Ensure we have a PIL Image (handle edge cases)
                if isinstance(image, Image.Image):
                    pass
                elif isinstance(image, torch.Tensor):
                    # Convert tensor to PIL (shouldn't happen with transform=None)
                    image = image.numpy().transpose(1, 2, 0)
                    image = (image * 255).astype(np.uint8)
                    image = Image.fromarray(image)
                elif isinstance(image, np.ndarray):
                    # Convert numpy array to PIL
                    if image.dtype != np.uint8:
                        image = (image * 255).astype(np.uint8)
                    image = Image.fromarray(image)
                else:
                    raise TypeError(f"Unexpected image type: {type(image)}")
                
                # Generate captions
                captions = generate_captions(
                    client=client,
                    image=image,
                    class_name=class_name,
                    num_captions=args.num_captions,
                    coco_examples=coco_examples,
                    model=args.model,
                    temperature=args.temperature,
                    max_retries=args.max_retries,
                )
                
                # Store captions
                if len(captions) > 0:
                    caption_data = {
                        "class_idx": class_idx,
                        "class_name": class_name,
                        "image_idx": img_idx,
                        "captions": captions,
                    }
                    # Add image path if available
                    if image_path is not None:
                        caption_data["image_path"] = image_path
                    
                    class_captions[img_key] = caption_data
                    total_images_processed += 1
                
                # Save periodically (every 10 images)
                if total_images_processed % 10 == 0:
                    all_captions.update(class_captions)
                    with open(output_file, "w") as f:
                        json.dump(all_captions, f, indent=2)
            
            # Update all captions with class captions
            all_captions.update(class_captions)
        
        # Save final results
        with open(output_file, "w") as f:
            json.dump(all_captions, f, indent=2)
        
        print("\n" + "="*60)
        print(f"{split.upper()} split complete!")
        print(f"Total images processed: {total_images_processed}")
        print(f"Total images skipped (already done): {total_images_skipped}")
        print(f"Captions saved to: {output_file}")
        print("="*60)


if __name__ == "__main__":
    main()
