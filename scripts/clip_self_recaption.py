#!/usr/bin/env python
"""
CLIP Self-Recaptioning: Learn text embeddings directly from CLIP

Learn per-sample text token embeddings to minimize CLIP loss without external VLM.
Inspired by DeepDream but for text tokens.

Key features:
- Frozen CLIP encoders, learn sample-wise text tokens
- CLIP contrastive loss as main objective  
- K-nearest vocab regularization for semantic grounding
- Optional supervised contrastive loss
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.datasets import CIFAR100CL, CUB200CL, ImageNetACL, ImageNetRCL

try:
    from torchvision.datasets import CocoCaptions
    COCO_AVAILABLE = True
except ImportError:
    COCO_AVAILABLE = False
    CocoCaptions = None

try:
    from open_clip.loss import ClipLoss
    from src.trainers.supervised_contrastive_loss import SupervisedClipLoss
    CLIP_LOSSES_AVAILABLE = True
except ImportError:
    CLIP_LOSSES_AVAILABLE = False


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class InfiniteSampler(torch.utils.data.Sampler):
    """Sampler that generates infinite random indices without restarting dataloader."""
    
    def __init__(self, dataset_size: int, shuffle: bool = True):
        self.dataset_size = dataset_size
        self.shuffle = shuffle
    
    def __iter__(self):
        while True:  # Infinite loop
            if self.shuffle:
                indices = torch.randperm(self.dataset_size).tolist()
            else:
                indices = list(range(self.dataset_size))
            for idx in indices:
                yield idx
    
    def __len__(self):
        return float('inf')  # Infinite length


class LearnableTokenDataset(Dataset):
    """Dataset wrapper providing sample indices for learnable token lookup."""
    
    def __init__(self, base_dataset: Dataset, class_names: List[str]):
        self.base_dataset = base_dataset
        self.class_names = class_names
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        class_name = self.class_names[label]
        return image, label, class_name, idx


class CLIPSelfRecaptioner(nn.Module):
    """
    Learn per-sample text token embeddings to minimize CLIP loss.
    
    Text template: [SOS] + class_name_tokens + learnable_tokens + [EOS]
    
    Supports two token learning modes:
    - 'embedding': Learn embeddings directly in R^D (continuous, recommended)
    - 'discrete': Learn soft token IDs with Gumbel-Softmax (discrete, experimental)
    
    Initialization options (for embedding mode):
    - init_method="class_name": Initialize from class name tokens (best semantic grounding)
    - init_method="vocab": Sample random vocab tokens (good semantic bootstrap)
    - init_method="gaussian": Sample from Gaussian noise (original method)
    """
    
    def __init__(
        self,
        clip_model: nn.Module,
        tokenizer,
        num_samples: int,
        num_learnable_tokens: int = 10,
        device: str = "cuda",
        token_mode: str = "embedding",  # 'embedding' or 'discrete'
        gumbel_temperature: float = 1.0,
        vocab_size: int = 49408,
        dtype: Optional[torch.dtype] = None,  # Training precision (bf16/fp16/fp32)
        init_method: str = "class_name",  # 'class_name', 'vocab', or 'gaussian'
        class_names: Optional[list] = None,  # Required for init_method='class_name'
    ):
        super().__init__()
        
        # Freeze CLIP
        for param in clip_model.parameters():
            param.requires_grad = False
        
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self.device = device
        self.num_learnable_tokens = num_learnable_tokens
        self.token_mode = token_mode
        self.gumbel_temperature = gumbel_temperature
        self.vocab_size = vocab_size
        self.dtype = dtype if dtype is not None else torch.float32
        self.init_method = init_method
        
        # Validate class_names if needed
        if init_method == "class_name" and class_names is None:
            raise ValueError("class_names must be provided when init_method='class_name'")
        if init_method == "class_name" and len(class_names) != num_samples:
            raise ValueError(f"len(class_names)={len(class_names)} must match num_samples={num_samples}")
        
        # Get dimensions from OpenCLIP model
        self.embed_dim = clip_model.transformer.width
        
        # Get token embedding matrix from OpenCLIP model
        token_emb = clip_model.token_embedding.weight
        
        self.register_buffer("token_embedding_matrix", token_emb.clone())
        
        # Initialize learnable parameters based on mode
        if token_mode == "embedding":
            # Option A: Reparameterized embeddings (direction × scale)
            # This decouples direction learning from magnitude, ensuring stable norms
            
            if init_method == "class_name":
                # Initialize from class name token embeddings (best semantic grounding)
                # Each sample's tokens initialized from its class name
                init_embeddings_list = []
                
                for class_name in class_names:
                    # Tokenize class name (e.g., "dog" -> [49406, 1929, 49407])
                    class_tokens = tokenizer([class_name]).squeeze(0)  # [seq_len]
                    
                    # Remove SOS/EOS tokens, keep only content tokens
                    # SOS is first token (49406), EOS is last token (49407)
                    content_tokens = class_tokens[1:-1]  # Remove SOS and EOS
                    
                    # Handle empty class names
                    if len(content_tokens) == 0:
                        # Fallback to random vocab tokens
                        content_tokens = torch.randint(0, vocab_size, (1,), device=device)
                    
                    # Get embeddings for content tokens
                    class_embeddings = token_emb[content_tokens]  # [num_class_tokens, D]
                    
                    # Truncate or repeat to match num_learnable_tokens
                    if len(class_embeddings) >= num_learnable_tokens:
                        # Truncate: take first N tokens
                        sample_embeddings = class_embeddings[:num_learnable_tokens]
                    else:
                        # Repeat: cycle through tokens
                        repeats = (num_learnable_tokens + len(class_embeddings) - 1) // len(class_embeddings)
                        sample_embeddings = class_embeddings.repeat(repeats, 1)[:num_learnable_tokens]
                    
                    init_embeddings_list.append(sample_embeddings)
                
                # Stack all samples: [num_samples, num_learnable_tokens, embed_dim]
                init_embeddings = torch.stack(init_embeddings_list, dim=0)
                
                # Extract directions (unit vectors) and scales (norms)
                init_scales = torch.norm(init_embeddings, dim=-1, keepdim=True)  # [N, T, 1]
                init_directions = F.normalize(init_embeddings, dim=-1)  # [N, T, D]
                
            elif init_method == "vocab":
                # Initialize from randomly sampled real token embeddings
                # This provides better semantic bootstrap than Gaussian noise
                total_tokens_needed = num_samples * num_learnable_tokens
                
                # Randomly sample token indices (with replacement)
                sampled_indices = torch.randint(
                    0, vocab_size, (total_tokens_needed,), device=device
                )
                
                # Get sampled token embeddings: [total_tokens_needed, embed_dim]
                sampled_embeddings = token_emb[sampled_indices]
                
                # Reshape to [num_samples, num_learnable_tokens, embed_dim]
                sampled_embeddings = sampled_embeddings.view(
                    num_samples, num_learnable_tokens, self.embed_dim
                )
                
                # Extract directions (unit vectors) and scales (norms)
                init_scales = torch.norm(sampled_embeddings, dim=-1, keepdim=True)  # [N, T, 1]
                init_directions = F.normalize(sampled_embeddings, dim=-1)  # [N, T, D]
                
            elif init_method == "gaussian":
                # Initialize from Gaussian noise (original method)
                # Sample from token embedding distribution
                token_mean = token_emb.mean(dim=0, keepdim=True)  # [1, D]
                token_std = token_emb.std(dim=0, keepdim=True)    # [1, D]
                init_embeddings = token_mean + token_std * torch.randn(
                    num_samples, num_learnable_tokens, self.embed_dim, device=device
                )
                
                # Extract directions (unit vectors) and scales (norms)
                init_scales = torch.norm(init_embeddings, dim=-1, keepdim=True)  # [N, T, 1]
                init_directions = F.normalize(init_embeddings, dim=-1)  # [N, T, D]
            else:
                raise ValueError(f"Unknown init_method: {init_method}. Use 'class_name', 'vocab', or 'gaussian'")
            
            # Store average token norm for reference
            token_norms = torch.norm(token_emb, dim=1)
            avg_token_norm = token_norms.mean()
            self.register_buffer("avg_token_norm", avg_token_norm)
            
            # Learnable parameters: directions (unit norm) and scales
            self.learnable_directions = nn.Parameter(init_directions.to(dtype=self.dtype))
            self.learnable_scales = nn.Parameter(init_scales.to(dtype=self.dtype))
        elif token_mode == "discrete":
            # Option B: Learn soft token IDs (logits over vocabulary)
            # Initialize logits with small noise (uniform over vocab initially)
            self.learnable_logits = nn.Parameter(
                (torch.randn(num_samples, num_learnable_tokens, vocab_size, device=device) * 0.01
                ).to(dtype=self.dtype)
            )
        else:
            raise ValueError(f"Unknown token_mode: {token_mode}. Use 'embedding' or 'discrete'")
        
        # Special tokens
        self.sot_token = 49406
        self.eot_token = 49407
        self.context_length = 77
    
    def get_learnable_embeddings(self, sample_indices: torch.Tensor, hard: bool = False):
        """
        Get learnable token embeddings for given sample indices.
        
        Args:
            sample_indices: [batch_size] indices of samples
            hard: If True and token_mode='discrete', use hard argmax instead of soft Gumbel
        
        Returns:
            embeddings: [batch_size, num_learnable_tokens, embed_dim]
        """
        if self.token_mode == 'embedding':
            # Reparameterized: normalize directions to unit sphere, then scale
            # This ensures stable norms throughout training
            directions = F.normalize(self.learnable_directions[sample_indices], dim=-1)
            scales = self.learnable_scales[sample_indices]
            return directions * scales  # [batch_size, num_tokens, embed_dim]
        
        elif self.token_mode == "discrete":
            # Get logits for these samples
            logits = self.learnable_logits[sample_indices]  # [batch, num_tokens, vocab]
            
            if hard:
                # Hard selection (argmax) - for inference/interpretation
                token_ids = logits.argmax(dim=-1)  # [batch, num_tokens]
                embeddings = self.token_embedding_matrix[token_ids]
            else:
                # Soft selection with Gumbel-Softmax (for training)
                # Add Gumbel noise
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
                gumbel_logits = (logits + gumbel_noise) / self.gumbel_temperature
                
                # Softmax to get soft token probabilities
                soft_probs = F.softmax(gumbel_logits, dim=-1)  # [batch, num_tokens, vocab]
                
                # Weighted sum of embeddings
                embeddings = torch.einsum('btv,ve->bte', soft_probs, self.token_embedding_matrix)
                # [batch, num_tokens, vocab] @ [vocab, embed] -> [batch, num_tokens, embed]
            
            return embeddings
        
        else:
            raise ValueError(f"Unknown token_mode: {self.token_mode}")
    
    def build_text_embeddings(self, class_names: List[str], sample_indices: torch.Tensor, hard: bool = False):
        """Build text embeddings: [SOS] + class + learnable + [EOT]"""
        batch_size = len(class_names)
        embeddings = torch.zeros(
            batch_size, self.context_length, self.embed_dim,
            device=self.device, dtype=self.token_embedding_matrix.dtype
        )
        eot_positions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        
        # Get learnable embeddings for all samples in batch
        learnable_embeds_batch = self.get_learnable_embeddings(sample_indices, hard=hard)
        
        for i, class_name in enumerate(class_names):
            # Tokenize class name
            text = f"a photo of a {class_name.replace('_', ' ')}"
            # encode() returns raw tokens WITHOUT SOS/EOT
            text_tokens = self.tokenizer.encode(text)
            
            # Get embeddings: [SOS] + text + learnable + [EOT]
            sot_embed = self.token_embedding_matrix[self.tokenizer.sot_token_id].unsqueeze(0)
            text_embeds = self.token_embedding_matrix[text_tokens]
            
            # Get learnable tokens for this sample
            learnable_embeds = learnable_embeds_batch[i]
            
            # Concatenate: [SOS] + text + learnable + [EOT]
            eot_embed = self.token_embedding_matrix[self.eot_token].unsqueeze(0)
            combined = torch.cat([sot_embed, text_embeds, learnable_embeds, eot_embed], dim=0)
            
            # Record EOT position (after SOS + text + learnable)
            eot_pos = 1 + len(text_tokens) + self.num_learnable_tokens
            eot_positions[i] = eot_pos
            
            # Copy to output
            seq_len = min(combined.shape[0], self.context_length)
            embeddings[i, :seq_len] = combined[:seq_len]
        
        return embeddings, eot_positions
    
    def encode_text_from_embeddings(self, text_embeddings: torch.Tensor, eot_positions: torch.Tensor):
        """Encode text using custom embeddings (following OpenCLIP's encode_text)."""
        # Cast to transformer's dtype
        cast_dtype = self.clip_model.transformer.get_cast_dtype()
        
        # Add positional embeddings (with dtype casting)
        x = text_embeddings.to(cast_dtype) + self.clip_model.positional_embedding.to(cast_dtype)
        
        # Transformer (with attention mask, no permutation needed - OpenCLIP expects batch-first)
        x = self.clip_model.transformer(x, attn_mask=self.clip_model.attn_mask)
        
        # Layer norm
        x = self.clip_model.ln_final(x)
        
        # Extract at EOT positions
        batch_size = x.shape[0]
        text_features = x[torch.arange(batch_size), eot_positions]
        
        # Text projection (handle both nn.Linear and matrix multiplication)
        if self.clip_model.text_projection is not None:
            if isinstance(self.clip_model.text_projection, nn.Linear):
                text_features = self.clip_model.text_projection(text_features)
            else:
                text_features = text_features @ self.clip_model.text_projection
        
        return text_features
    
    def forward(self, images: torch.Tensor, class_names: List[str], sample_indices: torch.Tensor):
        """Get image and text features."""
        # Encode images (frozen)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
        
        # Build and encode text
        text_embeddings, eot_positions = self.build_text_embeddings(class_names, sample_indices)
        text_features = self.encode_text_from_embeddings(text_embeddings, eot_positions)
        
        # Normalize
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        return image_features, text_features


def compute_vocab_distance_loss(learnable_embeds: torch.Tensor, token_matrix: torch.Tensor, k: int = 5):
    """
    K-nearest vocab distance for semantic grounding using cosine distance.
    
    Uses cosine distance (1 - cosine_similarity) to measure semantic difference
    while being invariant to magnitude. This allows learnable scales to remain
    stable without being penalized by the vocab loss.
    
    Args:
        learnable_embeds: [batch, num_tokens, embed_dim] learnable embeddings
        token_matrix: [vocab_size, embed_dim] vocabulary embeddings
        k: Number of nearest neighbors
    """
    tokens_flat = learnable_embeds.reshape(-1, learnable_embeds.shape[-1])  # [N, D]
    
    # Normalize for cosine similarity
    tokens_norm = F.normalize(tokens_flat, dim=-1)  # [N, D]
    vocab_norm = F.normalize(token_matrix, dim=-1)  # [vocab_size, D]
    
    # Compute cosine similarity: [N, vocab_size]
    cosine_sim = torch.matmul(tokens_norm, vocab_norm.t())
    
    # Convert to cosine distance: [N, vocab_size]
    cosine_dist = 1.0 - cosine_sim
    
    # Get k-nearest neighbors (smallest distances)
    topk_distances, _ = cosine_dist.topk(k, dim=1, largest=False)
    
    return topk_distances.mean()


class COCOCaptionWrapper(Dataset):
    """Wrapper for COCO Captions that returns a single caption per image."""
    def __init__(self, coco_dataset, max_samples: Optional[int] = None, random_caption: bool = False):
        self.coco_dataset = coco_dataset
        self.random_caption = random_caption
        self.indices = list(range(len(coco_dataset)))
        if max_samples is not None and max_samples < len(self.indices):
            random.shuffle(self.indices)
            self.indices = self.indices[:max_samples]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, captions = self.coco_dataset[real_idx]
        # Select caption: random or first
        if captions:
            caption = random.choice(captions) if self.random_caption else captions[0]
        else:
            caption = ""
        return image, caption


def load_coco_reference(
    coco_root: str,
    coco_ann_file: str,
    transform,
    max_samples: Optional[int] = None,
    random_caption: bool = False,
):
    """Load COCO Captions dataset as reference."""
    if not COCO_AVAILABLE:
        raise ImportError("CocoCaptions not available. Install: pip install torchvision")
    
    # Load base COCO dataset
    base_dataset = CocoCaptions(root=coco_root, annFile=coco_ann_file, transform=transform)
    
    # Wrap to return (image, caption) instead of (image, list_of_captions)
    dataset = COCOCaptionWrapper(base_dataset, max_samples=max_samples, random_caption=random_caption)
    
    print(f"Loaded COCO reference: {len(dataset)} samples")
    return dataset


def load_dataset(dataset_name: str, split: str, transform, data_dir: str = "./data", seed: int = 42):
    """Load dataset and return with class names."""
    # Use 1 step with all classes to get full dataset
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
    
    # Extract the actual train or test dataset from the continual learning wrapper
    if split == "train":
        base_dataset = dataset.dataset_train
    else:
        base_dataset = dataset.dataset_test
    
    # Get class names from the wrapper
    all_indices = list(range(dataset.num_classes))
    class_names = dataset.get_class_names(all_indices)
    
    return base_dataset, class_names


def main():
    parser = argparse.ArgumentParser(description="CLIP Self-Recaptioning")
    
    # Dataset
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar100", "cub200", "imagenet-r", "imagenet-a"],
                        help="Target dataset for learning text tokens")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Root directory for datasets")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"],
                        help="Dataset split to use")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples for testing (default: use all)")
    
    # Model
    parser.add_argument("--model", type=str, default="ViT-B-16-quickgelu",
                        help="OpenCLIP model name (e.g., ViT-B-16, ViT-L-14)")
    parser.add_argument("--pretrained", type=str, default="openai",
                        help="Pretrained weights source (e.g., openai, laion2b_s34b_b79k)")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Cache directory for OpenCLIP model")
    
    # Token learning mode
    parser.add_argument("--token_mode", type=str, default="embedding", choices=["embedding", "discrete"],
                        help="Token learning mode: 'embedding' (continuous) or 'discrete' (Gumbel-Softmax)")
    parser.add_argument("--num_tokens", type=int, default=10,
                        help="Number of learnable tokens per sample")
    parser.add_argument("--gumbel_temperature", type=float, default=1.0,
                        help="Gumbel-Softmax temperature for discrete mode (lower = more discrete)")
    parser.add_argument("--init_method", type=str, default="class_name",
                        choices=["class_name", "vocab", "gaussian"],
                        help="Initialization method: 'class_name' (from class names), 'vocab' (random tokens), 'gaussian' (noise)")
    
    # Training
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate for optimizer")
    parser.add_argument("--vocab_loss_weight", type=float, default=0.5,
                        help="Weight for K-nearest vocabulary distance loss")
    parser.add_argument("--vocab_k", type=int, default=5,
                        help="K for K-nearest vocabulary tokens (5-10 recommended, higher = more flexibility)")
    parser.add_argument("--use_supervised", action="store_true",
                        help="Enable supervised contrastive loss (uses class labels)")
    
    # COCO reference dataset (optional)
    parser.add_argument("--use_coco_reference", action="store_true",
                        help="Use COCO Captions as reference dataset for richer gradients")
    parser.add_argument("--coco_random_caption", action="store_true",
                        help="Randomly select caption for each COCO sample (default: use first)")
    parser.add_argument("--coco_root", type=str, default="./data/coco/train2017",
                        help="Path to COCO images directory")
    parser.add_argument("--coco_ann_file", type=str, default="./data/coco/annotations/captions_train2017.json",
                        help="Path to COCO captions annotation file")
    parser.add_argument("--coco_samples", type=int, default=10000,
                        help="Number of COCO samples to use as reference")
    parser.add_argument("--coco_loss_weight", type=float, default=0.2,
                        help="Weight for COCO reference loss component")
    
    # Mixed precision training
    parser.add_argument("--use_amp", action="store_true",
                        help="Enable automatic mixed precision training")
    parser.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"],
                        help="Mixed precision dtype (bfloat16 recommended for stability)")
    
    # Output
    parser.add_argument("--output", type=str, default="./learnable_tokens",
                        help="Output directory for learned tokens and training history")
    parser.add_argument("--save_interval", type=int, default=1,
                        help="Save checkpoint every N epochs (default: 1, save every epoch)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for training (cuda or cpu)")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    if not CLIP_LOSSES_AVAILABLE:
        raise ImportError("CLIP losses not available. Install: pip install open-clip-torch")
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup mixed precision training
    if args.use_amp:
        amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
        print(f"Mixed precision training enabled: {args.amp_dtype}")
        # GradScaler only needed for float16, not bfloat16
        use_scaler = (args.amp_dtype == "float16")
        if use_scaler:
            scaler = torch.amp.GradScaler(device=args.device)
            print("Using GradScaler for float16")
    else:
        amp_dtype = None
        use_scaler = False
        scaler = None
        print("Mixed precision training disabled")
    
    # Load CLIP
    print(f"\nLoading CLIP: {args.model} (pretrained: {args.pretrained})")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        args.model,
        pretrained=args.pretrained,
        cache_dir=args.cache_dir,
    )
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(args.model)
    
    # Load dataset
    print(f"\nLoading dataset: {args.dataset} ({args.split} split)")
    dataset, class_names = load_dataset(args.dataset, args.split, preprocess, args.data_dir, args.seed)
    
    # Limit samples
    if args.max_samples and args.max_samples < len(dataset):
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        dataset = Subset(dataset, indices[:args.max_samples])
    
    num_samples = len(dataset)
    print(f"Samples: {num_samples}, Classes: {len(class_names)}")
    
    # Estimate storage (accounting for precision)
    embed_dim = clip_model.transformer.width if hasattr(clip_model, 'transformer') else 512
    bytes_per_param = 4 if not args.use_amp else (2 if args.amp_dtype in ["float16", "bfloat16"] else 4)
    storage_mb = (num_samples * args.num_tokens * embed_dim * bytes_per_param) / (1024 * 1024)
    print(f"Estimated storage: {storage_mb:.2f} MiB ({amp_dtype if args.use_amp else 'float32'})")
    
    # Wrap dataset
    wrapped_dataset = LearnableTokenDataset(dataset, class_names)
    dataloader = DataLoader(
        wrapped_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Create per-sample class names list for initialization
    # class_names is a mapping {label_id: name}, we need [name_for_sample_0, name_for_sample_1, ...]
    if args.init_method == "class_name":
        print(f"\nBuilding per-sample class names for {num_samples} samples...")
        
        # Fast label extraction (avoid loading images)
        dataset_to_check = dataset
        subset_indices = None
        
        # Handle Subset wrapper
        if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
            dataset_to_check = dataset.dataset
            subset_indices = dataset.indices
            print("Detected Subset wrapper, accessing underlying dataset")
        
        # Try different fast access methods
        targets_list = None
        if hasattr(dataset_to_check, "targets"):
            # CIFAR, MNIST, etc.
            targets_list = dataset_to_check.targets
            if isinstance(targets_list, torch.Tensor):
                targets_list = targets_list.tolist()
            if subset_indices is not None:
                targets_list = [targets_list[i] for i in subset_indices]
            print("Using dataset.targets for fast indexing")
        elif hasattr(dataset_to_check, "samples"):
            # ImageFolder datasets
            targets_list = [s[1] for s in dataset_to_check.samples]
            if subset_indices is not None:
                targets_list = [targets_list[i] for i in subset_indices]
            print("Using dataset.samples for fast indexing")
        elif hasattr(dataset_to_check, "_samples"):
            # Stanford Cars
            targets_list = [s[1] for s in dataset_to_check._samples]
            if subset_indices is not None:
                targets_list = [targets_list[i] for i in subset_indices]
            print("Using dataset._samples for fast indexing")
        elif hasattr(dataset_to_check, "_labels"):
            # FGVC Aircraft, Food101, Oxford Pet
            targets_list = dataset_to_check._labels
            if isinstance(targets_list, torch.Tensor):
                targets_list = targets_list.tolist()
            if subset_indices is not None:
                targets_list = [targets_list[i] for i in subset_indices]
            print("Using dataset._labels for fast indexing")
        else:
            # Fallback: slow method (load dataset items)
            print("No fast target access found, loading dataset items (slower)...")
            targets_list = [dataset[i][1] for i in range(num_samples)]
        
        # Map labels to class names
        sample_class_names = [class_names[label] for label in targets_list]
    else:
        sample_class_names = None  # Not needed for other init methods
    
    # Create recaptioner
    print(f"\nInitializing with {args.num_tokens} learnable tokens per sample (mode: {args.token_mode}, init: {args.init_method})")
    recaptioner = CLIPSelfRecaptioner(
        clip_model=clip_model,
        tokenizer=tokenizer,
        num_samples=num_samples,
        num_learnable_tokens=args.num_tokens,
        device=device,
        token_mode=args.token_mode,
        gumbel_temperature=args.gumbel_temperature,
        dtype=amp_dtype if args.use_amp else None,  # Match training precision
        init_method=args.init_method,
        class_names=sample_class_names,  # Per-sample class names
    )
    
    # Count learnable parameters
    if args.token_mode == "embedding":
        num_params = recaptioner.learnable_directions.numel() + recaptioner.learnable_scales.numel()
        param_dtype = recaptioner.learnable_directions.dtype
    else:  # discrete
        num_params = recaptioner.learnable_logits.numel()
        param_dtype = recaptioner.learnable_logits.dtype
    print(f"Learnable parameters: {num_params:,} ({param_dtype})")
    
    # Load COCO reference dataset (optional)
    coco_dataloader = None
    if args.use_coco_reference:
        print("\nLoading COCO reference dataset...")
        coco_dataset = load_coco_reference(
            args.coco_root,
            args.coco_ann_file,
            preprocess,
            args.coco_samples,
            args.coco_random_caption
        )
        # Use infinite sampler to prevent dataloader restart
        coco_sampler = InfiniteSampler(len(coco_dataset), shuffle=True)
        coco_dataloader = DataLoader(
            coco_dataset,
            batch_size=args.batch_size,
            sampler=coco_sampler,  # Use infinite sampler instead of shuffle
            num_workers=args.num_workers,
            pin_memory=True,
        )
        coco_iter = iter(coco_dataloader)
    
    # Loss functions
    clip_loss_fn = SupervisedClipLoss() if args.use_supervised else ClipLoss()
    # Always create both loss functions for monitoring
    supervised_clip_loss_fn = SupervisedClipLoss()
    unsupervised_clip_loss_fn = ClipLoss()
    
    # Optimizer - optimize different parameters based on mode
    if args.token_mode == "embedding":
        optimizer = torch.optim.AdamW([recaptioner.learnable_directions, recaptioner.learnable_scales], lr=args.lr)
    else:  # discrete
        optimizer = torch.optim.AdamW([recaptioner.learnable_logits], lr=args.lr)
    
    # Get logit scale
    logit_scale = clip_model.logit_scale.exp() if hasattr(clip_model, 'logit_scale') else torch.tensor(100.0, device=device)
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting optimization for {args.epochs} epochs")
    print(f"{'='*60}\n")
    
    history = []
    for epoch in range(args.epochs):
        recaptioner.train()
        total_clip_loss = 0.0
        total_supervised_clip_loss = 0.0  # Always monitor both
        total_unsupervised_clip_loss = 0.0
        total_vocab_loss = 0.0
        total_coco_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, labels, class_names_batch, sample_indices in pbar:
            images = images.to(device)
            labels = labels.to(device)
            sample_indices = sample_indices.to(device)
            
            # Forward pass with optional mixed precision
            with torch.amp.autocast(device_type=args.device, enabled=args.use_amp, dtype=amp_dtype if args.use_amp else torch.float32):
                # Forward on target dataset
                image_features, text_features = recaptioner(images, class_names_batch, sample_indices)
                
                # CLIP contrastive loss (main objective)
                # Always compute both supervised and unsupervised for monitoring
                supervised_clip_loss = supervised_clip_loss_fn(image_features, text_features, logit_scale, labels=labels)
                unsupervised_clip_loss = unsupervised_clip_loss_fn(image_features, text_features, logit_scale)
                
                # Choose which one to optimize
                if args.use_supervised:
                    clip_loss = supervised_clip_loss
                else:
                    clip_loss = unsupervised_clip_loss
                
                # Vocab distance loss (semantic grounding)
                learnable_embeds = recaptioner.get_learnable_embeddings(sample_indices, hard=False)
                vocab_loss = compute_vocab_distance_loss(learnable_embeds, recaptioner.token_embedding_matrix, k=args.vocab_k)
                
                # COCO reference loss (optional)
                # Purpose: Mix COCO image-caption pairs as additional negatives in contrastive batch
                # Method: Concatenate COCO pairs with target pairs, compute contrastive loss on combined batch
                coco_loss = torch.tensor(0.0, device=device)
                if args.use_coco_reference:
                    # Infinite sampler ensures dataloader never exhausts
                    coco_images, coco_captions = next(coco_iter)
                    coco_images = coco_images.to(device)
                    
                    # Encode COCO images (frozen)
                    with torch.no_grad():
                        coco_image_features = clip_model.encode_image(coco_images, normalize=True)
                    
                    # Encode COCO captions (frozen)
                    # Wrapper already returns first caption per image, so coco_captions is list of strings
                    coco_text_tokens = tokenizer(coco_captions).to(device)
                    
                    with torch.no_grad():
                        coco_text_features = clip_model.encode_text(coco_text_tokens, normalize=True)
                    
                    # Combine batches: [target_features; coco_features]
                    # Positives: target_img[i] ↔ learned_text[i], coco_img[i] ↔ coco_caption[i]
                    # All cross-pairs are negatives (provides richer negative mining)
                    combined_image_features = torch.cat([image_features, coco_image_features], dim=0)
                    combined_text_features = torch.cat([text_features, coco_text_features], dim=0)
                    
                    # Compute contrastive loss on combined batch
                    # This gives gradients to learnable tokens via target pairs in the larger contrastive matrix
                    coco_loss = clip_loss_fn(combined_image_features, combined_text_features, logit_scale)
                
                # Total loss
                loss = (clip_loss + 
                        args.vocab_loss_weight * vocab_loss + 
                        args.coco_loss_weight * coco_loss)
            
            # Backward pass with optional gradient scaling
            optimizer.zero_grad()
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            # Track
            total_clip_loss += clip_loss.item()
            total_supervised_clip_loss += supervised_clip_loss.item()
            total_unsupervised_clip_loss += unsupervised_clip_loss.item()
            total_vocab_loss += vocab_loss.item()
            total_coco_loss += coco_loss.item()
            num_batches += 1
            
            # Update progress bar - always show both losses
            postfix = {
                'clip_sup': f'{supervised_clip_loss.item():.4f}',
                'clip_uns': f'{unsupervised_clip_loss.item():.4f}',
                'vocab': f'{vocab_loss.item():.4f}'
            }
            if args.use_coco_reference:
                postfix['coco'] = f'{coco_loss.item():.4f}'
            pbar.set_postfix(postfix)
        
        # Epoch summary
        avg_clip = total_clip_loss / num_batches
        avg_supervised_clip = total_supervised_clip_loss / num_batches
        avg_unsupervised_clip = total_unsupervised_clip_loss / num_batches
        avg_vocab = total_vocab_loss / num_batches
        avg_coco = total_coco_loss / num_batches
        
        print(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
        if args.use_supervised:
            print(f"  CLIP Loss (Supervised, optimizing): {avg_supervised_clip:.4f}")
            print(f"  CLIP Loss (Unsupervised, monitoring): {avg_unsupervised_clip:.4f}")
        else:
            print(f"  CLIP Loss (Unsupervised, optimizing): {avg_unsupervised_clip:.4f}")
            print(f"  CLIP Loss (Supervised, monitoring): {avg_supervised_clip:.4f}")
        print(f"  Vocab Loss: {avg_vocab:.4f}")
        if args.use_coco_reference:
            print(f"  COCO Loss: {avg_coco:.4f}")
        
        # Save to history
        epoch_data = {
            'epoch': epoch + 1,
            'clip_loss': avg_clip,  # The one being optimized
            'supervised_clip_loss': avg_supervised_clip,  # Always track both
            'unsupervised_clip_loss': avg_unsupervised_clip,
            'vocab_loss': avg_vocab,
            'coco_loss': avg_coco,
        }
        history.append(epoch_data)
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(recaptioner, args, num_samples, history, epoch=epoch+1, is_final=False)
    
    # Save final checkpoint
    save_checkpoint(recaptioner, args, num_samples, history, is_final=True)
    
    # Print interpretation for discrete mode
    if args.token_mode == "discrete":
        print("\n✓ Discrete mode: Hard token IDs saved for interpretation")
        print("  Use these IDs to look up actual vocabulary tokens")


def save_checkpoint(recaptioner, args, num_samples, history, epoch=None, is_final=False):
    """Save checkpoint with learnable tokens and training state."""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine filename
    if is_final:
        output_file = output_dir / f"{args.dataset}_{args.split}_tokens{args.num_tokens}_{args.token_mode}_final.pt"
        history_file = output_dir / f"{args.dataset}_{args.split}_history_{args.token_mode}.json"
    else:
        output_file = output_dir / f"{args.dataset}_{args.split}_tokens{args.num_tokens}_{args.token_mode}_epoch{epoch}.pt"
        history_file = None  # Only save history at the end
    
    # Build save dict
    save_dict = {
        'dataset_name': args.dataset,
        'split': args.split,
        'num_samples': num_samples,
        'num_learnable_tokens': args.num_tokens,
        'embed_dim': recaptioner.embed_dim,
        'token_mode': args.token_mode,
        'gumbel_temperature': args.gumbel_temperature,
        'history': history,
        'model': args.model,
        'pretrained': args.pretrained,
        'vocab_loss_weight': args.vocab_loss_weight,
        'vocab_k': args.vocab_k,
        'use_supervised': args.use_supervised,
        'use_coco_reference': args.use_coco_reference,
        'epoch': epoch if epoch is not None else len(history),
    }
    
    # Save learnable parameters based on mode
    if args.token_mode == "embedding":
        # Save reparameterized form: directions + scales
        save_dict['learnable_directions'] = recaptioner.learnable_directions.detach().cpu()
        save_dict['learnable_scales'] = recaptioner.learnable_scales.detach().cpu()
        save_dict['avg_token_norm'] = recaptioner.avg_token_norm.detach().cpu()
        # For backward compatibility, also save combined embeddings
        with torch.no_grad():
            directions = F.normalize(recaptioner.learnable_directions, dim=-1)
            embeddings = directions * recaptioner.learnable_scales
        save_dict['learnable_tokens'] = embeddings.detach().cpu()
    else:  # discrete
        save_dict['learnable_logits'] = recaptioner.learnable_logits.detach().cpu()
        # Also save hard token IDs for interpretation
        with torch.no_grad():
            hard_token_ids = recaptioner.learnable_logits.argmax(dim=-1).cpu()
            save_dict['hard_token_ids'] = hard_token_ids
    
    # Save checkpoint
    torch.save(save_dict, output_file)
    file_size_mb = output_file.stat().st_size / (1024**2)
    
    if is_final:
        print(f"\n✓ Saved final tokens to: {output_file}")
        print(f"  File size: {file_size_mb:.2f} MB")
        
        # Save history
        if history_file:
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
            print(f"✓ Saved history to: {history_file}")
    else:
        print(f"  Checkpoint saved: {output_file.name} ({file_size_mb:.2f} MB)")
    
    return output_file


if __name__ == "__main__":
    main()
