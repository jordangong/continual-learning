"""
Vision Transformer specific prompt tuning implementation.
Based on Learning to Prompt (L2P) paper with ViT integration.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models._manipulate import checkpoint_seq


class PromptPool(nn.Module):
    """
    Prompt pool implementation based on Learning to Prompt (L2P).

    This module maintains a pool of prompts and selects relevant prompts
    based on input features using a key-based selection mechanism.
    """

    def __init__(
        self,
        pool_size: int,
        prompt_length: int,
        embed_dim: int,
        init_type: str = "random",
        key_diversity_regularization: bool = False,
        frequency_diversity_regularization: bool = False,
        top_k: int = 5,
        batchwise_prompt: bool = False,
        prompt_key_init: str = "uniform",
        init_config: dict = None,
    ):
        """
        Initialize the prompt pool.

        Args:
            pool_size: Number of prompts in the pool
            prompt_length: Length of each prompt
            embed_dim: Embedding dimension
            init_type: Initialization type for prompts
            key_diversity_regularization: Whether to apply diversity regularization
            frequency_diversity_regularization: Whether to apply L2P-style frequency-based diversity regularization
            top_k: Number of top prompts to select
            batchwise_prompt: Whether to select prompts per batch or per sample
            prompt_key_init: Initialization type for prompt keys
            init_config: Configuration for initialization parameters
        """
        super().__init__()

        self.pool_size = pool_size
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim
        self.init_type = init_type
        self.key_diversity_regularization = key_diversity_regularization
        self.frequency_diversity_regularization = frequency_diversity_regularization
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt

        # Initialize configuration with defaults
        self.init_config = init_config or {}

        # Create prompt pool and keys
        self.prompt_pool = nn.Parameter(
            torch.empty(pool_size, prompt_length, embed_dim)
        )
        self.prompt_key = nn.Parameter(torch.empty(pool_size, embed_dim))

        # Initialize frequency tracking for L2P-style diversity regularization
        if self.frequency_diversity_regularization:
            # Register buffer so it's moved to device but not trained
            self.register_buffer("prompt_frequency", torch.zeros(pool_size, dtype=torch.int))

        # Initialize prompts and keys
        self._init_prompts(init_type)
        self._init_keys(prompt_key_init)

    def _init_prompts(self, init_type: str):
        """Initialize prompt embeddings."""
        if init_type == "random":
            std = self.init_config.get("normal_std", 0.02)
            nn.init.normal_(self.prompt_pool, std=std)
        elif init_type == "uniform":
            uniform_range = self.init_config.get("uniform_range", [-0.1, 0.1])
            nn.init.uniform_(self.prompt_pool, uniform_range[0], uniform_range[1])
        elif init_type == "xavier":
            nn.init.xavier_uniform_(self.prompt_pool)
        else:
            std = self.init_config.get("normal_std", 0.02)
            nn.init.normal_(self.prompt_pool, std=std)

    def _init_keys(self, init_type: str):
        """Initialize prompt keys."""
        if init_type == "normal":
            std = self.init_config.get("normal_std", 0.02)
            nn.init.normal_(self.prompt_key, std=std)
        elif init_type == "uniform":
            uniform_range = self.init_config.get("uniform_range", [-0.1, 0.1])
            nn.init.uniform_(self.prompt_key, uniform_range[0], uniform_range[1])
        elif init_type == "xavier":
            nn.init.xavier_uniform_(self.prompt_key)
        else:
            std = self.init_config.get("normal_std", 0.02)
            nn.init.normal_(self.prompt_key, std=std)

    def forward(
        self, query: torch.Tensor, train: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass to select and return prompts.

        Args:
            query: Query tensor for prompt selection [batch_size, embed_dim]
            train: Whether in training mode

        Returns:
            Tuple of (selected_prompts, similarity, top_k_similarity)
        """
        batch_size = query.size(0)

        # Compute similarity between query and prompt keys
        # query: [batch_size, embed_dim]
        # prompt_key: [pool_size, embed_dim]
        query_norm = F.normalize(query, p=2, dim=1)
        key_norm = F.normalize(self.prompt_key, p=2, dim=1)

        # Compute cosine similarity
        similarity = torch.matmul(query_norm, key_norm.t())  # [batch_size, pool_size]
        distance = 1 - similarity

        # Apply L2P-style frequency-based diversity regularization during training
        if train and self.frequency_diversity_regularization:
            # Normalize frequency to get penalty weights
            total_frequency = self.prompt_frequency.sum()
            if total_frequency == 0:
                normalized_frequency = torch.ones_like(self.prompt_frequency) / self.pool_size
            else:
                normalized_frequency = self.prompt_frequency / total_frequency
            # L2P paper: argmin γ(q(x), k_si) · h_si
            # To penalize frequent prompts, we multiply distance by frequency
            frequency_penalty = normalized_frequency.unsqueeze(0).expand(batch_size, -1)
            distance *= frequency_penalty

        if self.batchwise_prompt:
            # Select top-k prompts based on average distance across batch
            avg_distance = distance.mean(dim=0)  # [pool_size]
            _, top_k_indices = torch.topk(
                avg_distance, self.top_k, dim=0, largest=False
            )

            # Expand indices for all samples in batch
            top_k_indices = top_k_indices.unsqueeze(0).expand(batch_size, -1)
        else:
            # Select top-k prompts for each sample
            _, top_k_indices = torch.topk(distance, self.top_k, dim=1, largest=False)
        top_k_similarity = similarity.gather(1, top_k_indices)

        # Update frequency tracking during training
        if train and self.frequency_diversity_regularization:
            # Count how many times each prompt is selected
            counts = torch.bincount(top_k_indices.flatten(), minlength=self.pool_size)
            # Update frequency counter
            self.prompt_frequency += counts

        # Get selected prompts
        # top_k_indices: [batch_size, top_k]
        # prompt_pool: [pool_size, prompt_length, embed_dim]
        selected_prompts = self.prompt_pool[top_k_indices]
        # [batch_size, top_k, prompt_length, embed_dim]

        # Reshape to [batch_size, top_k * prompt_length, embed_dim]
        selected_prompts = selected_prompts.view(
            batch_size, self.top_k * self.prompt_length, self.embed_dim
        )

        return selected_prompts, similarity, top_k_similarity

    def get_diversity_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        """
        Compute diversity regularization loss to encourage diverse prompt selection.

        Args:
            similarity: Similarity scores [batch_size, pool_size]

        Returns:
            Diversity loss
        """
        if not self.key_diversity_regularization:
            return torch.tensor(0.0, device=similarity.device)

        # Compute pairwise cosine similarity between prompt keys
        key_norm = F.normalize(self.prompt_key, p=2, dim=1)
        key_similarity = torch.matmul(key_norm, key_norm.t())

        # Remove diagonal (self-similarity)
        mask = torch.eye(self.pool_size, device=key_similarity.device).bool()
        key_similarity = key_similarity.masked_fill(mask, 0)

        # Diversity loss encourages orthogonality between keys
        diversity_loss = key_similarity.abs().mean()

        return diversity_loss

    def reset_frequency_tracking(self):
        """
        Reset frequency tracking for L2P-style diversity regularization.
        This should be called at the beginning of each new task.
        """
        if self.frequency_diversity_regularization:
            self.prompt_frequency.zero_()
            self.frequency_update_count = 0


def global_pool_nlc(
    x: torch.Tensor,
    pool_type: str = "token",
    num_prefix_tokens: int = 1,
    num_prompt_tokens: int = 0,
    reduce_include_prefix: bool = False,
):
    if not pool_type:
        return x

    if pool_type == "token":
        x = x[:, 0]  # class token
    elif pool_type == "prompt_avg":
        x = x[:, :num_prompt_tokens].mean(dim=1)
    else:
        x = x if reduce_include_prefix else x[:, num_prefix_tokens:]
        if pool_type == "avg":
            x = x.mean(dim=1)
        elif pool_type == "avgmax":
            x = 0.5 * (x.amax(dim=1) + x.mean(dim=1))
        elif pool_type == "max":
            x = x.amax(dim=1)
        else:
            assert not pool_type, f"Unknown pool type {pool_type}"

    return x


class ViTPromptedModel(nn.Module):
    """
    Vision Transformer with prompt tuning support.
    Integrates prompts directly into the ViT architecture.
    """

    def __init__(
        self,
        base_model: nn.Module,
        prompt_config: Dict[str, Any],
        embed_dim: int,
        num_classes: int,
    ):
        """
        Initialize ViT with prompt tuning.

        Args:
            base_model: Base ViT model
            prompt_config: Prompt configuration
            embed_dim: Embedding dimension
            num_classes: Number of output classes
        """
        super().__init__()

        self.base_model = base_model
        self.prompt_config = prompt_config
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Prompt configuration
        self.prompt_length = prompt_config.get("prompt_length", 10)
        self.use_prompt_pool = prompt_config.get("prompt_pool", False)
        self.omit_cls_token = prompt_config.get("omit_cls_token", False)

        # Initialize prompts
        if self.use_prompt_pool:
            self._init_prompt_pool()
        else:
            self._init_simple_prompts()

        # Prompt dropout
        self.prompt_dropout = nn.Dropout(prompt_config.get("prompt_dropout", 0.0))

    def _init_prompt_pool(self):
        """Initialize prompt pool."""
        top_k = min(
            self.prompt_config.get("top_k", 5),
            self.prompt_config.get("pool_size", 10),
        )
        self.prompt_pool = PromptPool(
            pool_size=self.prompt_config.get("pool_size", 10),
            prompt_length=self.prompt_length,
            embed_dim=self.embed_dim,
            init_type=self.prompt_config.get("init_type", "random"),
            key_diversity_regularization=self.prompt_config.get(
                "key_diversity_regularization", False
            ),
            frequency_diversity_regularization=self.prompt_config.get(
                "frequency_diversity_regularization", False
            ),
            top_k=top_k,
            init_config=self.prompt_config.get("init_config", None),
        )

        self.num_prompt_tokens = self.prompt_length * top_k

    def _init_simple_prompts(self):
        """initialize simple prompt embeddings."""
        self.prompt_embeddings = nn.Parameter(
            torch.empty(self.prompt_length, self.embed_dim)
        )
        init_type = self.prompt_config.get("init_type", "random")
        init_config = self.prompt_config.get("init_config", {})

        if init_type == "random":
            std = init_config.get("normal_std", 0.02)
            nn.init.normal_(self.prompt_embeddings, std=std)
        elif init_type == "uniform":
            uniform_range = init_config.get("uniform_range", [-0.1, 0.1])
            nn.init.uniform_(self.prompt_embeddings, uniform_range[0], uniform_range[1])
        elif init_type == "xavier":
            nn.init.xavier_uniform_(self.prompt_embeddings)
        else:
            std = init_config.get("normal_std", 0.02)
            nn.init.normal_(self.prompt_embeddings, std=std)

        self.num_prompt_tokens = self.prompt_length

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        forward pass through vit with prompts integrated.

        args:
            x: input tensor [batch_size, channels, height, width]

        returns:
            output logits [batch_size, num_classes]
        """
        x = self.forward_features(x)
        x = self.forward_head(x)
        x = self.base_model.classifier(x)

        return x

    def get_auxiliary_losses(self) -> Dict[str, torch.Tensor]:
        """
        Get auxiliary losses for prompt tuning (e.g., diversity loss, similarity loss).

        Returns:
            Dictionary of auxiliary losses
        """
        losses = {}

        if self.use_prompt_pool:
            if hasattr(self, "_last_diversity_loss"):
                losses["diversity_loss"] = self._last_diversity_loss

            if hasattr(self, "_last_similarity"):
                # Similarity loss encourages the model to use prompts that are similar to the query
                # This is computed as the negative mean of the top-k similarities
                similarity_loss = -self._last_similarity.mean()
                losses["similarity_loss"] = similarity_loss

        return losses

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_prompt_pool:
            query = self.base_model.forward_features(x)
            prompt_embeddings, similarity, top_k_similarity = self.prompt_pool(
                query, train=self.training
            )
            self._last_similarity = top_k_similarity
            self._last_diversity_loss = self.prompt_pool.get_diversity_loss(similarity)
        else:
            prompt_embeddings = self.prompt_embeddings.unsqueeze(0).expand(
                x.size(0), -1, -1
            )
        prompt_embeddings = self.prompt_dropout(prompt_embeddings)

        x = self.base_model.backbone.patch_embed(x)
        x = self.base_model.backbone._pos_embed(x)
        x = self.base_model.backbone.patch_drop(x)

        prefix, x = (
            x[:, : self.base_model.backbone.num_prefix_tokens],
            x[:, self.base_model.backbone.num_prefix_tokens :],
        )

        if self.omit_cls_token:
            prefix = prefix[:, 1:]

        x = torch.cat([prefix, prompt_embeddings, x], dim=1)

        x = self.base_model.backbone.norm_pre(x)
        if self.base_model.backbone.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.base_model.backbone.blocks, x)
        else:
            x = self.base_model.backbone.blocks(x)
        x = self.base_model.backbone.norm(x)

        return x

    def pool(self, x: torch.Tensor, pool_type: Optional[str] = None) -> torch.Tensor:
        if self.base_model.backbone.attn_pool is not None:
            x = self.base_model.backbone.attn_pool(x)
            return x
        pool_type = (
            self.base_model.backbone.global_pool if pool_type is None else pool_type
        )
        x = global_pool_nlc(
            x,
            pool_type=pool_type,
            num_prefix_tokens=self.base_model.backbone.num_prefix_tokens,
            num_prompt_tokens=self.num_prompt_tokens,
        )
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.pool(x, pool_type="prompt_avg" if self.omit_cls_token else None)
        x = self.base_model.backbone.fc_norm(x)
        x = self.base_model.backbone.head_drop(x)
        return x if pre_logits else self.base_model.backbone.head(x)

    def reset_frequency_tracking(self):
        """
        Reset frequency tracking for L2P-style diversity regularization.
        This should be called at the beginning of each new task.
        """
        if hasattr(self, "prompt_pool") and self.prompt_pool is not None:
            self.prompt_pool.reset_frequency_tracking()

    # Forward common methods to base model
    def __getattr__(self, name: str):
        """Forward attribute access to base model if not found in ViTPromptedModel."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)


def create_vit_prompted_model(
    base_model: nn.Module,
    prompt_config: Dict,
    embed_dim: int,
    num_classes: int,
) -> nn.Module:
    """
    Create a ViT model with prompt tuning.

    Args:
        base_model: Base ViT model
        prompt_config: Prompt configuration
        embed_dim: Embedding dimension
        num_classes: Number of output classes

    Returns:
        ViT model with prompt tuning
    """
    return ViTPromptedModel(
        base_model=base_model,
        prompt_config=prompt_config,
        embed_dim=embed_dim,
        num_classes=num_classes,
    )
