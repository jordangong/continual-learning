"""
Vision Transformer specific prompt tuning implementation.
Based on Learning to Prompt (L2P) paper with ViT integration.
"""

from typing import Any, Dict, Tuple

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
        top_k: int = 5,
        batchwise_prompt: bool = True,
        prompt_key_init: str = "uniform",
    ):
        """
        Initialize the prompt pool.

        Args:
            pool_size: Number of prompts in the pool
            prompt_length: Length of each prompt
            embed_dim: Embedding dimension
            init_type: Initialization type for prompts
            key_diversity_regularization: Whether to apply diversity regularization
            top_k: Number of top prompts to select
            batchwise_prompt: Whether to select prompts per batch or per sample
            prompt_key_init: Initialization type for prompt keys
        """
        super().__init__()

        self.pool_size = pool_size
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.key_diversity_regularization = key_diversity_regularization

        # Initialize prompt pool
        self.prompt_pool = nn.Parameter(
            torch.empty(pool_size, prompt_length, embed_dim)
        )

        # Initialize prompt keys for selection
        self.prompt_key = nn.Parameter(torch.empty(pool_size, embed_dim))

        # Initialize parameters
        self._init_prompts(init_type)
        self._init_keys(prompt_key_init)

    def _init_prompts(self, init_type: str):
        """Initialize prompt embeddings."""
        if init_type == "random":
            nn.init.normal_(self.prompt_pool, std=0.02)
        elif init_type == "uniform":
            nn.init.uniform_(self.prompt_pool, -0.1, 0.1)
        elif init_type == "xavier":
            nn.init.xavier_uniform_(self.prompt_pool)
        else:
            nn.init.normal_(self.prompt_pool, std=0.02)

    def _init_keys(self, init_type: str):
        """Initialize prompt keys."""
        if init_type == "uniform":
            nn.init.uniform_(self.prompt_key, -1, 1)
        elif init_type == "normal":
            nn.init.normal_(self.prompt_key, std=0.02)
        else:
            nn.init.uniform_(self.prompt_key, -1, 1)

    def forward(
        self, query: torch.Tensor, train: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass to select and return prompts.

        Args:
            query: Query tensor for prompt selection [batch_size, embed_dim]
            train: Whether in training mode

        Returns:
            Tuple of (selected_prompts, prompt_mask, similarity_scores)
        """
        batch_size = query.size(0)

        # Compute similarity between query and prompt keys
        # query: [batch_size, embed_dim]
        # prompt_key: [pool_size, embed_dim]
        query_norm = F.normalize(query, p=2, dim=1)
        key_norm = F.normalize(self.prompt_key, p=2, dim=1)

        # Compute cosine similarity
        similarity = torch.matmul(query_norm, key_norm.t())  # [batch_size, pool_size]

        if self.batchwise_prompt:
            # Select top-k prompts based on average similarity across batch
            avg_similarity = similarity.mean(dim=0)  # [pool_size]
            _, top_k_indices = torch.topk(avg_similarity, self.top_k, dim=0)

            # Expand indices for all samples in batch
            top_k_indices = top_k_indices.unsqueeze(0).expand(batch_size, -1)
        else:
            # Select top-k prompts for each sample
            _, top_k_indices = torch.topk(similarity, self.top_k, dim=1)

        # Get selected prompts
        # top_k_indices: [batch_size, top_k]
        # prompt_pool: [pool_size, prompt_length, embed_dim]
        selected_prompts = self.prompt_pool[
            top_k_indices
        ]  # [batch_size, top_k, prompt_length, embed_dim]

        # Reshape to [batch_size, top_k * prompt_length, embed_dim]
        selected_prompts = selected_prompts.view(
            batch_size, self.top_k * self.prompt_length, self.embed_dim
        )

        # Create prompt mask (all ones since we're using all selected prompts)
        prompt_mask = torch.ones(
            batch_size,
            self.top_k * self.prompt_length,
            dtype=torch.bool,
            device=query.device,
        )

        return selected_prompts, prompt_mask, similarity

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

        # Initialize prompts
        if self.use_prompt_pool:
            self._init_prompt_pool()
        else:
            self._init_simple_prompts()

        # Prompt dropout
        self.prompt_dropout = nn.Dropout(prompt_config.get("prompt_dropout", 0.0))

    def _init_prompt_pool(self):
        """Initialize prompt pool."""
        self.prompt_pool = PromptPool(
            pool_size=self.prompt_config.get("pool_size", 10),
            prompt_length=self.prompt_length,
            embed_dim=self.embed_dim,
            init_type=self.prompt_config.get("init_type", "random"),
            key_diversity_regularization=self.prompt_config.get(
                "key_diversity_regularization", False
            ),
            top_k=min(5, self.prompt_config.get("pool_size", 10)),
        )

    def _init_simple_prompts(self):
        """Initialize simple prompt embeddings."""
        self.prompt_embeddings = nn.Parameter(
            torch.empty(self.prompt_length, self.embed_dim)
        )
        init_type = self.prompt_config.get("init_type", "random")
        if init_type == "random":
            nn.init.normal_(self.prompt_embeddings, std=0.02)
        elif init_type == "uniform":
            nn.init.uniform_(self.prompt_embeddings, -0.1, 0.1)
        elif init_type == "xavier":
            nn.init.xavier_uniform_(self.prompt_embeddings)
        else:
            nn.init.normal_(self.prompt_embeddings, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ViT with prompts integrated.

        Args:
            x: Input tensor [batch_size, channels, height, width]

        Returns:
            Output logits [batch_size, num_classes]
        """
        x = self.forward_features(x)
        x = self.base_model.backbone.forward_head(x)
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
            prompt_embeddings, _, similarity = self.prompt_pool(
                query, train=self.training
            )
            self._last_similarity = similarity
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
        x = torch.cat([prefix, prompt_embeddings, x], dim=1)

        x = self.base_model.backbone.norm_pre(x)
        if self.base_model.backbone.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.base_model.backbone.blocks, x)
        else:
            x = self.base_model.backbone.blocks(x)
        x = self.base_model.backbone.norm(x)

        return x


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
