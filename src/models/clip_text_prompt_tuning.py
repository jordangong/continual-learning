"""
CLIP Text Prompt Tuning (CoOp) Implementation.

Based on "Learning to Prompt for Vision-Language Models" (CoOp) by Zhou et al.
Adds learnable context tokens to the text encoder instead of using fixed templates.

References:
    - CoOp: https://arxiv.org/abs/2109.01134
    - CoCoOp: https://arxiv.org/abs/2203.05557
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip.model import text_global_pool


class CLIPTextPromptLearner(nn.Module):
    """
    Learnable text prompts for CLIP (CoOp).

    Replaces fixed text templates like "a photo of a {class}" with learnable context vectors:
    - [ctx_1] [ctx_2] ... [ctx_n] [CLASS]
    - [CLASS] [ctx_1] [ctx_2] ... [ctx_n]
    - [ctx_1] ... [ctx_m] [CLASS] [ctx_(m+1)] ... [ctx_n]
    """

    def __init__(
        self,
        num_classes: int,
        class_names: List[str],
        tokenizer: callable,
        text_encoder: nn.Module,
        n_ctx: int = 16,
        ctx_init: Optional[str] = None,
        ctx_position: str = "end",
        class_specific: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize text prompt learner.

        Args:
            num_classes: Number of classes
            class_names: List of class names
            tokenizer: CLIP tokenizer
            text_encoder: CLIP text encoder (needed to get token embedding and dimensions)
            n_ctx: Number of context tokens (default: 16)
            ctx_init: Optional initialization string for context (e.g., "a photo of a")
            ctx_position: Position of class token - "end", "middle", or "front"
            class_specific: Whether to use class-specific context vectors
            device: Device to place tensors on
        """
        super().__init__()

        self.num_classes = num_classes
        self.n_ctx = n_ctx
        self.ctx_position = ctx_position
        self.class_specific = class_specific
        self.device = device if device is not None else torch.device("cpu")

        # Get clip model reference and dimensions
        # text_encoder is CLIPTextEncoderWrapper, access the underlying clip_model
        self.clip_model = text_encoder.clip_model

        # Get token embedding layer and context dimension
        if hasattr(self.clip_model, "token_embedding"):
            self.token_embedding = self.clip_model.token_embedding
            ctx_dim = self.token_embedding.weight.shape[1]
        else:
            raise ValueError("CLIP model must have token_embedding attribute")

        # Initialize context vectors
        if ctx_init:
            # Initialize from a string (e.g., "a photo of a")
            ctx_init = ctx_init.replace("_", " ")
            prompt = tokenizer(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = self.token_embedding(prompt)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]  # Extract context tokens
            if ctx_vectors.shape[0] < n_ctx:
                # Pad with zeros if initialization string is shorter
                padding = torch.zeros(
                    n_ctx - ctx_vectors.shape[0], ctx_dim, device=self.device
                )
                ctx_vectors = torch.cat([ctx_vectors, padding], dim=0)
            prompt_prefix = ctx_init
        else:
            # Random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, device=self.device)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f"Initial context: '{prompt_prefix}'")
        print(f"Number of context tokens: {n_ctx}")

        # Create learnable parameters
        if class_specific:
            print("Using class-specific context vectors")
            # Each class has its own context vectors
            self.ctx = nn.Parameter(
                ctx_vectors.unsqueeze(0).expand(num_classes, -1, -1).clone()
            )
        else:
            print("Using shared context vectors")
            # All classes share the same context vectors
            self.ctx = nn.Parameter(ctx_vectors)

        # Prepare class name tokens
        self._prepare_class_name_tokens(class_names, tokenizer)

    def _prepare_class_name_tokens(self, class_names: List[str], tokenizer: callable):
        """Prepare and tokenize class names (including None for unseen classes)."""
        # Normalize class names, use placeholder for unseen (None) classes
        normalized_names = []
        for name in class_names:
            if name is None:
                normalized_names.append("unknown")  # Placeholder for unseen classes
            else:
                normalized_names.append(name.replace("_", " "))

        # Create prompts with placeholder context based on position
        # The actual context will be replaced with learnable embeddings
        ctx_placeholder = " ".join(["X"] * self.n_ctx)
        if self.ctx_position == "end":
            # [SOS] [CLASS] [CTX] [EOS]: context at the end
            prompts = [f"{name} {ctx_placeholder}." for name in normalized_names]
        elif self.ctx_position == "middle":
            # [SOS] [CTX_first_half] [CLASS] [CTX_second_half] [EOS]
            prompts = [f"{ctx_placeholder} {name}." for name in normalized_names]
        else:  # front
            # [SOS] [CTX] [CLASS] [EOS]: context at the front
            prompts = [f"{ctx_placeholder} {name}." for name in normalized_names]

        # Tokenize
        tokenized_prompts = tokenizer(prompts).to(
            self.device
        )  # [num_classes, max_length]

        with torch.no_grad():
            # Get embeddings for all tokens
            embedding = self.token_embedding(
                tokenized_prompts
            )  # [num_classes, max_length, ctx_dim]

        # Store tokenized prompts and embeddings
        self.register_buffer("tokenized_prompts", tokenized_prompts)

        # Store the number of tokens per class name (needed for all positions)
        self.n_cls = []
        for name in normalized_names:
            name_tokens = tokenizer([name])
            # Count actual tokens (excluding SOS, EOS, padding)
            n_cls_tokens = (name_tokens[0] != 0).sum().item() - 2  # -2 for SOS and EOS
            self.n_cls.append(n_cls_tokens)

        # Extract prefix and suffix based on context position
        # The learnable context will replace the X placeholder tokens
        if self.ctx_position == "end":
            # Prompt structure: [SOS] [CLASS] [CTX_placeholder] [.] [EOS] [PAD]
            # We'll construct: [SOS] [CLASS] [learned_CTX] [.] [EOS] [PAD]
            self.register_buffer("token_prefix", embedding[:, :1, :])  # [SOS]
            # Need to store class+period embeddings separately per class
            # Suffix will be period, EOS, padding (after class and context)
            # We'll handle this dynamically in forward() since class lengths vary
        elif self.ctx_position == "front":
            # Prompt structure: [SOS] [CTX_placeholder] [CLASS] [.] [EOS] [PAD]
            # We'll construct: [SOS] [learned_CTX] [CLASS] [.] [EOS] [PAD]
            self.register_buffer("token_prefix", embedding[:, :1, :])  # [SOS]
            self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx :, :])  # [CLASS] [.] [EOS] [PAD]
        else:  # middle
            # Prompt structure: [SOS] [CTX_placeholder] [CLASS] [.] [EOS] [PAD]
            # We'll construct: [SOS] [CTX_first_half] [CLASS] [CTX_second_half] [.] [EOS] [PAD]
            self.register_buffer("token_prefix", embedding[:, :1, :])  # [SOS]
            # Store full suffix (CTX_placeholder + CLASS + period + EOS + PAD)
            # We'll extract class tokens in forward() based on n_cls
            self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx :, :])  # [CLASS] [.] [EOS] [PAD]
        
        # For "end" position, store the full embedding to extract class tokens dynamically
        if self.ctx_position == "end":
            self.register_buffer("token_embedding_full", embedding)

    def forward(self) -> torch.Tensor:
        """
        Generate prompted text embeddings.

        Returns:
            Prompted embeddings: [num_classes, max_length, ctx_dim]
        """
        ctx = self.ctx

        if not self.class_specific:
            # Shared context: expand to all classes
            ctx = ctx.unsqueeze(0).expand(self.num_classes, -1, -1)

        prefix = self.token_prefix

        if self.ctx_position == "end":
            # Construct: [SOS] [CLASS] [learned_CTX] [.] [EOS] [PAD]
            # Original: [SOS] [CLASS_tokens] [X_tokens] [.] [EOS] [PAD]
            prompts = []
            for i in range(self.num_classes):
                n_cls = self.n_cls[i]
                prefix_i = prefix[i : i + 1, :, :]  # [SOS]
                class_i = self.token_embedding_full[i : i + 1, 1 : 1 + n_cls, :]  # [CLASS]
                ctx_i = ctx[i : i + 1, :, :]  # [learned_CTX]
                # Suffix: period + EOS + padding (after CLASS and CTX positions)
                suffix_i = self.token_embedding_full[i : i + 1, 1 + n_cls + self.n_ctx :, :]  # [.] [EOS] [PAD]
                
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
            
        elif self.ctx_position == "front":
            # Construct: [SOS] [learned_CTX] [CLASS] [.] [EOS] [PAD]
            suffix = self.token_suffix
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
            
        elif self.ctx_position == "middle":
            # Construct: [SOS] [CTX_first_half] [CLASS] [CTX_second_half] [.] [EOS] [PAD]
            half_n_ctx = self.n_ctx // 2
            suffix = self.token_suffix
            prompts = []
            for i in range(self.num_classes):
                n_cls = self.n_cls[i]
                prefix_i = prefix[i : i + 1, :, :]  # [SOS]
                ctx_i = ctx[i : i + 1, :, :]  # [learned_CTX]
                class_i = suffix[i : i + 1, :n_cls, :]  # [CLASS]
                suffix_i = suffix[i : i + 1, n_cls:, :]  # [.] [EOS] [PAD]

                prompt = torch.cat(
                    [
                        prefix_i,  # [SOS]
                        ctx_i[:, :half_n_ctx, :],  # First half of context
                        class_i,  # [CLASS]
                        ctx_i[:, half_n_ctx:, :],  # Second half of context
                        suffix_i,  # [.] [EOS] [PAD]
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError(f"Unknown ctx_position: {self.ctx_position}")

        return prompts


class CLIPTextPromptedClassifier(nn.Module):
    """
    CLIP classifier with learnable text prompts (CoOp).

    Wraps CLIPClassifier and replaces the text encoding process with learnable prompts.
    """

    def __init__(
        self,
        base_classifier,
        n_ctx: int = 16,
        ctx_init: Optional[str] = None,
        ctx_position: str = "end",
        class_specific: bool = False,
    ):
        """
        Initialize prompted CLIP classifier.

        Args:
            base_classifier: Base CLIPClassifier instance
            n_ctx: Number of context tokens
            ctx_init: Optional initialization string
            ctx_position: Position of class token - "end", "middle", or "front"
            class_specific: Whether to use class-specific context
        """
        super().__init__()

        # Store base classifier (but we'll replace its text encoding)
        self.base_classifier = base_classifier

        # Configuration
        self.n_ctx = n_ctx
        self.ctx_init = ctx_init
        self.ctx_position = ctx_position
        self.class_specific = class_specific

        # Freeze text encoder - we only train the prompts
        for param in self.base_classifier.text_encoder.parameters():
            param.requires_grad = False

        print("Text encoder frozen - only prompts are trainable")

        # Prompt learner will be initialized when set_class_names() is called
        self.prompt_learner = None
        
        # Track cumulative class names for continual learning
        self._cumulative_class_names = []

    def _initialize_prompt_learner(self, class_names: List[str]) -> None:
        """Create and initialize the prompt learner with all dataset classes."""
        # Use total number of classes in dataset, not just classes seen so far
        # The trainer will handle masking logits for unseen classes
        self.prompt_learner = CLIPTextPromptLearner(
            num_classes=self.base_classifier.num_classes,  # Total classes in dataset
            class_names=class_names,
            tokenizer=self.base_classifier.tokenizer,
            text_encoder=self.base_classifier.text_encoder,
            n_ctx=self.n_ctx,
            ctx_init=self.ctx_init,
            ctx_position=self.ctx_position,
            class_specific=self.class_specific,
            device=self.base_classifier.device,
        )

    def set_class_names(
        self, class_names: List[str], class_indices: Optional[List[int]] = None
    ) -> None:
        """Set class names and update prompt learner for new classes."""
        # Forward to base classifier to maintain its state
        self.base_classifier.set_class_names(class_names, class_indices)
        
        # Update cumulative class names tracking
        if class_indices is None:
            class_indices = list(range(len(class_names)))
        
        # Expand cumulative list to match total classes
        if len(self._cumulative_class_names) < self.base_classifier.num_classes:
            self._cumulative_class_names.extend([None] * (self.base_classifier.num_classes - len(self._cumulative_class_names)))
        
        # Add new class names
        for i, class_idx in enumerate(class_indices):
            self._cumulative_class_names[class_idx] = class_names[i]
        
        # Initialize prompt learner once with full class list (includes None for unseen classes)
        if self.prompt_learner is None:
            # First time: create prompt learner for all classes
            # Pass the full list including None placeholders for unseen classes
            self._initialize_prompt_learner(self._cumulative_class_names)
        else:
            # Update prompt learner with new class names
            # Only need to update the class name tokens for newly added classes
            self.prompt_learner._prepare_class_name_tokens(
                self._cumulative_class_names, 
                self.base_classifier.tokenizer
            )

        # Note: We don't compute cached embeddings here because:
        # - During training: embeddings are recomputed dynamically in forward()
        # - During inference: embeddings are cached when switching to eval mode via train(False)

    def _update_text_embeddings_with_prompts(self):
        """Compute text embeddings using learned prompts."""
        if self.prompt_learner is None:
            return

        with torch.no_grad():
            # Get prompted embeddings
            prompts = self.prompt_learner()  # [num_classes, max_length, ctx_dim]

            # Pass through text encoder's transformer
            # We need to bypass token_embedding and go straight to transformer
            clip_model = self.base_classifier.text_encoder.clip_model
            tokenized_prompts = self.prompt_learner.tokenized_prompts

            # Follow OpenCLIP encode_text pipeline exactly:
            # 1. Cast to transformer dtype
            cast_dtype = clip_model.transformer.get_cast_dtype()
            x = prompts.to(cast_dtype)

            # 2. Add positional embedding
            x = x + clip_model.positional_embedding.to(cast_dtype)

            # 3. Pass through transformer with attention mask
            x = clip_model.transformer(x, attn_mask=clip_model.attn_mask)

            # 4. Apply final layer norm
            x = clip_model.ln_final(x)  # [num_classes, max_length, ctx_dim]

            # 5. Global pooling (extract features from EOT token)
            # Use text_global_pool from open_clip to handle different pooling strategies
            text_features = text_global_pool(
                x,
                tokenized_prompts,
                clip_model.text_pool_type
                if hasattr(clip_model, "text_pool_type")
                else "argmax",
                eos_token_id=getattr(clip_model, "text_eos_id", None),
            )

            # 6. Apply text projection if exists
            if (
                hasattr(clip_model, "text_projection")
                and clip_model.text_projection is not None
            ):
                if isinstance(clip_model.text_projection, nn.Linear):
                    text_features = clip_model.text_projection(text_features)
                else:
                    text_features = text_features @ clip_model.text_projection

            # Apply external text projection if available (for projection tuning)
            if self.base_classifier.text_projection is not None:
                text_features = self.base_classifier.text_projection(text_features)

            # Normalize
            if self.base_classifier.normalize:
                text_features = F.normalize(text_features, p=2, dim=-1)

            # Update cached embeddings
            self.base_classifier.text_embeddings = text_features
            self.base_classifier.text_embeddings_initialized = True

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass with prompted text embeddings.

        Args:
            x: Image features [batch_size, feature_dim]
            **kwargs: Additional arguments for base classifier

        Returns:
            Logits [batch_size, num_classes]
        """
        # During training, recompute text embeddings with current prompts
        if self.training and self.prompt_learner is not None:
            # Get prompted embeddings
            prompts = self.prompt_learner()  # [num_classes, max_length, ctx_dim]

            # Pass through text encoder's transformer
            clip_model = self.base_classifier.text_encoder.clip_model
            tokenized_prompts = self.prompt_learner.tokenized_prompts

            # Follow OpenCLIP encode_text pipeline exactly:
            # 1. Cast to transformer dtype
            cast_dtype = clip_model.transformer.get_cast_dtype()
            x_text = prompts.to(cast_dtype)

            # 2. Add positional embedding
            x_text = x_text + clip_model.positional_embedding.to(cast_dtype)

            # 3. Pass through transformer with attention mask
            x_text = clip_model.transformer(x_text, attn_mask=clip_model.attn_mask)

            # 4. Apply final layer norm
            x_text = clip_model.ln_final(x_text)

            # 5. Global pooling (extract features from EOT token)
            text_features = text_global_pool(
                x_text,
                tokenized_prompts,
                clip_model.text_pool_type
                if hasattr(clip_model, "text_pool_type")
                else "argmax",
                eos_token_id=getattr(clip_model, "text_eos_id", None),
            )

            # 6. Apply text projection if exists
            if (
                hasattr(clip_model, "text_projection")
                and clip_model.text_projection is not None
            ):
                if isinstance(clip_model.text_projection, nn.Linear):
                    text_features = clip_model.text_projection(text_features)
                else:
                    text_features = text_features @ clip_model.text_projection

            # Apply external text projection if available
            if self.base_classifier.text_projection is not None:
                text_features = self.base_classifier.text_projection(text_features)

            # Normalize
            if self.base_classifier.normalize:
                text_features = F.normalize(text_features, p=2, dim=-1)

            # Temporarily replace text embeddings
            original_embeddings = self.base_classifier.text_embeddings
            self.base_classifier.text_embeddings = text_features

            # Forward through base classifier
            output = self.base_classifier(x, **kwargs)

            # Restore original embeddings
            self.base_classifier.text_embeddings = original_embeddings

            return output
        else:
            # During inference, use cached embeddings
            return self.base_classifier(x, **kwargs)

    def train(self, mode: bool = True):
        """Override train() to update cached embeddings when switching to eval mode."""
        super().train(mode)

        # When switching to eval mode (mode=False), update cached embeddings with learned prompts
        if not mode and self.prompt_learner is not None:
            self._update_text_embeddings_with_prompts()

        return self

    def __getattr__(self, name: str):
        """Forward attribute access to base classifier for compatibility."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_classifier, name)


def create_clip_text_prompted_model(
    base_model,
    coop_config: Dict[str, Any],
) -> nn.Module:
    """
    Create a CLIP model with text prompt tuning (CoOp).

    Args:
        base_model: Base model with CLIPClassifier
        coop_config: CoOp configuration dictionary

    Returns:
        Model with prompted CLIP classifier
    """
    # Check if model has CLIP classifier
    if not hasattr(base_model, "classifier"):
        raise ValueError("Base model must have a classifier attribute")

    if not hasattr(base_model.classifier, "text_encoder"):
        raise ValueError("Classifier must be a CLIPClassifier with text_encoder")

    # Wrap the classifier with prompted version
    base_model.classifier = CLIPTextPromptedClassifier(
        base_classifier=base_model.classifier,
        n_ctx=coop_config.get("n_ctx", 16),
        ctx_init=coop_config.get("ctx_init", None),
        ctx_position=coop_config.get("ctx_position", "end"),
        class_specific=coop_config.get("class_specific", False),
    )

    print("Created CLIP text prompted model (CoOp)")
    print(f"  - Context length: {coop_config.get('n_ctx', 16)}")
    print(f"  - Context position: {coop_config.get('ctx_position', 'end')}")
    print(f"  - Class-specific: {coop_config.get('class_specific', False)}")

    return base_model
