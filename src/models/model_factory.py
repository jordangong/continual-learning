import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import open_clip
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Import for SAE
try:
    import saev.nn
except ImportError:
    print("Warning: saev module not found. SAE functionality will not be available.")

from .vit_prompt_tuning import create_vit_prompted_model
from .lora import create_lora_model
from .adapter import create_adapter_model
from .projection import create_projection_model


class CLIPTextEncoderWrapper(nn.Module):
    """Wrapper for CLIP text encoder that provides a unified interface.

    This wraps the full encode_text pipeline (token_embedding, positional_embedding,
    transformer, ln_final, text_projection) so it can be treated as a single module
    for freezing/unfreezing and forward passes.

    Note: This wrapper only saves/loads text encoder parameters to avoid duplicating
    the visual encoder in checkpoints.
    """

    def __init__(self, clip_model: nn.Module):
        """
        Args:
            clip_model: Full CLIP model with encode_text method
        """
        super().__init__()
        clip_model = copy.deepcopy(clip_model)
        if hasattr(clip_model, "visual"):
            del clip_model.visual
        if hasattr(clip_model, "logit_scale"):
            del clip_model.logit_scale
        if hasattr(clip_model, "logit_bias"):
            del clip_model.logit_bias
        self.clip_model = clip_model

    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """Encode text tokens to embeddings.

        Args:
            text_tokens: Tokenized text [batch_size, sequence_length]
        Returns:
            Text embeddings [batch_size, feature_dim]
        """
        return self.clip_model.encode_text(text_tokens, normalize=False)


def _create_learned_classifier(
    in_features: int,
    num_classes: int,
    classifier_type: str = "linear",
    hidden_dim: Optional[int] = None,
    dropout: float = 0.0,
    normalize: bool = False,
) -> nn.Module:
    """Create a learned classifier (linear, MLP, or prototypical).

    Args:
        in_features: Input feature dimension
        num_classes: Number of output classes
        classifier_type: "linear", "mlp", or "prototypical"
        hidden_dim: Hidden dimension for MLP (required if classifier_type="mlp")
        dropout: Dropout probability for MLP
        normalize: Whether to normalize features (for prototypical classifier)

    Returns:
        nn.Module: Linear, MLP, or Prototypical classifier
    """
    if classifier_type == "linear":
        return nn.Linear(in_features, num_classes)
    elif classifier_type == "mlp":
        if hidden_dim is None:
            raise ValueError("hidden_dim must be provided for MLP classifier")
        return nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    elif classifier_type == "prototypical":
        return PrototypicalClassifier(in_features, num_classes, normalize=normalize)
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")


class CLIPClassifier(nn.Module):
    """CLIP-based classifier using text encoder for zero-shot classification."""

    def __init__(
        self,
        text_encoder: nn.Module,
        tokenizer: callable,
        num_classes: int,
        feature_dim: int,
        class_names: Optional[List[str]] = None,
        text_templates: Optional[List[str]] = None,
        mode: str = "text",
        temperature: float = 1.0,
        normalize: bool = False,
        hybrid_weight: float = 0.5,
        ensemble_text: bool = False,
        freeze_text_encoder: bool = False,
        learned_classifier_type: str = "linear",
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        learnable_temperature: bool = False,
        use_log_temperature: bool = False,
        pretraining_temperature: float = 1.0,
        learnable_pretraining_temperature: bool = False,
        use_log_pretraining_temperature: bool = False,
        logit_bias: Optional[float] = None,
        learnable_logit_bias: bool = False,
        learnable_hybrid_weight: bool = False,
        normalize_class_names: bool = False,
        disable_learned_classifier_at_inference: bool = False,
        device: Optional[torch.device] = None,
        text_projection: Optional[nn.Module] = None,
        vision_projection: Optional[nn.Module] = None,
    ):
        """
        Args:
            text_encoder: CLIP text encoder module
            tokenizer: CLIP tokenizer function
            num_classes: Number of output classes
            feature_dim: Dimension of image features
            class_names: List of class names for text embedding generation
            text_templates: List of prompt templates (e.g., "a photo of a {}")
            mode: Classification mode:
                - "text": Use text encoder only (frozen or trainable based on freeze_text_encoder)
                - "hybrid": Combine text + learned classifier (both can be trainable)
            temperature: Temperature scaling for logits (CLIP-style: multiply logits, typically exp(logit_scale) ≈ 100)
            normalize: Whether to normalize features
            hybrid_weight: Weight for text vs learned in hybrid mode (0=learned, 1=text)
            ensemble_text: Average over multiple text templates
            freeze_text_encoder: Whether to freeze text encoder (True for zero-shot, False for training)
            learned_classifier_type: Type of learned classifier ("linear" or "mlp")
            hidden_dim: Hidden dimension for MLP classifier (only used if learned_classifier_type="mlp")
            dropout: Dropout probability for MLP classifier
            learnable_temperature: Whether temperature should be learnable
            use_log_temperature: If learnable_temperature=True, parameterize in log space
            logit_bias: Logit bias value (extracted from CLIP model, typically for CustomTextCLIP)
            learnable_logit_bias: Whether logit_bias should be learnable
            learnable_hybrid_weight: Whether hybrid_weight should be learnable (only used in hybrid mode)
            normalize_class_names: Replace underscores with spaces in class names before creating prompts
            disable_learned_classifier_at_inference: Disable learned classifier during inference (eval mode), use text-only
            device: Device to place text embeddings on
        """
        super().__init__()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.mode = mode
        self.normalize = normalize
        self.ensemble_text = ensemble_text
        self.normalize_class_names = normalize_class_names
        self.text_projection = text_projection  # Optional projection for text features
        self.vision_projection = vision_projection  # Optional projection for vision features
        
        # Hybrid weight handling (for hybrid mode)
        if learnable_hybrid_weight:
            self.hybrid_weight = nn.Parameter(torch.tensor(hybrid_weight))
        else:
            self.hybrid_weight = hybrid_weight
        self.device = device if device is not None else torch.device("cpu")
        self.learned_classifier_type = learned_classifier_type
        self.disable_learned_classifier_at_inference = disable_learned_classifier_at_inference

        # Temperature handling (similar to ClassifierHead)
        self.use_log_temperature = use_log_temperature
        if learnable_temperature:
            if use_log_temperature:
                self.log_temperature = nn.Parameter(
                    torch.log(torch.tensor(temperature))
                )
            else:
                self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.temperature = temperature

        # Pretraining temperature handling (separate from regular temperature)
        # This allows decoupling pretraining loss temperature from regular loss temperature
        # PROOF uses learnable pretraining temperature while regular loss uses temperature=1.0
        self.use_log_pretraining_temperature = use_log_pretraining_temperature
        if learnable_pretraining_temperature:
            if use_log_pretraining_temperature:
                self.log_pretraining_temperature = nn.Parameter(
                    torch.log(torch.tensor(pretraining_temperature))
                )
            else:
                self.pretraining_temperature = nn.Parameter(torch.tensor(pretraining_temperature))
        else:
            self.pretraining_temperature = pretraining_temperature

        # Logit bias handling (extracted from CLIP model during initialization)
        if logit_bias is not None:
            if learnable_logit_bias:
                self.logit_bias = nn.Parameter(torch.tensor(logit_bias))
            else:
                self.register_buffer("logit_bias", torch.tensor(logit_bias))
        else:
            self.logit_bias = None

        # Default text templates if not provided
        if text_templates is None:
            self.text_templates = [
                "a photo of a {}.",
                "a photo of the {}.",
            ]
        else:
            self.text_templates = text_templates

        # Control text encoder training
        self.freeze_text_encoder = freeze_text_encoder
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        else:
            for param in self.text_encoder.parameters():
                param.requires_grad = True

        # Initialize text embeddings buffer (will be set via set_class_names)
        # Note: When text encoder is trainable, embeddings are recomputed in forward pass
        self.register_buffer("text_embeddings", torch.zeros(num_classes, feature_dim))
        self.text_embeddings_initialized = False

        # For hybrid modes, create a learnable classifier
        if self.mode == "hybrid":
            self.learned_classifier = _create_learned_classifier(
                in_features=feature_dim,
                num_classes=num_classes,
                classifier_type=learned_classifier_type,
                hidden_dim=hidden_dim,
                dropout=dropout,
                normalize=normalize,
            )
        else:
            self.learned_classifier = None

        # Initialize class names if provided
        if class_names is not None:
            self.set_class_names(class_names)

    def set_class_names(
        self, class_names: List[str], class_indices: Optional[List[int]] = None
    ) -> None:
        """Store class names and optionally precompute text embeddings.

        Args:
            class_names: List of class names
            class_indices: Optional list of class indices to update (for incremental learning)
                          If None, assumes class_names correspond to indices 0, 1, 2, ...
        """
        if class_indices is None:
            class_indices = list(range(len(class_names)))

        # Store class names (required for trainable text encoder)
        if not hasattr(self, "_class_names"):
            self._class_names = [None] * self.num_classes
        for i, class_idx in enumerate(class_indices):
            if class_idx < self.num_classes:
                self._class_names[class_idx] = class_names[i]

        # Only precompute and cache embeddings when text encoder is frozen
        # For trainable text encoder, embeddings are computed dynamically in forward pass
        if self.freeze_text_encoder:
            with torch.no_grad():
                new_text_embeddings = self._encode_class_names(class_names)

            # Cache text embeddings
            for i, class_idx in enumerate(class_indices):
                if class_idx < self.num_classes:
                    self.text_embeddings[class_idx] = new_text_embeddings[i]

        self.text_embeddings_initialized = True

    def _encode_captions(self, captions: List[str]) -> torch.Tensor:
        """Encode caption strings to text embeddings.

        Args:
            captions: List of caption strings [batch_size]

        Returns:
            Text embeddings tensor [batch_size, feature_dim]
        """
        if not captions:
            return torch.empty(0, self.feature_dim, device=self.device)

        # Tokenize captions
        text_tokens = self.tokenizer(captions)
        text_tokens = text_tokens.to(self.device)

        # Extract text features
        text_features = self.text_encoder(text_tokens)
        
        # Apply text projection if available (for projection tuning)
        if self.text_projection is not None:
            text_features = self.text_projection(text_features)

        # Normalize (required for contrastive learning)
        if self.normalize:
            text_features = F.normalize(text_features, p=2, dim=-1)

        return text_features

    def _encode_class_names(self, class_names: List[str], batch_size: int = 64) -> torch.Tensor:
        """Encode multiple class names to text embeddings with batching to avoid OOM.

        Args:
            class_names: List of class names to encode
            batch_size: Maximum number of prompts to process in one batch (default: 64)
                       This limits memory usage when encoding many classes (e.g., 200 classes × 2 templates = 400 prompts)

        Returns:
            Text embeddings tensor [num_classes, feature_dim]
        """
        if not class_names:
            return torch.empty(0, self.feature_dim, device=self.device)

        # Normalize class names if enabled (replace underscores with spaces)
        if self.normalize_class_names:
            class_names = [name.replace("_", " ") for name in class_names]

        # Generate all prompts for all classes
        all_prompts = []
        for class_name in class_names:
            prompts = [template.format(class_name) for template in self.text_templates]
            all_prompts.extend(prompts)

        # Process prompts in batches to avoid OOM
        # When we have many classes (e.g., step 9 with 200 classes × 2 templates = 400 prompts),
        # processing all at once can cause OOM in text encoder's attention
        all_text_features_list = []
        num_prompts = len(all_prompts)
        
        for start_idx in range(0, num_prompts, batch_size):
            end_idx = min(start_idx + batch_size, num_prompts)
            batch_prompts = all_prompts[start_idx:end_idx]
            
            # Tokenize batch
            text_tokens = self.tokenizer(batch_prompts)
            text_tokens = text_tokens.to(self.device)
            
            # Extract text features for this batch
            batch_text_features = self.text_encoder(text_tokens)
            
            # Apply text projection if available (for projection tuning)
            if self.text_projection is not None:
                batch_text_features = self.text_projection(batch_text_features)
            
            # Normalize if specified
            if self.normalize:
                batch_text_features = F.normalize(batch_text_features, p=2, dim=-1)
            
            all_text_features_list.append(batch_text_features)
        
        # Concatenate all batches
        all_text_features = torch.cat(all_text_features_list, dim=0)

        # Reshape to [num_classes, num_templates, feature_dim]
        num_templates = len(self.text_templates)
        all_text_features = all_text_features.view(len(class_names), num_templates, -1)

        # Ensemble over templates if specified
        if self.ensemble_text:
            class_embeddings = all_text_features.mean(
                dim=1
            )  # [num_classes, feature_dim]
        else:
            # Use first template only
            class_embeddings = all_text_features[:, 0, :]  # [num_classes, feature_dim]

        return class_embeddings

    def _compute_text_embeddings(self, batch_size: Optional[int] = None) -> torch.Tensor:
        """Compute text embeddings from class names (used when text encoder is trainable).

        Args:
            batch_size: Current batch size from forward pass, used to determine text encoding batch size.
                       If None, uses default batch_size=64 in _encode_class_names.

        Returns:
            Text embeddings tensor [num_classes, feature_dim]
        """
        if not hasattr(self, "_class_names") or all(name is None for name in self._class_names):
            # Fallback to cached embeddings if class names not stored
            return self.text_embeddings

        # Separate valid class names from None entries
        valid_class_names = []
        valid_indices = []
        for idx, class_name in enumerate(self._class_names):
            if class_name is not None:
                valid_class_names.append(class_name)
                valid_indices.append(idx)

        # Batch encode all valid class names
        # Use current batch size if available to optimize memory usage
        if valid_class_names:
            if batch_size is not None:
                valid_embeddings = self._encode_class_names(valid_class_names, batch_size=batch_size)
            else:
                valid_embeddings = self._encode_class_names(valid_class_names)
        else:
            valid_embeddings = torch.empty(0, self.feature_dim, device=self.device)

        # Build full embeddings tensor with zeros for uninitialized classes
        all_embeddings = torch.zeros(
            self.num_classes, self.feature_dim, device=self.device
        )
        for i, idx in enumerate(valid_indices):
            all_embeddings[idx] = valid_embeddings[i]

        return all_embeddings

    def forward(
        self,
        x: torch.Tensor,
        return_separate_logits: bool = False,
        text_embeddings: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass using text embeddings and/or learned classifier.

        Args:
            x: Image features [batch_size, feature_dim]
            return_separate_logits: If True and in hybrid mode, return (text_logits, learned_logits) separately
                                    instead of combined logits. Used for competitive distillation.
            text_embeddings: Optional pre-computed text embeddings [num_classes, feature_dim].
                           If provided, uses these instead of computing (avoids duplicate encoding).

        Returns:
            If return_separate_logits=False or mode='text':
                Logits [batch_size, num_classes]
            If return_separate_logits=True and mode='hybrid':
                Tuple of (text_logits, learned_logits) [batch_size, num_classes] each
        """
        # Get temperature value
        if self.use_log_temperature:
            temperature = torch.exp(self.log_temperature)
        else:
            temperature = self.temperature

        if self.mode == "text":
            # Text encoder only (can be frozen or trainable)
            if not self.text_embeddings_initialized:
                return torch.zeros(x.size(0), self.num_classes, device=x.device)

            # Get text embeddings (use pre-computed if provided, otherwise compute)
            if text_embeddings is None:
                if self.freeze_text_encoder:
                    text_embeddings = self.text_embeddings
                else:
                    text_embeddings = self._compute_text_embeddings(batch_size=x.size(0))

            # Normalize image features if specified
            if self.normalize:
                x = F.normalize(x, p=2, dim=1)

            # NOTE: PROOF fusion is NOT supported in text mode
            # Text mode may not have learned_classifier/prototypes, making fusion meaningless
            # Use hybrid mode with hybrid_weight=1.0 for text-only predictions with PROOF fusion

            # Compute cosine similarity scaled by temperature (CLIP-style: multiply by logit_scale.exp())
            logits = torch.matmul(x, text_embeddings.t())
            logits = logits * temperature

            # Apply logit_bias if present (CustomTextCLIP support)
            if self.logit_bias is not None:
                logits = logits + self.logit_bias

            return logits

        elif self.mode == "hybrid":
            # Combination of text and learned classifiers
            # Both text encoder and learned classifier can be trainable

            # Check if we should disable learned classifier during inference
            # This makes inference use CLIP text classifier only (more versatile and elegant)
            if self.disable_learned_classifier_at_inference and not self.training:
                # Use text-only inference during eval mode
                # Get text embeddings (recompute if text encoder is trainable)
                if self.freeze_text_encoder:
                    text_embeddings = self.text_embeddings
                else:
                    text_embeddings = self._compute_text_embeddings(batch_size=x.size(0))
                
                # Normalize image features if specified
                if self.normalize:
                    x = F.normalize(x, p=2, dim=1)
                
                # Compute text-only logits
                logits = torch.matmul(x, text_embeddings.t()) * temperature
                
                # Apply logit_bias if present
                if self.logit_bias is not None:
                    logits = logits + self.logit_bias
                
                return logits

            # Get text embeddings (use pre-computed if provided, otherwise compute)
            if text_embeddings is None:
                if self.freeze_text_encoder:
                    text_embeddings = self.text_embeddings
                else:
                    text_embeddings = self._compute_text_embeddings(batch_size=x.size(0))

            # Normalize image features if specified
            if self.normalize:
                x_norm = F.normalize(x, p=2, dim=1)
            else:
                x_norm = x

            # Apply PROOF fusion if enabled
            # PROOF requires hybrid mode with prototypical classifier for meaningful prototypes
            fused_prototypes = None
            if hasattr(self, "proof_fusion") and self.proof_fusion is not None:
                # Get image prototypes from learned classifier
                if not hasattr(self.learned_classifier, "prototypes"):
                    raise ValueError(
                        "PROOF fusion requires hybrid mode with prototypical classifier. "
                        "The learned_classifier must have 'prototypes' attribute. "
                        "Set classifier.mode='hybrid' and classifier.learned_classifier_type='prototypical'."
                    )
                
                # CRITICAL: Only use prototypes and text embeddings for SEEN classes (prototype_counts > 0)
                # In continual learning, we haven't seen all classes yet
                # Note: Prototypes are always initialized before training, so seen_classes_mask will never be empty
                seen_classes_mask = self.learned_classifier.prototype_counts > 0
                
                # Get only seen class prototypes and text embeddings
                image_prototypes = self.learned_classifier.prototypes.data.clone()
                image_prototypes_seen = image_prototypes[seen_classes_mask]
                text_embeddings_seen = text_embeddings[seen_classes_mask]
                
                # IMPORTANT: Apply vision projection to prototypes if available
                # Prototypes are stored as raw backbone outputs, but need to be projected
                # to match the space of the image features x
                if self.vision_projection is not None:
                    image_prototypes_seen = self.vision_projection(image_prototypes_seen)
                
                # Normalize prototypes if specified
                if self.normalize:
                    image_prototypes_seen = F.normalize(image_prototypes_seen, p=2, dim=1)
                
                # Apply fusion to get fused features and fused prototypes
                # Fused prototypes will be used for prototype loss computation
                x, text_embeddings_seen, fused_prototypes_seen = self.proof_fusion(x_norm, text_embeddings_seen, image_prototypes_seen)
                x_norm = x  # PROOF does not normalize features after fusion

                # Fill seen class prototypes and text embeddings
                text_embeddings[seen_classes_mask] = text_embeddings_seen.to(text_embeddings.dtype)  # Fix autocast issue
                fused_prototypes = torch.zeros_like(image_prototypes)
                fused_prototypes[seen_classes_mask] = fused_prototypes_seen.to(fused_prototypes.dtype)
                
            # Compute raw logits (without temperature scaling or bias)
            text_logits_raw = torch.matmul(x_norm, text_embeddings.t())

            # Learned logits (raw, without temperature scaling)
            # If PROOF fusion enabled, this uses fused prototypes (prototype loss)
            if hasattr(self, "proof_fusion") and self.proof_fusion is not None and fused_prototypes is not None:
                learned_logits_raw = torch.matmul(x, fused_prototypes.t())
            else:
                learned_logits_raw = self.learned_classifier(x)
            
            # Return raw unscaled unbiased logits for competitive distillation if requested
            # This allows distillation to control temperature independently without
            # gradient flow to the model's temperature parameter or logit_bias
            if return_separate_logits:
                return text_logits_raw, learned_logits_raw
            
            # Apply temperature scaling for final predictions
            text_logits = text_logits_raw * temperature
            learned_logits = learned_logits_raw * temperature
            
            # Apply logit_bias AFTER temperature scaling (CustomTextCLIP support)
            # This ensures bias has consistent effect regardless of temperature value
            if self.logit_bias is not None:
                text_logits = text_logits + self.logit_bias
            
            # Weighted combination
            logits = (
                self.hybrid_weight * text_logits
                + (1 - self.hybrid_weight) * learned_logits
            )

            return logits

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def forward_for_pretraining_loss(
        self, 
        image_features: torch.Tensor, 
        targets: torch.Tensor,
        captions: Optional[List[str]] = None,
        text_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for pretraining loss computation (CLIP/SigLIP contrastive loss).

        This method returns normalized image features and text features.
        If captions are provided, encodes them as text features.
        Otherwise, uses class name embeddings for the target classes.

        Args:
            image_features: Image features [batch_size, feature_dim]
            targets: Target class labels [batch_size]
            captions: Optional list of caption strings [batch_size]
            text_embeddings: Optional pre-computed text embeddings [num_classes, feature_dim].
                           If provided, uses these instead of computing (avoids duplicate encoding).

        Returns:
            Tuple of (normalized_image_features, normalized_text_features)
            - normalized_image_features: [batch_size, feature_dim]
            - normalized_text_features: [batch_size, feature_dim]
        """
        if self.mode not in ["text", "hybrid"]:
            raise ValueError(
                f"Pretraining loss only supported for mode='text' or 'hybrid', got mode='{self.mode}'."
            )

        # Get text features either from captions or class names
        if captions is not None:
            # Encode caption text directly
            text_features = self._encode_captions(captions)
        else:
            # Use class name embeddings (original behavior)
            if not self.text_embeddings_initialized:
                raise RuntimeError(
                    "Text embeddings not initialized. Call set_class_names() first."
                )

            # Get text embeddings (use pre-computed if provided, otherwise compute)
            if text_embeddings is not None:
                # Use pre-computed embeddings (avoids duplicate encoding in combined loss)
                all_text_embeddings = text_embeddings
            elif self.freeze_text_encoder:
                all_text_embeddings = self.text_embeddings
            else:
                all_text_embeddings = self._compute_text_embeddings()

            # Extract text embeddings for the target classes
            # targets: [batch_size], text_features: [batch_size, feature_dim]
            text_features = all_text_embeddings[targets]

        # Normalize both image and text features (required for contrastive loss)
        image_features_norm = F.normalize(image_features, p=2, dim=1)
        text_features_norm = text_features  # Text features already normalized

        return image_features_norm, text_features_norm

    def update_prototypes(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """Update prototypes if using prototypical learned classifier in hybrid mode."""
        if self.mode == "hybrid" and hasattr(self.learned_classifier, "update_prototypes"):
            self.learned_classifier.update_prototypes(features, labels)

    def compute_prototypes(
        self,
        features_list: List[torch.Tensor],
        labels_list: List[torch.Tensor],
        reset: bool = True,
    ) -> None:
        """Compute prototypes if using prototypical learned classifier in hybrid mode."""
        if self.mode == "hybrid" and hasattr(self.learned_classifier, "compute_prototypes"):
            self.learned_classifier.compute_prototypes(features_list, labels_list, reset)


class PrototypicalClassifier(nn.Module):
    """Prototypical classifier using class prototypes for zero-shot classification."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        normalize: bool = True,
    ):
        """
        Args:
            in_features: Input feature dimension
            num_classes: Number of output classes
        """
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.normalize = normalize

        # Initialize prototypes as learnable parameters
        self.prototypes = nn.Parameter(torch.zeros(num_classes, in_features))
        # Keep counts as buffer since they're not learnable
        self.register_buffer(
            "prototype_counts", torch.zeros(num_classes, dtype=torch.long)
        )

        # Flag to track if prototypes have been initialized
        self.prototypes_initialized = False

    def update_prototypes(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """Update class prototypes with new features.

        Args:
            features: Feature vectors [batch_size, in_features]
            labels: Class labels [batch_size]
        """
        for c in range(self.num_classes):
            # Find all samples of this class in the batch
            idx = (labels == c).nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                continue

            # Get features for this class
            class_features = features[idx]
            class_count = len(idx)

            # Update running average of prototypes
            current_count = self.prototype_counts[c]
            total_count = current_count + class_count

            if total_count > 0:  # Avoid division by zero
                # Use .data to modify the parameter without triggering autograd
                new_prototype = (
                    current_count * self.prototypes[c].data + class_features.sum(dim=0)
                ) / total_count
                self.prototypes.data[c] = new_prototype
                self.prototype_counts[c] = total_count

        # Mark prototypes as initialized if any were updated
        if features.size(0) > 0:
            self.prototypes_initialized = True

    def compute_prototypes(
        self,
        features_list: List[torch.Tensor],
        labels_list: List[torch.Tensor],
        reset: bool = True,
    ) -> None:
        """Compute class prototypes from a list of features and labels.

        Args:
            features_list: List of feature tensors
            labels_list: List of label tensors
            reset: Whether to reset existing prototypes before computing
        """
        # Reset prototypes and counts if requested
        if reset:
            # For learnable parameters, we need to avoid in-place operations
            # Create a new tensor with zeros instead of using zero_()
            self.prototypes.data = torch.zeros_like(self.prototypes)
            self.prototype_counts.zero_()

        # Process each batch of features and labels
        for features, labels in zip(features_list, labels_list):
            self.update_prototypes(features, labels)

        # Mark prototypes as initialized
        self.prototypes_initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using similarity between features and prototypes.

        Args:
            x: Input features [batch_size, in_features]

        Returns:
            Logits based on similarity [batch_size, num_classes]
        """
        # Check if prototypes have been initialized
        if not self.prototypes_initialized:
            # Return zeros if prototypes haven't been initialized yet
            return torch.zeros(x.size(0), self.num_classes, device=x.device)

        if self.normalize:
            # Normalize feature vectors and prototypes for cosine similarity
            x_features = F.normalize(x, p=2, dim=1)
            prototypes_features = F.normalize(self.prototypes, p=2, dim=1)
            # Compute cosine similarity
            logits = torch.matmul(x_features, prototypes_features.t())
        else:
            # Use dot product without normalization
            logits = torch.matmul(x, self.prototypes.t())

        return logits


class ClassifierHead(nn.Module):
    """Classifier head for feature vectors."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        classifier_type: str = "linear",
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        temperature: float = 1.0,
        normalize: bool = True,
        learnable_temperature: bool = False,
        use_log_temperature: bool = False,
    ):
        """
        Args:
            in_features: Input feature dimension
            num_classes: Number of output classes
            classifier_type: Type of classifier (linear, mlp, or prototypical)
            hidden_dim: Hidden dimension for MLP (only used if classifier_type is mlp)
            dropout: Dropout probability
            temperature: Temperature scaling parameter for logits
            learnable_temperature: Whether the temperature should be a learnable parameter
            use_log_temperature: If learnable_temperature=True, whether to parameterize in log space for stability
        """
        super().__init__()
        self.classifier_type = classifier_type
        self.use_log_temperature = use_log_temperature
        if learnable_temperature:
            if use_log_temperature:
                # Store temperature in log space for better optimization stability
                self.log_temperature = nn.Parameter(
                    torch.log(torch.tensor(temperature))
                )
            else:
                # Store temperature directly
                self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.temperature = temperature

        # Create classifier using shared function
        if classifier_type in ["linear", "mlp", "prototypical"]:
            self.classifier = _create_learned_classifier(
                in_features=in_features,
                num_classes=num_classes,
                classifier_type=classifier_type,
                hidden_dim=hidden_dim,
                dropout=dropout,
                normalize=normalize,
            )
        elif classifier_type == "clip_text":
            # CLIP text-based classifier - requires text_encoder and tokenizer
            # These will be passed during initialization from PretrainedModel
            raise ValueError(
                "CLIP text classifier must be created through PretrainedModel, "
                "not through ClassifierHead directly."
            )
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with temperature scaling."""
        logits = self.classifier(x)
        if self.use_log_temperature:
            # Convert log_temperature back to actual temperature using exp
            temperature = torch.exp(self.log_temperature)
        else:
            # Use temperature parameter directly
            temperature = self.temperature
        return logits / temperature

    def update_prototypes(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """Update prototypes if using prototypical classifier."""
        if hasattr(self.classifier, "update_prototypes"):
            self.classifier.update_prototypes(features, labels)

    def compute_prototypes(
        self,
        features_list: List[torch.Tensor],
        labels_list: List[torch.Tensor],
        reset: bool = True,
    ) -> None:
        """Compute prototypes if using prototypical classifier."""
        if hasattr(self.classifier, "compute_prototypes"):
            self.classifier.compute_prototypes(features_list, labels_list, reset=reset)


class PretrainedModel(nn.Module):
    """Wrapper for pretrained models."""

    def __init__(
        self,
        model_config: Dict[str, Any],
        num_classes: int,
        cache_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model_config: Model configuration
            num_classes: Number of output classes
            cache_dir: Optional cache directory for model weights
            device: Optional device to place model on
        """
        super().__init__()

        self.model_name = model_config["name"]
        self.source = model_config["source"]
        self.pretrained = model_config.get("pretrained", True)
        self.freeze_backbone = model_config.get("freeze_backbone", False)
        self.freeze_classifier = model_config.get("freeze_classifier", False)
        self.skip_blocks = model_config.get("skip_blocks", 0)
        self.skip_final_norm = model_config.get("skip_final_norm", False)
        self.skip_proj = model_config.get("skip_proj", None)
        if self.skip_proj is not None and self.source.lower() != "openclip":
            raise ValueError(
                "The 'skip_proj' option is only supported for OpenCLIP models"
            )
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Store cache directory
        self.cache_dir = cache_dir

        # Get classifier configuration
        classifier_config = model_config.get("classifier", {})
        classifier_type = classifier_config.get("type", "linear")
        hidden_dim = classifier_config.get("hidden_dim", None)
        dropout = classifier_config.get("dropout", 0.0)
        temperature = classifier_config.get("temperature", 1.0)
        normalize = classifier_config.get("normalize", False)
        learnable_temperature = classifier_config.get("learnable_temperature", False)
        use_log_temperature = classifier_config.get("use_log_temperature", False)
        
        # Pretraining temperature (separate from regular temperature for decoupling)
        pretraining_temperature = classifier_config.get("pretraining_temperature", 1.0)
        learnable_pretraining_temperature = classifier_config.get("learnable_pretraining_temperature", False)
        use_log_pretraining_temperature = classifier_config.get("use_log_pretraining_temperature", False)

        # CLIP text encoder configuration
        self.use_text_encoder = (
            classifier_type == "clip_text" and self.source.lower() == "openclip"
        )

        # Pretraining loss configuration (for CLIP models)
        self.use_pretraining_loss = classifier_config.get("use_pretraining_loss", False)
        self.pretraining_loss_type = classifier_config.get("pretraining_loss_type", "clip")
        pretraining_loss_weight = classifier_config.get("pretraining_loss_weight", 1.0)
        learnable_pretraining_loss_weight = classifier_config.get("learnable_pretraining_loss_weight", False)
        self.use_regular_loss = classifier_config.get("use_regular_loss", False)
        regular_loss_weight = classifier_config.get("regular_loss_weight", 1.0)
        learnable_regular_loss_weight = classifier_config.get("learnable_regular_loss_weight", False)
        self.supervised_contrastive = classifier_config.get("supervised_contrastive", False)

        # Make loss weights learnable if specified
        if learnable_pretraining_loss_weight:
            self.pretraining_loss_weight = nn.Parameter(torch.tensor(pretraining_loss_weight))
        else:
            self.pretraining_loss_weight = pretraining_loss_weight

        if learnable_regular_loss_weight:
            self.regular_loss_weight = nn.Parameter(torch.tensor(regular_loss_weight))
        else:
            self.regular_loss_weight = regular_loss_weight

        # SAE configuration
        self.sae_config = model_config.get("sae", {})
        self.use_sae = self.sae_config.get("use_sae", False)
        self.sae_checkpoint_path = self.sae_config.get("checkpoint_path", None)
        self.sae_layer = self.sae_config.get("layer", -2)
        self.sae_token_type = self.sae_config.get("token_type", "all")
        self.sae = None

        if self.use_sae and not self.sae_checkpoint_path:
            raise ValueError(
                "SAE checkpoint path must be provided when use_sae is True"
            )

        # Load backbone (and optionally text encoder for CLIP)
        if self.use_text_encoder:
            (
                self.backbone,
                self.feature_dim,
                text_encoder,
                tokenizer,
                clip_model,
            ) = self._load_backbone()
        else:
            self.backbone, self.feature_dim = self._load_backbone()

        # Freeze backbone if specified
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Create classifier
        if self.use_text_encoder:
            # Create CLIP text-based classifier
            mode = classifier_config.get("mode", "text")
            text_templates = classifier_config.get("text_templates", None)
            hybrid_weight = classifier_config.get("hybrid_weight", 0.5)
            learnable_hybrid_weight = classifier_config.get("learnable_hybrid_weight", False)
            ensemble_text = classifier_config.get("ensemble_text", False)
            freeze_text_encoder = classifier_config.get("freeze_text_encoder", False)
            learned_classifier_type = classifier_config.get(
                "learned_classifier_type", "linear"
            )
            use_pretrained_temperature = classifier_config.get(
                "use_pretrained_temperature", True
            )
            logit_bias = classifier_config.get("logit_bias", None)
            use_pretrained_logit_bias = classifier_config.get(
                "use_pretrained_logit_bias", False
            )
            learnable_logit_bias = classifier_config.get("learnable_logit_bias", False)
            normalize_class_names = classifier_config.get("normalize_class_names", False)
            disable_learned_classifier_at_inference = classifier_config.get(
                "disable_learned_classifier_at_inference", False
            )

            # Extract CLIP's pre-trained temperature if requested
            if use_pretrained_temperature and hasattr(clip_model, "logit_scale"):
                # CLIP stores temperature as log(scale), we need exp(logit_scale)
                temperature = clip_model.logit_scale.exp().item()
                print(f"Using CLIP pre-trained temperature: {temperature:.2f}")
            elif use_pretrained_temperature:
                print(
                    f"Warning: use_pretrained_temperature=True but model has no logit_scale. Using temperature={temperature}"
                )
            elif temperature is not None:
                print(f"Using manually configured temperature: {temperature:.2f}")

            # Extract CLIP's pre-trained temperature for pretraining loss if requested
            use_pretrained_pretraining_temperature = classifier_config.get(
                "use_pretrained_pretraining_temperature", True
            )
            if use_pretrained_pretraining_temperature and hasattr(clip_model, "logit_scale"):
                # CLIP stores temperature as log(scale), we need exp(logit_scale)
                pretraining_temperature = clip_model.logit_scale.exp().item()
                print(f"Using CLIP pre-trained pretraining temperature: {pretraining_temperature:.2f}")
            elif use_pretrained_pretraining_temperature:
                print(
                    f"Warning: use_pretrained_pretraining_temperature=True but model has no logit_scale. Using pretraining_temperature={pretraining_temperature}"
                )
            elif pretraining_temperature is not None:
                print(f"Using manually configured pretraining temperature: {pretraining_temperature:.2f}")

            # Extract CLIP's logit_bias if requested (CustomTextCLIP support)
            if use_pretrained_logit_bias:
                if (
                    hasattr(clip_model, "logit_bias")
                    and clip_model.logit_bias is not None
                ):
                    logit_bias = clip_model.logit_bias.item()
                    print(f"Using CLIP pre-trained logit_bias: {logit_bias:.4f}")
                else:
                    print(
                        f"Warning: use_pretrained_logit_bias=True but model has no logit_bias. Using logit_bias={logit_bias}"
                    )
            elif logit_bias is not None:
                print(f"Using manually configured logit_bias: {logit_bias:.4f}")

            self.classifier = CLIPClassifier(
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                num_classes=num_classes,
                feature_dim=self.feature_dim,
                mode=mode,
                text_templates=text_templates,
                temperature=temperature,
                normalize=normalize,
                hybrid_weight=hybrid_weight,
                ensemble_text=ensemble_text,
                freeze_text_encoder=freeze_text_encoder,
                learned_classifier_type=learned_classifier_type,
                hidden_dim=hidden_dim,
                dropout=dropout,
                learnable_temperature=learnable_temperature,
                use_log_temperature=use_log_temperature,
                pretraining_temperature=pretraining_temperature,
                learnable_pretraining_temperature=learnable_pretraining_temperature,
                use_log_pretraining_temperature=use_log_pretraining_temperature,
                logit_bias=logit_bias,
                learnable_logit_bias=learnable_logit_bias,
                learnable_hybrid_weight=learnable_hybrid_weight,
                normalize_class_names=normalize_class_names,
                disable_learned_classifier_at_inference=disable_learned_classifier_at_inference,
                device=self.device,
            )

            # Validate pretraining loss configuration
            if self.use_pretraining_loss:
                if mode == "hybrid" and not self.use_regular_loss:
                    raise ValueError(
                        "Hybrid mode with pretraining loss requires use_regular_loss=true "
                        "because hybrid combines text embeddings with learned classifier."
                    )
                weight_str = f"{self.pretraining_loss_weight.item() if isinstance(self.pretraining_loss_weight, nn.Parameter) else self.pretraining_loss_weight}"
                learnable_str = " (learnable)" if isinstance(self.pretraining_loss_weight, nn.Parameter) else ""
                print(
                    f"Using {self.pretraining_loss_type.upper()} pretraining loss "
                    f"(weight={weight_str}{learnable_str})"
                )
                if self.use_regular_loss:
                    reg_weight_str = f"{self.regular_loss_weight.item() if isinstance(self.regular_loss_weight, nn.Parameter) else self.regular_loss_weight}"
                    reg_learnable_str = " (learnable)" if isinstance(self.regular_loss_weight, nn.Parameter) else ""
                    print(
                        f"Also using regular cross-entropy loss "
                        f"(weight={reg_weight_str}{reg_learnable_str})"
                    )
        else:
            # Create standard classifier
            self.classifier = ClassifierHead(
                in_features=self.feature_dim,
                num_classes=num_classes,
                classifier_type=classifier_type,
                hidden_dim=hidden_dim,
                dropout=dropout,
                temperature=temperature,
                normalize=normalize,
                learnable_temperature=learnable_temperature,
                use_log_temperature=use_log_temperature,
            )

        # Freeze classifier if specified
        if self.freeze_classifier:
            for name, param in self.classifier.named_parameters():
                # In case of learnable temperature and logit bias, we don't freeze it
                if "temperature" not in name or "logit_bias" not in name:
                    param.requires_grad = False

    def _load_backbone(self):
        """Load backbone model based on configuration.

        Returns:
            For standard models: (backbone, feature_dim)
            For CLIP with text encoder: (backbone, feature_dim, text_encoder, tokenizer)
        """
        if self.source.lower() == "timm":
            model = timm.create_model(
                self.model_name,
                pretrained=self.pretrained,
                num_classes=0,  # Remove classifier
                cache_dir=self.cache_dir,
            )
            # Get feature dimension
            if hasattr(model, "num_features"):
                feature_dim = model.num_features
            else:
                # For ViT models
                feature_dim = model.embed_dim

            # Skip the norm layer if configured (for ViT models)
            if self.skip_final_norm and hasattr(model, "norm"):
                # Replace the norm layer with an identity layer
                model.norm = nn.Identity()
                print(f"Replaced norm layer with Identity in {self.model_name}")

            # Load SAE if configured
            if self.use_sae and hasattr(saev, "nn"):
                self.sae = saev.nn.load(self.sae_checkpoint_path)
                # Wrap the model to intercept activations at the specified layer
                if hasattr(model, "blocks") and isinstance(model.blocks, nn.Sequential):
                    self._wrap_transformer_blocks(model)

            # Apply block skipping if configured
            if self.skip_blocks > 0:
                if hasattr(model, "blocks") and isinstance(model.blocks, nn.Sequential):
                    self._skip_transformer_blocks(model)

            return model, feature_dim

        elif self.source.lower() == "openclip":
            model, _, _ = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained if self.pretrained else None,
                cache_dir=self.cache_dir,
            )

            # Skip the norm layer if configured (for OpenCLIP ViT models)
            if self.skip_final_norm and hasattr(model.visual, "ln_post"):
                # Replace the ln_post (norm) layer with an identity layer
                model.visual.ln_post = nn.Identity()
                print(
                    f"Replaced ln_post layer with Identity in OpenCLIP {self.model_name}"
                )

            # Skip the projection layer if configured (for OpenCLIP ViT models)
            if self.skip_proj and hasattr(model.visual, "proj"):
                # Replace the projection layer with an identity layer or None
                if model.visual.proj is not None:
                    # Update the output dimension to match the pre-projection dimension
                    if hasattr(model.visual, "output_dim") and hasattr(
                        model.visual.proj, "shape"
                    ):
                        model.visual.output_dim = model.visual.proj.shape[0]
                    model.visual.proj = None
                    print(f"Removed projection layer in OpenCLIP {self.model_name}")

            # Get feature dimension (for CLIP models) - after potential modifications
            # For OpenCLIP
            if hasattr(model.visual, "output_dim"):
                feature_dim = model.visual.output_dim
            # For CustomTextCLIP
            elif hasattr(model, "text") and hasattr(model.text, "output_dim"):
                feature_dim = model.text.output_dim
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")

            # Load SAE if configured
            if self.use_sae and hasattr(saev, "nn"):
                self.sae = saev.nn.load(self.sae_checkpoint_path)
                # Wrap the visual transformer blocks to intercept activations at the specified layer
                if hasattr(model.visual, "transformer") and hasattr(
                    model.visual.transformer, "resblocks"
                ):
                    self._wrap_transformer_blocks(model.visual)

            # Apply block skipping if configured
            if self.skip_blocks > 0:
                if hasattr(model.visual, "transformer") and hasattr(
                    model.visual.transformer, "resblocks"
                ):
                    self._skip_transformer_blocks(model.visual)

            # Return visual encoder + optionally text encoder and tokenizer
            if self.use_text_encoder:
                # Return full model components for CLIP text classification
                # Wrap the full CLIP model's encode_text pipeline in a module
                text_encoder = CLIPTextEncoderWrapper(model)
                tokenizer = open_clip.get_tokenizer(self.model_name)
                print(
                    f"Loaded CLIP model with text encoder wrapper for {self.model_name}"
                )
                # Return full CLIP model for accessing logit_scale
                return model.visual, feature_dim, text_encoder, tokenizer, model
            else:
                # Return only the visual part of CLIP (original behavior)
                return model.visual, feature_dim

        else:
            raise ValueError(f"Unsupported model source: {self.source}")

    def _skip_transformer_blocks(self, model):
        """Skip the final k transformer blocks if skip_blocks > 0.
        This is separate from the SAE functionality.
        """
        # Determine which attribute contains the transformer blocks
        if hasattr(model, "blocks"):
            blocks_attr = "blocks"
        elif hasattr(model, "transformer") and hasattr(model.transformer, "resblocks"):
            blocks_attr = "transformer.resblocks"
        else:
            raise ValueError(
                f"Cannot find transformer blocks in model {self.model_name}"
            )

        # Get the blocks
        if blocks_attr == "blocks":
            blocks = model.blocks
        else:
            blocks = model.transformer.resblocks

        num_blocks = len(blocks)

        # Skip the final k blocks
        if self.skip_blocks >= num_blocks:
            raise ValueError(
                f"Cannot skip {self.skip_blocks} blocks when model only has {num_blocks} blocks"
            )

        # Create a new Sequential module with only the blocks we want to keep
        if blocks_attr == "blocks":
            model.blocks = nn.Sequential(*list(blocks)[: num_blocks - self.skip_blocks])
        else:
            model.transformer.resblocks = nn.Sequential(
                *list(blocks)[: num_blocks - self.skip_blocks]
            )

        print(
            f"Skipped the final {self.skip_blocks} blocks.",
            f"New number of blocks: {num_blocks - self.skip_blocks}",
        )

    def _wrap_transformer_blocks(self, model):
        """Wrap transformer blocks to intercept activations at the specified layer."""
        # Determine which attribute contains the transformer blocks
        if hasattr(model, "blocks"):
            blocks_attr = "blocks"
        elif hasattr(model, "transformer") and hasattr(model.transformer, "resblocks"):
            blocks_attr = "transformer.resblocks"
        else:
            raise ValueError(
                f"Cannot find transformer blocks in model {self.model_name}"
            )

        # Get the blocks
        if blocks_attr == "blocks":
            blocks = model.blocks
        else:
            blocks = model.transformer.resblocks

        # Calculate the actual layer index (handle negative indexing)
        num_blocks = len(blocks)
        layer_idx = (
            self.sae_layer if self.sae_layer >= 0 else num_blocks + self.sae_layer
        )

        # Ensure the layer index is valid
        if layer_idx < 0 or layer_idx >= num_blocks:
            raise ValueError(
                f"Invalid layer index {self.sae_layer} for model with {num_blocks} blocks"
            )

        # Get the target block
        target_block = blocks[layer_idx]

        # Store the original forward method
        self.original_forward = target_block.forward

        # Define a new forward method that applies SAE
        def new_forward(self_block, x, *args, **kwargs):
            # Call the original forward method
            output = self.original_forward(x, *args, **kwargs)

            # Apply SAE to the output
            if self.use_sae and self.sae is not None:
                # Determine which tokens to apply SAE to based on configuration
                if self.sae_token_type == "cls":
                    # Apply SAE only to CLS token (first token)
                    cls_token = output[:, 0:1, :]
                    reconstructed_cls, _, _, _ = self.sae(cls_token)
                    # Replace only the CLS token in the output
                    reconstructed_output = output.clone()
                    reconstructed_output[:, 0:1, :] = reconstructed_cls
                elif self.sae_token_type == "patch":
                    # Apply SAE only to patch tokens (all except first token)
                    patch_tokens = output[:, 1:, :]
                    reconstructed_patches, _, _, _ = self.sae(patch_tokens)
                    # Replace only the patch tokens in the output
                    reconstructed_output = output.clone()
                    reconstructed_output[:, 1:, :] = reconstructed_patches
                else:  # "all" - default
                    # Apply SAE to all tokens
                    reconstructed_output, _, _, _ = self.sae(output)

                return reconstructed_output

            return output

        # Replace the forward method
        import types

        target_block.forward = types.MethodType(new_forward, target_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.forward_features(x)
        logits = self.classifier(features)
        return logits

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification."""
        return self.backbone(x)

    def forward_for_pretraining_loss(
        self, x: torch.Tensor, targets: torch.Tensor, captions: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for pretraining loss computation.

        This method extracts image features and gets corresponding text features
        for computing CLIP-style contrastive loss.

        Args:
            x: Input images [batch_size, C, H, W]
            targets: Target class labels [batch_size]
            captions: Optional list of caption strings [batch_size]

        Returns:
            Tuple of (normalized_image_features, normalized_text_features)
        """
        if not self.use_text_encoder:
            raise ValueError(
                "forward_for_pretraining_loss only available for CLIP text-based classifiers"
            )

        # Extract image features
        image_features = self.forward_features(x)

        # Get text features using the classifier
        return self.classifier.forward_for_pretraining_loss(image_features, targets, captions)

    def set_class_names(
        self, class_names: List[str], class_indices: Optional[List[int]] = None
    ) -> None:
        """Set class names for CLIP text classifier.

        Args:
            class_names: List of class names
            class_indices: Optional list of class indices (for incremental updates)
        """
        if isinstance(self.classifier, CLIPClassifier):
            self.classifier.set_class_names(class_names, class_indices)
        else:
            print("Warning: set_class_names called on non-CLIP classifier")

    def update_prototypes(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """Update prototypes if using prototypical classifier."""
        if hasattr(self.classifier, "update_prototypes"):
            self.classifier.update_prototypes(features, labels)

    def compute_prototypes(
        self,
        features_list: List[torch.Tensor],
        labels_list: List[torch.Tensor],
        reset: bool = True,
    ) -> None:
        """Compute prototypes if using prototypical classifier.

        Args:
            features_list: List of feature tensors
            labels_list: List of label tensors
            reset: Whether to reset existing prototypes before computing
        """
        if hasattr(self.classifier, "compute_prototypes"):
            self.classifier.compute_prototypes(features_list, labels_list, reset=reset)

    def init_prototypes_from_data(
        self, data_loader: torch.utils.data.DataLoader, reset: bool = True
    ) -> None:
        """Initialize prototypes from a data loader.

        Args:
            data_loader: DataLoader containing samples to compute prototypes from
            reset: Whether to reset existing prototypes before computing
        """
        # Extract features
        features_list = []
        labels_list = []

        # Set model to eval mode
        self.eval()

        # Extract features using the backbone
        with torch.no_grad():
            # Add progress bar with tqdm
            data_iter = tqdm(data_loader, desc="Extracting features for prototypes")
            for batch in data_iter:
                # Handle both 2-tuple and 3-tuple batches (with captions)
                if len(batch) == 3:
                    inputs, targets, _ = batch  # Ignore captions for prototype computation
                else:
                    inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                features = self.forward_features(inputs)

                # Offload features and labels to CPU to save GPU memory
                features_list.append(features.cpu())
                labels_list.append(targets.cpu())

        # Compute prototypes - move data back to device for computation
        # We do this batch by batch to avoid OOM errors
        features_device_list = []
        labels_device_list = []

        # Process in smaller batches to avoid OOM
        for i, (features, labels) in enumerate(zip(features_list, labels_list)):
            # Move back to device
            features_device = features.to(self.device)
            labels_device = labels.to(self.device)

            # Add to device lists
            features_device_list.append(features_device)
            labels_device_list.append(labels_device)

            # Process in chunks to avoid OOM
            if (i + 1) % 10 == 0 or i == len(features_list) - 1:
                # Compute prototypes for this chunk
                self.compute_prototypes(
                    features_device_list, labels_device_list, reset=(reset and i == 0)
                )

                # Clear lists to free memory
                features_device_list = []
                labels_device_list = []


def get_pretrained_normalization_params(
    model_config: Dict[str, Any],
    cache_dir: Optional[str] = None,
) -> Tuple[Tuple[float], Tuple[float]]:
    """
    Get the normalization parameters (mean and std) used by the pretrained model.

    Args:
        model_config: Model configuration
        cache_dir: Optional cache directory for model weights

    Returns:
        Tuple of (mean, std) for normalization
    """
    source = model_config.get("source", "timm").lower()
    model_name = model_config.get("name", "resnet50")

    # Default ImageNet normalization
    default_mean = (0.485, 0.456, 0.406)
    default_std = (0.229, 0.224, 0.225)

    if source == "timm":
        cfg = timm.models.get_pretrained_cfg(model_name)
    elif source == "openclip":
        tag = model_config.get("pretrained", "openai")
        cfg = open_clip.get_pretrained_cfg(model_name, tag)

    # Handle both dictionary and PretrainedCfg object
    if hasattr(cfg, "mean") and hasattr(cfg, "std"):
        # PretrainedCfg object case
        return cfg.mean, cfg.std
    elif isinstance(cfg, dict) and "mean" in cfg and "std" in cfg:
        # Dictionary case
        return cfg["mean"], cfg["std"]

    # Return default ImageNet normalization if we couldn't extract from model
    return default_mean, default_std


def create_model(
    model_config: Dict[str, Any],
    num_classes: int,
    device: str = "cuda",
    cache_dir: Optional[str] = None,
    continual_config: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """
    Create a model based on configuration.

    Args:
        model_config: Model configuration
        num_classes: Number of output classes
        device: Device to put model on
        cache_dir: Optional cache directory for model weights
        continual_config: Optional configuration for continual learning

    Returns:
        Instantiated model
    """
    device_obj = torch.device(device)
    model = PretrainedModel(
        model_config=model_config,
        num_classes=num_classes,
        cache_dir=cache_dir,
        device=device_obj,
    )

    # Add prompt tuning if configured
    if continual_config.get("strategy", "") == "prompt_tuning":
        assert (
            model_config["name"].startswith("vit") and model_config["source"] == "timm"
        ), "Prompt tuning is only supported for timm ViT models"
        model = create_vit_prompted_model(
            base_model=model,
            prompt_config=continual_config.get("prompt_tuning", {}),
            embed_dim=model.feature_dim,
            num_classes=num_classes,
        )
    
    # Add LoRA if configured
    elif continual_config.get("strategy", "") == "lora":
        model = create_lora_model(
            base_model=model,
            lora_config=continual_config.get("lora", {}),
        )
    
    # Add Adapter if configured
    elif continual_config.get("strategy", "") == "adapter":
        adapter_config = continual_config.get("adapter", {})
        
        # Check if this is a CLIP model with text encoder (CLIPClassifier)
        has_clip_text_encoder = (
            model_config["source"] == "openclip" and 
            hasattr(model, "classifier") and
            hasattr(model.classifier, "text_encoder") and
            isinstance(model.classifier.text_encoder, CLIPTextEncoderWrapper)
        )
        
        if has_clip_text_encoder:
            # CLIP dual-encoder model: visual encoder in backbone, text encoder in classifier
            print("\nDetected CLIP dual-encoder model")
            
            # Apply adapters to vision encoder (backbone) if configured
            if adapter_config.get("adapt_vision", True):
                print("Applying adapters to vision encoder (backbone)...")
                model.backbone = create_adapter_model(
                    base_model=model.backbone,
                    adapter_config=adapter_config,
                    model_source="openclip",
                )
            
            # Apply adapters to text encoder if configured
            if adapter_config.get("adapt_text", True):
                print("Applying adapters to text encoder (transformer)...")
                # Text encoder is wrapped in CLIPTextEncoderWrapper
                # The actual transformer is at clip_model.transformer
                model.classifier.text_encoder.clip_model.transformer = create_adapter_model(
                    base_model=model.classifier.text_encoder.clip_model.transformer,
                    adapter_config=adapter_config,
                    model_source="openclip",
                )
        else:
            # Standard single-encoder model (timm ViT, OpenCLIP visual only, etc.)
            # All models have backbone+classifier structure, adapt the backbone
            print("Applying adapters to backbone...")
            model.backbone = create_adapter_model(
                base_model=model.backbone,
                adapter_config=adapter_config,
                model_source=model_config["source"],
            )
    
    # Add Projection if configured
    elif continual_config.get("strategy", "") == "projection":
        projection_config = continual_config.get("projection", {})
        print("\nApplying projection tuning...")
        model = create_projection_model(
            model=model,
            projection_config=projection_config,
        )

    model = model.to(device_obj)
    return model
