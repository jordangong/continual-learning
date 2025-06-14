from typing import Any, Dict, List, Optional, Tuple

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
        """
        super().__init__()
        self.classifier_type = classifier_type
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.temperature = temperature

        if classifier_type == "linear":
            self.classifier = nn.Linear(in_features, num_classes)
        elif classifier_type == "mlp":
            assert hidden_dim is not None, (
                "hidden_dim must be provided for MLP classifier"
            )
            self.classifier = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        elif classifier_type == "prototypical":
            self.classifier = PrototypicalClassifier(
                in_features, num_classes, normalize=normalize
            )
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with temperature scaling."""
        logits = self.classifier(x)
        return logits / self.temperature

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

        # Load backbone
        self.backbone, self.feature_dim = self._load_backbone()

        # Freeze backbone if specified
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Create classifier
        classifier_config = model_config.get("classifier", {})
        classifier_type = classifier_config.get("type", "linear")
        hidden_dim = classifier_config.get("hidden_dim", None)
        dropout = classifier_config.get("dropout", 0.0)
        temperature = classifier_config.get("temperature", 1.0)
        normalize = classifier_config.get("normalize", False)
        learnable_temperature = classifier_config.get("learnable_temperature", False)

        self.classifier = ClassifierHead(
            in_features=self.feature_dim,
            num_classes=num_classes,
            classifier_type=classifier_type,
            hidden_dim=hidden_dim,
            dropout=dropout,
            temperature=temperature,
            normalize=normalize,
            learnable_temperature=learnable_temperature,
        )

        # Freeze classifier if specified
        if self.freeze_classifier:
            for param in self.classifier.parameters():
                param.requires_grad = False

    def _load_backbone(self) -> Tuple[nn.Module, int]:
        """Load backbone model based on configuration."""
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
            feature_dim = model.visual.output_dim

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

            # Return only the visual part of CLIP
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
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification."""
        return self.backbone(x)

    def update_prototypes(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """Update prototypes if using prototypical classifier."""
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
            for inputs, targets in data_iter:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                features = self.backbone(inputs)

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
        # Get the default_cfg from timm
        default_cfg = timm.models.get_pretrained_cfg(model_name)
        # Handle both dictionary and PretrainedCfg object
        if hasattr(default_cfg, "mean") and hasattr(default_cfg, "std"):
            # PretrainedCfg object case
            return default_cfg.mean, default_cfg.std
        elif (
            isinstance(default_cfg, dict)
            and "mean" in default_cfg
            and "std" in default_cfg
        ):
            # Dictionary case
            return default_cfg["mean"], default_cfg["std"]

    elif source == "openclip":
        # For OpenCLIP, we create the model and transforms to get normalization params
        _, preprocess, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=None,  # Don't need weights for this
            cache_dir=cache_dir,
        )
        # Extract normalization parameters from the transform pipeline
        for transform in preprocess.transforms:
            if hasattr(transform, "mean") and hasattr(transform, "std"):
                return transform.mean, transform.std

    # Return default ImageNet normalization if we couldn't extract from model
    return default_mean, default_std


def create_model(
    model_config: Dict[str, Any],
    num_classes: int,
    device: str = "cuda",
    cache_dir: Optional[str] = None,
) -> nn.Module:
    """
    Create a model based on configuration.

    Args:
        model_config: Model configuration
        num_classes: Number of output classes
        device: Device to put model on
        cache_dir: Optional cache directory for model weights

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

    model = model.to(device_obj)
    return model
