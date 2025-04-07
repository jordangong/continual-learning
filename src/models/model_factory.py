from typing import Any, Dict, List, Optional, Tuple

import open_clip
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class PrototypicalClassifier(nn.Module):
    """Prototypical classifier using class prototypes for zero-shot classification."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            in_features: Input feature dimension
            num_classes: Number of output classes
            device: Optional device to place tensors on
        """
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Initialize prototypes with zeros, they will be computed during training
        self.register_buffer("prototypes", torch.zeros(num_classes, in_features))
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
                self.prototypes[c] = (
                    current_count * self.prototypes[c] + class_features.sum(dim=0)
                ) / total_count
                self.prototype_counts[c] = total_count

        # Mark prototypes as initialized if any were updated
        if features.size(0) > 0:
            self.prototypes_initialized = True

    # Removed init_prototypes_from_data method as it's now handled at the model level

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
            self.prototypes.zero_()
            self.prototype_counts.zero_()

        # Process each batch of features and labels
        for features, labels in zip(features_list, labels_list):
            self.update_prototypes(features, labels)

        # Mark prototypes as initialized
        self.prototypes_initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using cosine similarity between features and prototypes.

        Args:
            x: Input features [batch_size, in_features]

        Returns:
            Logits based on cosine similarity [batch_size, num_classes]
        """
        # Check if prototypes have been initialized
        if not self.prototypes_initialized:
            # Return zeros if prototypes haven't been initialized yet
            return torch.zeros(x.size(0), self.num_classes, device=x.device)

        # Normalize feature vectors and prototypes
        x_norm = F.normalize(x, p=2, dim=1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)

        # Compute cosine similarity
        logits = torch.matmul(x_norm, prototypes_norm.t())

        # Scale logits (optional, can be adjusted)
        temperature = 0.1
        return logits / temperature


class ClassifierHead(nn.Module):
    """Classifier head for feature vectors."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        classifier_type: str = "linear",
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            in_features: Input feature dimension
            num_classes: Number of output classes
            classifier_type: Type of classifier (linear, mlp, or prototypical)
            hidden_dim: Hidden dimension for MLP (only used if classifier_type is mlp)
            dropout: Dropout probability
            device: Optional device to place tensors on (used by prototypical)
        """
        super().__init__()

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
            self.classifier = PrototypicalClassifier(in_features, num_classes, device)
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.classifier(x)

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
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Store cache directory
        self.cache_dir = cache_dir

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

        self.classifier = ClassifierHead(
            in_features=self.feature_dim,
            num_classes=num_classes,
            classifier_type=classifier_type,
            hidden_dim=hidden_dim,
            dropout=dropout,
            device=self.device,
        )

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

            return model, feature_dim

        elif self.source.lower() == "openclip":
            model, _, _ = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained if self.pretrained else False,
                cache_dir=self.cache_dir,
            )
            # Get feature dimension (for CLIP models)
            feature_dim = model.visual.output_dim

            # Return only the visual part of CLIP
            return model.visual, feature_dim

        else:
            raise ValueError(f"Unsupported model source: {self.source}")

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
                    features_device_list, labels_device_list, reset=reset
                )

                # Clear lists to free memory
                features_device_list = []
                labels_device_list = []


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
