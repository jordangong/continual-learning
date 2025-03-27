from typing import Any, Dict, Optional, Tuple

import open_clip
import timm
import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    """Classifier head for feature vectors."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        classifier_type: str = "linear",
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        """
        Args:
            in_features: Input feature dimension
            num_classes: Number of output classes
            classifier_type: Type of classifier (linear or mlp)
            hidden_dim: Hidden dimension for MLP (only used if classifier_type is mlp)
            dropout: Dropout probability
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
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.classifier(x)


class PretrainedModel(nn.Module):
    """Wrapper for pretrained models."""

    def __init__(
        self,
        model_config: Dict[str, Any],
        num_classes: int,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            model_config: Model configuration
            num_classes: Number of output classes
        """
        super().__init__()

        self.model_name = model_config["name"]
        self.source = model_config["source"]
        self.pretrained = model_config.get("pretrained", True)
        self.freeze_backbone = model_config.get("freeze_backbone", False)

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

    Returns:
        Instantiated model
    """
    model = PretrainedModel(
        model_config=model_config, num_classes=num_classes, cache_dir=cache_dir
    )

    model = model.to(device)
    return model
