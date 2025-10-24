"""
Projection-based Parameter-Efficient Fine-Tuning (PEFT).

This module implements projection tuning, one of the simplest PEFT methods:
- Insert a learnable projection layer between frozen backbone and classifier
- Only the projection layer is trainable, backbone remains frozen
- Supports multiple projection types: linear, MLP, residual

Reference:
- A simple but effective approach used in various continual learning works
- Similar to "feature transformation" or "feature adapter" approaches
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class ProjectionLayer(nn.Module):
    """
    Learnable projection layer for PEFT.
    
    This layer projects features from the backbone before passing to classifier.
    Multiple projection types are supported:
    - linear: Simple linear projection
    - mlp: Multi-layer perceptron with activation and dropout
    - residual: Linear projection with residual connection
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        projection_type: str = "linear",
        hidden_dim: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.1,
        activation: str = "gelu",
        init_scale: float = 0.01,
    ):
        """
        Args:
            input_dim: Input feature dimension (from backbone)
            output_dim: Output feature dimension (to classifier)
            projection_type: Type of projection ("linear", "mlp", "residual")
            hidden_dim: Hidden dimension for MLP (if None, uses input_dim)
            num_layers: Number of layers for MLP projection
            dropout: Dropout rate
            activation: Activation function name
            init_scale: Scale for weight initialization
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.projection_type = projection_type
        self.init_scale = init_scale
        
        # Activation function
        activation_map = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
        }
        self.activation = activation_map.get(activation, nn.GELU())
        
        # Default hidden_dim to input_dim if not specified
        if hidden_dim is None:
            hidden_dim = input_dim
        
        # Build projection based on type
        if projection_type == "linear":
            # Simple linear projection
            self.projection = nn.Linear(input_dim, output_dim)
        
        elif projection_type == "mlp":
            # Multi-layer perceptron
            layers = []
            current_dim = input_dim
            
            # Hidden layers
            for i in range(num_layers - 1):
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(self.activation)
                layers.append(nn.Dropout(dropout))
                current_dim = hidden_dim
            
            # Output layer
            layers.append(nn.Linear(current_dim, output_dim))
            
            self.projection = nn.Sequential(*layers)
        
        elif projection_type == "residual":
            # Linear projection with residual connection
            # Only works if input_dim == output_dim
            if input_dim != output_dim:
                raise ValueError(
                    f"Residual projection requires input_dim == output_dim, "
                    f"got {input_dim} != {output_dim}"
                )
            self.projection = nn.Linear(input_dim, output_dim)
        
        else:
            raise ValueError(
                f"Unknown projection_type: {projection_type}. "
                f"Choose from: linear, mlp, residual"
            )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize projection weights with small values."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Small random initialization
                nn.init.normal_(module.weight, std=self.init_scale)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through projection.
        
        Args:
            x: Input features [batch_size, ..., input_dim]
        
        Returns:
            Projected features [batch_size, ..., output_dim]
        """
        if self.projection_type == "residual":
            # Residual connection: output = input + projection(input)
            return x + self.projection(x)
        else:
            # Direct projection
            return self.projection(x)


class ProjectionWrapper(nn.Module):
    """
    Wrapper that adds projection layer between backbone and classifier.
    
    This wrapper:
    1. Freezes the backbone (only projection is trainable)
    2. Inserts projection layer between backbone and classifier
    3. Forwards calls appropriately
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        classifier: nn.Module,
        projection: ProjectionLayer,
        freeze_backbone: bool = True,
    ):
        """
        Args:
            backbone: Pretrained backbone (will be frozen)
            classifier: Classifier head
            projection: Projection layer to insert
            freeze_backbone: Whether to freeze backbone parameters
        """
        super().__init__()
        self.backbone = backbone
        self.projection = projection
        self.classifier = classifier
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through backbone -> projection -> classifier.
        
        Args:
            x: Input tensor [batch_size, ...]
        
        Returns:
            Logits [batch_size, num_classes]
        """
        # Extract features from frozen backbone
        features = self.backbone(x)
        
        # Project features
        projected_features = self.projection(features)
        
        # Classify
        logits = self.classifier(projected_features)
        
        return logits
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features (backbone + projection, no classifier).
        
        Args:
            x: Input tensor [batch_size, ...]
        
        Returns:
            Projected features [batch_size, feature_dim]
        """
        features = self.backbone(x)
        projected_features = self.projection(features)
        return projected_features
    
    def get_projection_parameters(self):
        """Get projection parameters for optimization."""
        return list(self.projection.parameters())


def create_projection_model(
    model: nn.Module,
    projection_config: Dict[str, Any],
) -> nn.Module:
    """
    Create a projection-tuned model from a base model.
    
    This wraps the model's backbone and classifier with a projection layer.
    The backbone is frozen and only the projection layer is trainable.
    
    For CLIP dual-encoder models, creates separate projections for vision and text encoders.
    
    Args:
        model: Base model with backbone and classifier attributes
        projection_config: Configuration dict with keys:
            - projection_type: Type of projection (linear, mlp, residual)
            - hidden_dim: Hidden dimension for MLP (optional)
            - num_layers: Number of layers for MLP
            - dropout: Dropout rate
            - activation: Activation function
            - init_scale: Weight initialization scale
            - adapt_vision: Apply projection to vision encoder (CLIP only)
            - adapt_text: Apply projection to text encoder (CLIP only)
    
    Returns:
        ProjectionWrapper or modified model with projection layers
    """
    # Import here to avoid circular dependency
    from .model_factory import CLIPTextEncoderWrapper, CLIPClassifier
    
    # Extract configuration
    projection_type = projection_config.get("projection_type", "linear")
    hidden_dim = projection_config.get("hidden_dim", None)
    num_layers = projection_config.get("num_layers", 1)
    dropout = projection_config.get("dropout", 0.1)
    activation = projection_config.get("activation", "gelu")
    init_scale = projection_config.get("init_scale", 0.01)
    adapt_vision = projection_config.get("adapt_vision", True)
    adapt_text = projection_config.get("adapt_text", True)
    
    # Get feature dimension from backbone
    if hasattr(model, "feature_dim"):
        feature_dim = model.feature_dim
    elif hasattr(model.backbone, "num_features"):
        feature_dim = model.backbone.num_features
    else:
        raise ValueError("Cannot determine feature dimension from model")
    
    # Check if this is a CLIP model with text encoder
    has_clip_text_encoder = (
        hasattr(model, "classifier") and
        isinstance(model.classifier, CLIPClassifier) and
        hasattr(model.classifier, "text_encoder") and
        isinstance(model.classifier.text_encoder, CLIPTextEncoderWrapper)
    )
    
    if has_clip_text_encoder:
        # CLIP dual-encoder model: apply projections to vision and/or text encoders
        print("\n" + "="*60)
        print("Detected CLIP dual-encoder model")
        print(f"Feature dim: {feature_dim}")
        print("="*60)
        
        vision_projection = None
        text_projection = None
        total_params = 0
        
        # Create vision projection if requested
        if adapt_vision:
            print("\nApplying projection to vision encoder (backbone)...")
            vision_projection = ProjectionLayer(
                input_dim=feature_dim,
                output_dim=feature_dim,
                projection_type=projection_type,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                activation=activation,
                init_scale=init_scale,
            )
            vision_params = sum(p.numel() for p in vision_projection.parameters())
            total_params += vision_params
            print(f"  Vision projection parameters: {vision_params:,}")
        
        # Create text projection if requested
        if adapt_text:
            print("\nApplying projection to text encoder...")
            text_projection = ProjectionLayer(
                input_dim=feature_dim,
                output_dim=feature_dim,
                projection_type=projection_type,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                activation=activation,
                init_scale=init_scale,
            )
            text_params = sum(p.numel() for p in text_projection.parameters())
            total_params += text_params
            print(f"  Text projection parameters: {text_params:,}")
            
            # Add text projection to classifier
            model.classifier.text_projection = text_projection
        
        print(f"\nTotal projection parameters: {total_params:,}")
        print("="*60 + "\n")
        
        # Wrap with vision projection if requested
        if adapt_vision:
            wrapped_model = ProjectionWrapper(
                backbone=model.backbone,
                classifier=model.classifier,
                projection=vision_projection,
                freeze_backbone=True,
            )
            
            # Copy over attributes
            if hasattr(model, "feature_dim"):
                wrapped_model.feature_dim = model.feature_dim
            if hasattr(model, "use_text_encoder"):
                wrapped_model.use_text_encoder = model.use_text_encoder
            if hasattr(model, "device"):
                wrapped_model.device = model.device
            
            # Copy methods
            if hasattr(model, "set_class_names"):
                wrapped_model.set_class_names = model.set_class_names
            if hasattr(model, "forward_for_pretraining_loss"):
                wrapped_model.forward_for_pretraining_loss = model.forward_for_pretraining_loss
            
            return wrapped_model
        else:
            # Only text projection, no vision projection
            return model
    
    else:
        # Standard single-encoder model
        projection = ProjectionLayer(
            input_dim=feature_dim,
            output_dim=feature_dim,
            projection_type=projection_type,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            init_scale=init_scale,
        )
        
        # Print projection info
        num_params = sum(p.numel() for p in projection.parameters())
        print("\n" + "="*60)
        print("Projection Configuration:")
        print(f"  Type: {projection_type}")
        print(f"  Feature dim: {feature_dim}")
        if projection_type == "mlp":
            print(f"  Hidden dim: {hidden_dim or feature_dim}")
            print(f"  Num layers: {num_layers}")
        print(f"  Dropout: {dropout}")
        print(f"  Activation: {activation}")
        print(f"  Init scale: {init_scale}")
        print(f"\nProjection Parameters: {num_params:,}")
        print("="*60 + "\n")
        
        # Wrap model with projection
        wrapped_model = ProjectionWrapper(
            backbone=model.backbone,
            classifier=model.classifier,
            projection=projection,
            freeze_backbone=True,
        )
        
        # Copy over other attributes from original model
        if hasattr(model, "feature_dim"):
            wrapped_model.feature_dim = model.feature_dim
        if hasattr(model, "use_text_encoder"):
            wrapped_model.use_text_encoder = model.use_text_encoder
        if hasattr(model, "device"):
            wrapped_model.device = model.device
        
        # Copy methods that might be needed
        if hasattr(model, "set_class_names"):
            wrapped_model.set_class_names = model.set_class_names
        if hasattr(model, "forward_for_pretraining_loss"):
            wrapped_model.forward_for_pretraining_loss = model.forward_for_pretraining_loss
        
        return wrapped_model
