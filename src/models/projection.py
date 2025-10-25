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


class ExpandableProjection(nn.Module):
    """
    Expandable projection for continual learning (inspired by PROOF).
    
    This module maintains multiple projections (one per continual learning step)
    and fuses their outputs. Previous projections are frozen when learning new ones.
    
    Fusion methods:
    - add: Simple addition (PROOF default)
    - weighted_sum: Weighted sum with learnable/fixed weights
    - attention: Attention-based fusion
    - gated: Gated fusion with learnable gates
    
    Reference:
    - Da-Wei Zhou et al., "Learning without Forgetting for Vision-Language Models"
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
        fusion_method: str = "add",
        learnable_fusion: bool = False,
        freeze_previous: bool = True,
    ):
        """
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            projection_type: Type of projection ("linear", "mlp", "residual")
            hidden_dim: Hidden dimension for MLP
            num_layers: Number of layers for MLP
            dropout: Dropout rate
            activation: Activation function
            init_scale: Weight initialization scale
            fusion_method: How to fuse projection outputs ("add", "weighted_sum", "attention", "gated")
            learnable_fusion: Whether fusion weights/gates are learnable
            freeze_previous: Whether to freeze previous projections when adding new ones
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.projection_type = projection_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation
        self.init_scale = init_scale
        self.fusion_method = fusion_method
        self.learnable_fusion = learnable_fusion
        self.freeze_previous = freeze_previous
        
        # List to store projections (one per step)
        self.projections = nn.ModuleList()
        
        # Fusion parameters will be initialized when adding projections
        # (Don't initialize to None as it causes issues with register_buffer)
        
        # Add first projection
        self.add_projection()
    
    def add_projection(self):
        """Add a new projection for the current continual learning step."""
        # Freeze all previous projections if requested
        if self.freeze_previous:
            for proj in self.projections:
                for param in proj.parameters():
                    param.requires_grad = False
        
        # Create new projection
        new_projection = ProjectionLayer(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            projection_type=self.projection_type,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            activation=self.activation,
            init_scale=self.init_scale,
        )
        
        self.projections.append(new_projection)
        
        # Update fusion parameters
        self._update_fusion_parameters()
        
        print(f"Added projection #{len(self.projections)} (total: {len(self.projections)})")
    
    def _update_fusion_parameters(self):
        """Update fusion parameters based on current number of projections."""
        num_projections = len(self.projections)
        
        if self.fusion_method == "weighted_sum":
            # Initialize/update weights for weighted sum
            if self.learnable_fusion:
                # Learnable weights (softmax normalized)
                self.fusion_weights = nn.Parameter(
                    torch.ones(num_projections) / num_projections
                )
            else:
                # Fixed uniform weights
                new_weights = torch.ones(num_projections) / num_projections
                if hasattr(self, "fusion_weights"):
                    # Update existing buffer by setting it directly
                    self.fusion_weights = new_weights
                else:
                    # Register new buffer for first time
                    self.register_buffer("fusion_weights", new_weights)
        
        elif self.fusion_method == "attention":
            # Simple attention mechanism
            if self.learnable_fusion:
                self.attention = nn.Sequential(
                    nn.Linear(self.output_dim, self.output_dim // 4),
                    nn.ReLU(),
                    nn.Linear(self.output_dim // 4, 1),
                )
            else:
                # Non-learnable attention (just use mean)
                self.attention = None
        
        elif self.fusion_method == "gated":
            # Gated fusion (one gate per projection)
            if self.learnable_fusion:
                self.gates = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.output_dim, self.output_dim // 4),
                        nn.ReLU(),
                        nn.Linear(self.output_dim // 4, 1),
                        nn.Sigmoid(),
                    )
                    for _ in range(num_projections)
                ])
            else:
                # Fixed gates (all equal)
                self.gates = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all projections with fusion.
        
        Args:
            x: Input features [batch_size, input_dim]
        
        Returns:
            Fused projected features [batch_size, output_dim]
        """
        if len(self.projections) == 0:
            raise RuntimeError("No projections available. Call add_projection() first.")
        
        # Get outputs from all projections
        projection_outputs = []
        for proj in self.projections:
            output = proj(x)
            projection_outputs.append(output)
        
        # Stack outputs [num_projections, batch_size, output_dim]
        stacked_outputs = torch.stack(projection_outputs, dim=0)
        
        # Fuse according to selected method
        if self.fusion_method == "add":
            # Simple addition (PROOF default)
            fused = stacked_outputs.sum(dim=0)
        
        elif self.fusion_method == "weighted_sum":
            # Weighted sum with optional learnable weights
            if self.learnable_fusion:
                # Apply softmax to ensure weights sum to 1
                weights = torch.softmax(self.fusion_weights, dim=0)
            else:
                weights = self.fusion_weights
            
            # weights: [num_projections], stacked_outputs: [num_projections, batch_size, output_dim]
            # Reshape weights for broadcasting: [num_projections, 1, 1]
            weights = weights.view(-1, 1, 1)
            fused = (stacked_outputs * weights).sum(dim=0)
        
        elif self.fusion_method == "attention":
            # Attention-based fusion
            if self.learnable_fusion and self.attention is not None:
                # Compute attention scores for each projection output
                # stacked_outputs: [num_projections, batch_size, output_dim]
                attention_scores = []
                for i in range(len(self.projections)):
                    score = self.attention(stacked_outputs[i])  # [batch_size, 1]
                    attention_scores.append(score)
                
                # Stack and normalize scores
                attention_scores = torch.stack(attention_scores, dim=0)  # [num_projections, batch_size, 1]
                attention_weights = torch.softmax(attention_scores, dim=0)
                
                # Weighted sum with attention
                fused = (stacked_outputs * attention_weights).sum(dim=0)
            else:
                # Non-learnable: just use mean
                fused = stacked_outputs.mean(dim=0)
        
        elif self.fusion_method == "gated":
            # Gated fusion
            if self.learnable_fusion and self.gates is not None:
                # Compute gates for each projection
                gated_outputs = []
                for i, gate in enumerate(self.gates):
                    gate_value = gate(stacked_outputs[i])  # [batch_size, 1]
                    gated_output = stacked_outputs[i] * gate_value
                    gated_outputs.append(gated_output)
                
                fused = torch.stack(gated_outputs, dim=0).sum(dim=0)
            else:
                # Non-learnable: just use sum
                fused = stacked_outputs.sum(dim=0)
        
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        return fused
    
    def num_projections(self) -> int:
        """Return the number of projections."""
        return len(self.projections)
    
    def get_active_projection(self) -> nn.Module:
        """Get the most recent (active) projection."""
        if len(self.projections) == 0:
            raise RuntimeError("No projections available.")
        return self.projections[-1]


class ProofFusionLayer(nn.Module):
    """
    PROOF Fusion Layer for continual learning with vision-language models.
    
    This layer implements the fusion mechanism from PROOF (PROjectiOn Fusion):
    - Fuses image features, text features, image prototypes, and context prompts
    - Uses single-head self-attention for fusion
    - Context prompts are expandable (one set per task, previous frozen)
    
    Reference:
    - Da-Wei Zhou et al., "Learning without Forgetting for Vision-Language Models"
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_context_prompts: int = 4,
        freeze_previous: bool = True,
    ):
        """
        Args:
            feature_dim: Feature dimension
            num_context_prompts: Number of learnable context prompts per task
            freeze_previous: Whether to freeze previous context prompts when adding new ones
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_context_prompts = num_context_prompts
        self.freeze_previous = freeze_previous
        
        # Context prompts for each task (expandable like projections)
        self.context_prompts = nn.ParameterList()
        
        # Single-head self-attention (as in PROOF paper)
        self.q_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.k_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.v_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.out_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        
        # Initialize attention weights
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        # Add first set of context prompts
        self.add_context_prompts()
    
    def add_context_prompts(self):
        """Add new context prompts for a new continual learning task."""
        # Freeze all previous context prompts if requested
        if self.freeze_previous:
            for prompt in self.context_prompts:
                prompt.requires_grad = False
        
        # Create new prompts for the current task
        new_prompts = nn.Parameter(
            torch.randn(self.num_context_prompts, self.feature_dim) * 0.02
        )
        self.context_prompts.append(new_prompts)
        
        print(f"Added context prompts #{len(self.context_prompts)} (total: {len(self.context_prompts)})")
    
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        image_prototypes: torch.Tensor,
    ) -> tuple:
        """
        Fusion forward pass using self-attention.
        
        Following PROOF paper implementation:
        - Expand text, prototypes, prompts to batch size
        - Apply self-attention on batch
        - Average expanded outputs to get fused text features and prototypes
        
        Args:
            image_features: Image features from current batch [batch_size, feature_dim]
            text_features: Text features for all classes [num_classes, feature_dim]
            image_prototypes: Image prototypes for seen classes [num_seen_classes, feature_dim]
        
        Returns:
            fused_image_features: Fused image features [batch_size, feature_dim]
            fused_text_features: Fused text features [num_classes, feature_dim]
            fused_prototypes: Fused prototypes [num_seen_classes, feature_dim]
        """
        batch_size = image_features.shape[0]
        num_classes = text_features.shape[0]
        
        # Collect all context prompts from all tasks (flatten)
        all_context_prompts = torch.cat([p for p in self.context_prompts], dim=0)  # [num_context, D]
        
        # Expand text features, prototypes, and prompts to batch size
        # This follows the original PROOF implementation
        text_features_expanded = text_features.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_classes, D]
        prototypes_expanded = image_prototypes.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_prototypes, D]
        prompts_expanded = all_context_prompts.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_context, D]
        
        # Concatenate along sequence dimension
        # Shape: [B, 1 + num_classes + num_prototypes + num_context, D]
        all_tokens = torch.cat([
            image_features.unsqueeze(1),   # [B, 1, D]
            text_features_expanded,        # [B, num_classes, D]
            prototypes_expanded,           # [B, num_prototypes, D]
            prompts_expanded,              # [B, num_context, D]
        ], dim=1)
        
        # Apply self-attention (single-head as in PROOF)
        # Reshape for attention: [B*N, D] where N = sequence length
        B, N, D = all_tokens.shape
        tokens_flat = all_tokens.view(B * N, D)
        
        Q = self.q_proj(tokens_flat).view(B, N, D)  # [B, N, D]
        K = self.k_proj(tokens_flat).view(B, N, D)  # [B, N, D]
        V = self.v_proj(tokens_flat).view(B, N, D)  # [B, N, D]
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.feature_dim ** 0.5)  # [B, N, N]
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [B, N, D]
        
        # Output projection
        output_flat = attn_output.view(B * N, D)
        output = self.out_proj(output_flat).view(B, N, D)  # [B, N, D]
        
        # Extract fused features
        # Image features: first token for each batch item
        fused_image_features = output[:, 0, :]  # [B, D]
        
        # Text features: average over batch dimension for text tokens
        # Indices: 1 to 1+num_classes
        fused_text_features = output[:, 1:1+num_classes, :].mean(dim=0)  # [num_classes, D]
        
        # Prototype features: average over batch dimension for prototype tokens
        # Indices: 1+num_classes to 1+num_classes+num_prototypes
        num_prototypes = image_prototypes.shape[0]
        fused_prototypes = output[:, 1+num_classes:1+num_classes+num_prototypes, :].mean(dim=0)  # [num_prototypes, D]
        
        return fused_image_features, fused_text_features, fused_prototypes
    
    def num_context_prompts_total(self) -> int:
        """Return the total number of context prompts across all tasks."""
        return len(self.context_prompts) * self.num_context_prompts


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
        original_model: Optional[nn.Module] = None,
    ):
        """
        Args:
            backbone: Pretrained backbone (will be frozen)
            classifier: Classifier head
            projection: Projection layer to insert
            freeze_backbone: Whether to freeze backbone parameters
            original_model: Optional reference to original model (for attribute forwarding)
        """
        super().__init__()
        self.backbone = backbone
        self.projection = projection
        self.classifier = classifier
        self._original_model = original_model  # Store with _ prefix to avoid conflicts
        
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
    
    def forward_for_pretraining_loss(
        self, x: torch.Tensor, targets: torch.Tensor
    ):
        """
        Forward pass for pretraining loss computation with projection.
        
        This method ensures the projection layer is applied to image features
        before they are used in the pretraining loss computation.
        
        Args:
            x: Input images [batch_size, C, H, W]
            targets: Target class labels [batch_size]
            
        Returns:
            Tuple of (normalized_image_features, normalized_text_features)
        """
        # Extract image features from backbone
        image_features = self.backbone(x)
        
        # Apply projection to image features
        projected_features = self.projection(image_features)
        
        # Pass projected features to classifier's forward_for_pretraining_loss
        return self.classifier.forward_for_pretraining_loss(projected_features, targets)
    
    def __getattr__(self, name: str):
        """Forward attribute access to base module for compatibility."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._original_model, name)


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
    expandable = projection_config.get("expandable", False)
    fusion_method = projection_config.get("fusion_method", "add")
    learnable_fusion = projection_config.get("learnable_fusion", False)
    freeze_previous = projection_config.get("freeze_previous", True)
    use_proof_fusion = projection_config.get("use_proof_fusion", False)
    num_context_prompts = projection_config.get("num_context_prompts", 4)
    
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
            if expandable:
                vision_projection = ExpandableProjection(
                    input_dim=feature_dim,
                    output_dim=feature_dim,
                    projection_type=projection_type,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                    activation=activation,
                    init_scale=init_scale,
                    fusion_method=fusion_method,
                    learnable_fusion=learnable_fusion,
                    freeze_previous=freeze_previous,
                )
                print(f"  Expandable vision projection (fusion: {fusion_method})")
            else:
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
            if expandable:
                text_projection = ExpandableProjection(
                    input_dim=feature_dim,
                    output_dim=feature_dim,
                    projection_type=projection_type,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                    activation=activation,
                    init_scale=init_scale,
                    fusion_method=fusion_method,
                    learnable_fusion=learnable_fusion,
                    freeze_previous=freeze_previous,
                )
                print(f"  Expandable text projection (fusion: {fusion_method})")
            else:
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
        
        # Create PROOF fusion layer if requested
        fusion_layer = None
        if use_proof_fusion:
            print("\nApplying PROOF fusion layer...")
            
            # Validate PROOF fusion requirements
            if not hasattr(model.classifier, "mode") or model.classifier.mode != "hybrid":
                raise ValueError(
                    "PROOF fusion requires classifier.mode='hybrid'. "
                    "PROOF needs real prototypes from a prototypical classifier, "
                    "which are only available in hybrid mode. "
                    "Set classifier.mode='hybrid' and use hybrid_weight=1.0 for text-only predictions."
                )
            
            if not hasattr(model.classifier, "learned_classifier") or model.classifier.learned_classifier is None:
                raise ValueError(
                    "PROOF fusion requires a learned_classifier in hybrid mode. "
                    "The learned_classifier should be a prototypical classifier with 'prototypes' attribute."
                )
            
            fusion_layer = ProofFusionLayer(
                feature_dim=feature_dim,
                num_context_prompts=num_context_prompts,
                freeze_previous=freeze_previous,
            )
            fusion_params = sum(p.numel() for p in fusion_layer.parameters())
            total_params += fusion_params
            print(f"  PROOF fusion parameters: {fusion_params:,}")
            print(f"  Context prompts per task: {num_context_prompts}")
            print("  ⚠️  PROOF requires hybrid mode with prototypical classifier")
            print("  ⚠️  Set hybrid_weight=1.0 for text-only predictions")
            
            # Add fusion layer to classifier
            model.classifier.proof_fusion = fusion_layer
        
        print(f"\nTotal projection parameters: {total_params:,}")
        print("="*60 + "\n")
        
        # Add vision projection to classifier (for PROOF fusion to use)
        # This must be done before wrapping so the classifier can project prototypes
        if adapt_vision and vision_projection is not None:
            model.classifier.vision_projection = vision_projection
        
        # Wrap with vision projection if requested
        if adapt_vision:
            wrapped_model = ProjectionWrapper(
                backbone=model.backbone,
                classifier=model.classifier,
                projection=vision_projection,
                freeze_backbone=True,
                original_model=model,
            )
            
            return wrapped_model
        else:
            # Only text projection, no vision projection
            return model
    
    else:
        # Standard single-encoder model
        if expandable:
            projection = ExpandableProjection(
                input_dim=feature_dim,
                output_dim=feature_dim,
                projection_type=projection_type,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                activation=activation,
                init_scale=init_scale,
                fusion_method=fusion_method,
                learnable_fusion=learnable_fusion,
                freeze_previous=freeze_previous,
            )
        else:
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
        if expandable:
            print(f"  Expandable: True (fusion: {fusion_method})")
        if projection_type == "mlp":
            print(f"  Hidden dim: {hidden_dim or feature_dim}")
            print(f"  Num layers: {num_layers}")
        print(f"  Dropout: {dropout}")
        print(f"  Activation: {activation}")
        print(f"  Init scale: {init_scale}")
        print(f"\nProjection Parameters: {num_params:,}")
        print("="*60 + "\n")
        
        # Wrap model with projection, passing original model for attribute forwarding
        wrapped_model = ProjectionWrapper(
            backbone=model.backbone,
            classifier=model.classifier,
            projection=projection,
            freeze_backbone=True,
            original_model=model,
        )
        
        return wrapped_model
