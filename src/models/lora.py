"""
Low-Rank Adaptation (LoRA) implementation for Parameter-Efficient Fine-Tuning.

Based on the paper:
"LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
https://arxiv.org/abs/2106.09685

This implementation provides a generic LoRA wrapper that can be applied to various 
neural network architectures including Vision Transformers, ResNets, and CLIP models.
"""

import math
from typing import Any, Dict, List, Set

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """
    Base LoRA layer that wraps a linear layer with low-rank adaptation.
    
    Instead of fine-tuning W directly, we keep W frozen and learn:
        h = Wx + (B·A)x
    where A ∈ R^(r×d_in), B ∈ R^(d_out×r), and r << min(d_in, d_out)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        merge_weights: bool = False,
        fan_in_fan_out: bool = False,
    ):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: Rank of the low-rank decomposition (r in the paper)
            alpha: LoRA scaling factor (scaling is alpha/r)
            dropout: Dropout probability for LoRA layers
            merge_weights: Whether to merge LoRA weights into original weights
            fan_in_fan_out: Set to True for Conv1D layers (e.g., GPT-2 style)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout_prob = dropout
        self.merged = False
        self.merge_weights = merge_weights
        self.fan_in_fan_out = fan_in_fan_out
        
        # LoRA matrices
        # A is initialized with random Gaussian, B is initialized to zero
        # This ensures that at initialization, the LoRA path contributes nothing
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout layer
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize LoRA parameters using Kaiming uniform initialization for A."""
        # Initialize A with Kaiming uniform (similar to nn.Linear)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Initialize B to zero so that BA is zero at initialization
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LoRA layer.
        
        Args:
            x: Input tensor [batch_size, ..., in_features]
            
        Returns:
            Output tensor [batch_size, ..., out_features]
        """
        # Compute LoRA contribution: (dropout(x) @ A^T @ B^T) * scaling
        lora_output = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t() * self.scaling
        return lora_output
    
    def merge_lora_weights(self, base_weight: torch.Tensor) -> torch.Tensor:
        """
        Merge LoRA weights into base weight.
        
        Args:
            base_weight: Original weight matrix [out_features, in_features]
            
        Returns:
            Merged weight matrix
        """
        if self.merged:
            return base_weight
        
        # Compute LoRA delta: B @ A * scaling
        lora_delta = (self.lora_B @ self.lora_A) * self.scaling
        
        if self.fan_in_fan_out:
            # For Conv1D-style layers where weight is transposed
            merged_weight = base_weight + lora_delta.t()
        else:
            merged_weight = base_weight + lora_delta
        
        return merged_weight
    
    def unmerge_lora_weights(self, merged_weight: torch.Tensor) -> torch.Tensor:
        """
        Unmerge LoRA weights from merged weight.
        
        Args:
            merged_weight: Merged weight matrix
            
        Returns:
            Original base weight matrix
        """
        if not self.merged:
            return merged_weight
        
        # Compute LoRA delta: B @ A * scaling
        lora_delta = (self.lora_B @ self.lora_A) * self.scaling
        
        if self.fan_in_fan_out:
            base_weight = merged_weight - lora_delta.t()
        else:
            base_weight = merged_weight - lora_delta
        
        return base_weight


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    
    This module wraps an existing nn.Linear layer and adds LoRA low-rank adaptation.
    The original weights are frozen and only LoRA parameters are trained.
    
    Note:
        This class exposes `weight` and `bias` properties that forward to the base layer,
        ensuring compatibility with code that directly accesses these attributes
        (e.g., OpenCLIP's `get_weight_dtype()` method).
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        merge_weights: bool = False,
        fan_in_fan_out: bool = False,
        bias: str = "none",
    ):
        """
        Args:
            base_layer: Original nn.Linear layer to adapt
            rank: Rank of LoRA decomposition
            alpha: LoRA scaling factor
            dropout: Dropout probability
            merge_weights: Whether to merge weights after training
            fan_in_fan_out: Conv1D-style layer flag
            bias: Bias handling: "none", "all", or "lora_only"
        """
        super().__init__()
        
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.bias_mode = bias
        
        # Freeze the original layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
        
        # Create LoRA layer
        self.lora = LoRALayer(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            merge_weights=merge_weights,
            fan_in_fan_out=fan_in_fan_out,
        )
        
        # Handle bias (use private attributes to avoid conflict with property)
        if bias == "all" and base_layer.bias is None:
            # Add trainable bias if not present
            self._bias = nn.Parameter(torch.zeros(self.out_features))
            self._lora_bias = None
        elif bias == "lora_only":
            # Add separate LoRA bias
            self._lora_bias = nn.Parameter(torch.zeros(self.out_features))
            self._bias = None
        else:
            # Use existing bias or no bias
            self._bias = None
            self._lora_bias = None
    
    @property
    def weight(self):
        """Forward weight access to base layer for compatibility."""
        return self.base_layer.weight
    
    @property
    def bias(self):
        """Forward bias access to base layer for compatibility."""
        # Check if we have a custom bias attribute
        if hasattr(self, '_bias') and self._bias is not None:
            return self._bias
        elif hasattr(self, '_lora_bias') and self._lora_bias is not None:
            return self._lora_bias
        return self.base_layer.bias
    
    @bias.setter
    def bias(self, value):
        """Allow setting bias for initialization."""
        if value is None:
            self._bias = None
        else:
            self._bias = value
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA-adapted linear layer."""
        # Base layer output
        output = self.base_layer(x)
        
        # Add LoRA contribution
        output = output + self.lora(x)
        
        # Add additional bias if specified
        if self._bias is not None:
            output = output + self._bias
        elif self._lora_bias is not None:
            output = output + self._lora_bias
        
        return output
    
    def merge_weights(self):
        """Merge LoRA weights into base layer."""
        if self.lora.merged:
            return
        
        # Merge LoRA weights into base layer
        merged_weight = self.lora.merge_lora_weights(self.base_layer.weight.data)
        self.base_layer.weight.data = merged_weight
        self.lora.merged = True
    
    def unmerge_weights(self):
        """Unmerge LoRA weights from base layer."""
        if not self.lora.merged:
            return
        
        # Unmerge LoRA weights from base layer
        base_weight = self.lora.unmerge_lora_weights(self.base_layer.weight.data)
        self.base_layer.weight.data = base_weight
        self.lora.merged = False


class LoRAModel(nn.Module):
    """
    Wrapper that applies LoRA to a pretrained model.
    
    This automatically identifies target modules and replaces them with LoRA-adapted versions.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        lora_config: Dict[str, Any],
    ):
        """
        Args:
            base_model: The base pretrained model to adapt
            lora_config: LoRA configuration dictionary containing:
                - rank: Rank of LoRA matrices
                - alpha: LoRA scaling factor
                - dropout: Dropout probability
                - target_modules: List of module name patterns to apply LoRA to
                - merge_weights: Whether to merge weights after training
                - bias: Bias handling mode
                - fan_in_fan_out: Conv1D-style flag
        """
        super().__init__()
        
        self.base_model = base_model
        self.lora_config = lora_config
        
        # Extract configuration
        self.rank = lora_config.get("rank", 8)
        self.alpha = lora_config.get("alpha", 16.0)
        self.dropout = lora_config.get("dropout", 0.0)
        self.target_modules = lora_config.get("target_modules", ["qkv", "proj", "fc1", "fc2"])
        self.merge_weights = lora_config.get("merge_weights", False)
        self.bias = lora_config.get("bias", "none")
        self.fan_in_fan_out = lora_config.get("fan_in_fan_out", False)
        
        # Track which modules have LoRA applied (initialize before _apply_lora)
        self.lora_modules: Set[str] = set()
        
        # Apply LoRA to target modules
        self._apply_lora()
    
    def _should_apply_lora(self, name: str, module: nn.Module) -> bool:
        """
        Check if LoRA should be applied to this module based on target patterns.
        
        Args:
            name: Module name
            module: Module instance
            
        Returns:
            True if LoRA should be applied
        
        Note:
            Only applies to nn.Linear and its subclasses (e.g., NonDynamicallyQuantizableLinear).
            Does NOT apply to nn.Parameter objects like in_proj_weight in nn.MultiheadAttention.
        """
        if not isinstance(module, nn.Linear):
            return False
        
        # Check if name matches any target pattern
        for pattern in self.target_modules:
            if pattern in name:
                return True
        
        return False
    
    def _apply_lora(self):
        """Apply LoRA to all matching modules in the model."""
        # Recursively replace matching Linear layers with LoRA versions
        self._replace_linear_with_lora(self.base_model, "")
    
    def _replace_linear_with_lora(self, module: nn.Module, prefix: str = ""):
        """
        Recursively replace Linear layers with LoRA-adapted versions.
        
        Args:
            module: Current module to process
            prefix: Name prefix for tracking module path
        """
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if self._should_apply_lora(full_name, child):
                # Replace with LoRA version
                lora_layer = LoRALinear(
                    base_layer=child,
                    rank=self.rank,
                    alpha=self.alpha,
                    dropout=self.dropout,
                    merge_weights=self.merge_weights,
                    fan_in_fan_out=self.fan_in_fan_out,
                    bias=self.bias,
                )
                setattr(module, name, lora_layer)
                self.lora_modules.add(full_name)
            else:
                # Recursively process children
                self._replace_linear_with_lora(child, full_name)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.base_model(x)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification."""
        if hasattr(self.base_model, "forward_features"):
            return self.base_model.forward_features(x)
        else:
            raise NotImplementedError("Base model does not have forward_features method")
    
    def merge_lora_weights(self):
        """Merge all LoRA weights into base model weights."""
        for module in self.base_model.modules():
            if isinstance(module, LoRALinear):
                module.merge_weights()
    
    def unmerge_lora_weights(self):
        """Unmerge all LoRA weights from base model weights."""
        for module in self.base_model.modules():
            if isinstance(module, LoRALinear):
                module.unmerge_weights()
    
    def get_lora_parameters(self) -> List[nn.Parameter]:
        """Get all LoRA parameters for optimization."""
        lora_params = []
        for module in self.base_model.modules():
            if isinstance(module, LoRALinear):
                lora_params.extend([module.lora.lora_A, module.lora.lora_B])
                if module._bias is not None:
                    lora_params.append(module._bias)
                if module._lora_bias is not None:
                    lora_params.append(module._lora_bias)
        return lora_params
    
    def print_lora_info(self):
        """Print information about LoRA adaptation."""
        total_params = sum(p.numel() for p in self.base_model.parameters())
        lora_params = sum(p.numel() for p in self.get_lora_parameters())
        trainable_params = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        
        print("\n" + "="*60)
        print("LoRA Configuration:")
        print(f"  Rank: {self.rank}")
        print(f"  Alpha: {self.alpha}")
        print(f"  Scaling: {self.alpha / self.rank}")
        print(f"  Dropout: {self.dropout}")
        print(f"  Target modules: {self.target_modules}")
        print("\nParameter Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  LoRA parameters: {lora_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")
        print(f"\nModules with LoRA applied: {len(self.lora_modules)}")
        for module_name in sorted(self.lora_modules):
            print(f"  - {module_name}")
        print("="*60 + "\n")
    
    # Forward common methods to base model
    def __getattr__(self, name: str):
        """Forward attribute access to base model if not found in LoRAModel."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)


def create_lora_model(
    base_model: nn.Module,
    lora_config: Dict[str, Any],
) -> LoRAModel:
    """
    Create a LoRA-adapted model.
    
    Args:
        base_model: Base pretrained model
        lora_config: LoRA configuration
        
    Returns:
        LoRA-adapted model
    """
    lora_model = LoRAModel(base_model=base_model, lora_config=lora_config)
    
    # Print LoRA information
    lora_model.print_lora_info()
    
    return lora_model
