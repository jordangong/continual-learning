"""
Adapter tuning implementation for parameter-efficient fine-tuning.

Based on the papers:
- "Parameter-Efficient Transfer Learning for NLP" (Houlsby et al., 2019)
- "AdapterFusion: Non-Destructive Task Composition for Transfer Learning" (Pfeiffer et al., 2020)

Adapters are small bottleneck modules inserted into transformer layers that adapt
pretrained models while keeping the original weights frozen.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class AdapterLayer(nn.Module):
    """
    Bottleneck adapter module.
    
    Architecture:
        input (d) -> down_project (r) -> activation -> up_project (d) -> residual + input
    
    where r << d (typically r = 64 or 128 for d = 768)
    """
    
    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int = 64,
        dropout: float = 0.1,
        init_scale: float = 1e-3,
        activation: str = "gelu",
    ):
        """
        Args:
            input_dim: Dimension of input features
            bottleneck_dim: Dimension of bottleneck (adapter capacity)
            dropout: Dropout probability
            init_scale: Scale for weight initialization (small for near-identity init)
            activation: Activation function name
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.init_scale = init_scale
        
        # Down projection
        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        
        # Activation
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Up projection
        self.up_project = nn.Linear(bottleneck_dim, input_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize with small weights for near-identity initialization
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize adapter weights to near-identity."""
        # Initialize down projection with small values
        nn.init.normal_(self.down_project.weight, std=self.init_scale)
        nn.init.zeros_(self.down_project.bias)
        
        # Initialize up projection with even smaller values (near zero)
        nn.init.normal_(self.up_project.weight, std=self.init_scale * 0.1)
        nn.init.zeros_(self.up_project.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through adapter with residual connection.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim) or (batch, input_dim)
            
        Returns:
            Output with same shape as input
        """
        # Adapter bottleneck
        h = self.down_project(x)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.up_project(h)
        
        # Residual connection
        return x + h


class AdapterWrapper(nn.Module):
    """
    Wrapper that adds adapter after a module (typically attention or MLP).
    
    This wraps an existing module and inserts an adapter after it.
    
    Note: The base module is NOT frozen here - freezing should be done at the model level
    to ensure all base parameters are frozen, not just adapter-wrapped ones.
    """
    
    def __init__(
        self,
        base_module: nn.Module,
        adapter: AdapterLayer,
    ):
        """
        Args:
            base_module: Original module to wrap
            adapter: Adapter layer to insert after base module
        """
        super().__init__()
        self.base_module = base_module
        self.adapter = adapter
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass through base module then adapter."""
        output = self.base_module(*args, **kwargs)
        
        # Handle different output types (some modules return tuples)
        if isinstance(output, tuple):
            # Apply adapter to first element (typically the main output)
            adapted_output = self.adapter(output[0])
            return (adapted_output,) + output[1:]
        else:
            return self.adapter(output)
    
    def __getattr__(self, name: str):
        """Forward attribute access to base module for compatibility."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            # Forward to base module if not found in wrapper
            return getattr(self.base_module, name)


class AdapterModel(nn.Module):
    """
    Wrapper that applies adapters to a pretrained model.
    
    This automatically identifies transformer blocks and inserts adapters
    after attention and/or MLP layers based on configuration.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        adapter_config: Dict[str, Any],
        model_source: str = "timm",
    ):
        """
        Args:
            base_model: Pretrained model to adapt
            adapter_config: Adapter configuration dictionary
            model_source: Model source ("timm", "openclip") for architecture detection
        """
        super().__init__()
        self.base_model = base_model
        self.model_source = model_source
        
        # Extract configuration
        self.bottleneck_dim = adapter_config.get("bottleneck_dim", 64)
        self.dropout = adapter_config.get("dropout", 0.1)
        self.init_scale = adapter_config.get("init_scale", 1e-3)
        self.activation = adapter_config.get("activation", "gelu")
        self.adapter_after_attn = adapter_config.get("adapter_after_attn", False)
        self.adapter_after_mlp = adapter_config.get("adapter_after_mlp", True)
        
        # Track adapted modules
        self.adapter_modules = []
        
        # Apply adapters
        self._apply_adapters()
    
    def _detect_hidden_dim(self, module: nn.Module) -> Optional[int]:
        """
        Detect hidden dimension from a module.
        
        Args:
            module: Module to inspect
            
        Returns:
            Hidden dimension or None if not found
        """
        # Try to find the LAST Linear layer (which determines output dimension)
        last_linear = None
        for child in module.modules():
            if isinstance(child, nn.Linear):
                last_linear = child
        
        if last_linear is not None:
            return last_linear.out_features
        return None
    
    def _apply_adapters(self):
        """Apply adapters to transformer blocks in the model."""
        if self.model_source == "timm":
            self._apply_adapters_timm()
        elif self.model_source == "openclip":
            self._apply_adapters_openclip()
        else:
            raise ValueError(f"Unknown model source: {self.model_source}")
        
        print(f"\nApplied {len(self.adapter_modules)} adapters to the model")
    
    def _apply_adapters_timm(self):
        """Apply adapters to timm ViT models."""
        # Timm ViT structure: model.blocks[i].{attn, mlp}
        if not hasattr(self.base_model, "blocks"):
            print("Warning: Model doesn't have 'blocks' attribute, skipping adapter insertion")
            return
        
        for block_idx, block in enumerate(self.base_model.blocks):
            hidden_dim = None
            
            # Adapter after attention
            if self.adapter_after_attn and hasattr(block, "attn"):
                if hidden_dim is None:
                    hidden_dim = self._detect_hidden_dim(block.attn)
                
                if hidden_dim is not None:
                    adapter = AdapterLayer(
                        input_dim=hidden_dim,
                        bottleneck_dim=self.bottleneck_dim,
                        dropout=self.dropout,
                        init_scale=self.init_scale,
                        activation=self.activation,
                    )
                    block.attn = AdapterWrapper(block.attn, adapter)
                    self.adapter_modules.append(f"blocks.{block_idx}.attn")
            
            # Adapter after MLP
            if self.adapter_after_mlp and hasattr(block, "mlp"):
                if hidden_dim is None:
                    hidden_dim = self._detect_hidden_dim(block.mlp)
                
                if hidden_dim is not None:
                    adapter = AdapterLayer(
                        input_dim=hidden_dim,
                        bottleneck_dim=self.bottleneck_dim,
                        dropout=self.dropout,
                        init_scale=self.init_scale,
                        activation=self.activation,
                    )
                    block.mlp = AdapterWrapper(block.mlp, adapter)
                    self.adapter_modules.append(f"blocks.{block_idx}.mlp")
    
    def _apply_adapters_openclip(self):
        """Apply adapters to OpenCLIP models (ResidualAttentionBlock)."""
        # OpenCLIP structure: transformer.resblocks[i].{attn, mlp}
        
        # Check if this is a FULL CLIP model with BOTH vision and text encoders
        # (not just a visual encoder that happens to have a 'transformer' attribute)
        is_full_clip = (
            hasattr(self.base_model, "visual") and 
            hasattr(self.base_model, "transformer") and
            hasattr(self.base_model, "encode_image") and
            hasattr(self.base_model, "encode_text")
        )
        
        if is_full_clip:
            # Full CLIP model - need to check config for which encoders to adapt
            # This will be handled by the model factory
            raise NotImplementedError("Full CLIP model adaptation should be handled by model factory")
        
        # Single transformer (either vision or text encoder)
        # Visual encoder: has 'transformer' attribute with 'resblocks'
        # Text transformer: directly has 'resblocks'
        if hasattr(self.base_model, "resblocks"):
            resblocks = self.base_model.resblocks
        elif hasattr(self.base_model, "transformer") and hasattr(self.base_model.transformer, "resblocks"):
            resblocks = self.base_model.transformer.resblocks
        else:
            print("Warning: Cannot find resblocks in OpenCLIP model")
            return
        
        for block_idx, block in enumerate(resblocks):
            hidden_dim = None
            
            # Adapter after attention
            if self.adapter_after_attn and hasattr(block, "attn"):
                # Try to get dimension from attention output projection
                if hasattr(block.attn, "out_proj"):
                    hidden_dim = block.attn.out_proj.out_features
                else:
                    # Fallback: try general dimension detection
                    hidden_dim = self._detect_hidden_dim(block.attn)
                
                if hidden_dim is not None:
                    adapter = AdapterLayer(
                        input_dim=hidden_dim,
                        bottleneck_dim=self.bottleneck_dim,
                        dropout=self.dropout,
                        init_scale=self.init_scale,
                        activation=self.activation,
                    )
                    block.attn = AdapterWrapper(block.attn, adapter)
                    self.adapter_modules.append(f"resblocks.{block_idx}.attn")
            
            # Adapter after MLP
            if self.adapter_after_mlp and hasattr(block, "mlp"):
                if hidden_dim is None:
                    hidden_dim = self._detect_hidden_dim(block.mlp)
                
                if hidden_dim is not None:
                    adapter = AdapterLayer(
                        input_dim=hidden_dim,
                        bottleneck_dim=self.bottleneck_dim,
                        dropout=self.dropout,
                        init_scale=self.init_scale,
                        activation=self.activation,
                    )
                    block.mlp = AdapterWrapper(block.mlp, adapter)
                    self.adapter_modules.append(f"resblocks.{block_idx}.mlp")
    
    def forward(self, *args, **kwargs):
        """Forward pass through the adapted model."""
        return self.base_model(*args, **kwargs)
    
    def get_adapter_parameters(self):
        """Get all adapter parameters for optimization."""
        adapter_params = []
        for module in self.base_model.modules():
            if isinstance(module, AdapterLayer):
                adapter_params.extend(module.parameters())
        return adapter_params
    
    def print_adapter_info(self):
        """Print information about adapter configuration."""
        total_params = sum(p.numel() for p in self.base_model.parameters())
        adapter_params = sum(p.numel() for p in self.get_adapter_parameters())
        trainable_params = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        
        print("\n" + "="*60)
        print("Adapter Configuration:")
        print(f"  Bottleneck dim: {self.bottleneck_dim}")
        print(f"  Dropout: {self.dropout}")
        print(f"  Init scale: {self.init_scale}")
        print(f"  Activation: {self.activation}")
        print(f"  After attention: {self.adapter_after_attn}")
        print(f"  After MLP: {self.adapter_after_mlp}")
        print("\nParameter Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Adapter parameters: {adapter_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")
        
        print(f"\nModules with adapters: {len(self.adapter_modules)}")
        for module_name in sorted(self.adapter_modules):
            print(f"  - {module_name}")
        print("="*60 + "\n")
    
    # Forward common methods to base model
    def __getattr__(self, name: str):
        """Forward attribute access to base model if not found in AdapterModel."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)


def create_adapter_model(
    base_model: nn.Module,
    adapter_config: Dict[str, Any],
    model_source: str = "timm",
) -> AdapterModel:
    """
    Create an adapter-augmented model.
    
    Args:
        base_model: Pretrained model to adapt
        adapter_config: Adapter configuration dictionary
        model_source: Model source for architecture detection
        
    Returns:
        AdapterModel wrapping the base model with adapters
    """
    adapter_model = AdapterModel(
        base_model=base_model,
        adapter_config=adapter_config,
        model_source=model_source,
    )
    
    # Print configuration
    adapter_model.print_adapter_info()
    
    return adapter_model
