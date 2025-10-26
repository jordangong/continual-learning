"""
Low-Rank Adaptation (LoRA) implementation for Parameter-Efficient Fine-Tuning.

Based on the paper:
"LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
https://arxiv.org/abs/2106.09685

This implementation provides a generic LoRA wrapper that can be applied to various 
neural network architectures including Vision Transformers, ResNets, and CLIP models.
"""

import math
from typing import Any, Dict, List

import torch
import torch.nn as nn


class LoRAParameter(nn.Module):
    """
    LoRA adaptation for nn.Parameter objects (e.g., weights in nn.MultiheadAttention).
    
    This wraps a frozen Parameter with low-rank adaptation:
        output = frozen_param @ input + (lora_B @ lora_A) @ input * scaling
    """
    
    def __init__(
        self,
        base_param: nn.Parameter,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        """
        Args:
            base_param: Original parameter to adapt [out_features, in_features]
            rank: Rank of LoRA decomposition
            alpha: LoRA scaling factor
            dropout: Dropout probability
        """
        super().__init__()
        
        out_features, in_features = base_param.shape
        self.out_features = out_features
        self.in_features = in_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout_prob = dropout
        
        # Register frozen base parameter
        self.register_buffer('base_param', base_param.data.clone())
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()
        
        # Initialize
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize LoRA parameters."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA-adapted parameter to input.
        
        Args:
            x: Input tensor [..., in_features]
        
        Returns:
            Output tensor [..., out_features]
        """
        # Base transformation
        output = torch.nn.functional.linear(x, self.base_param, None)
        
        # LoRA contribution
        lora_out = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t() * self.scaling
        
        return output + lora_out


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


class LoRAMultiheadAttention(nn.Module):
    """
    Wrapper for nn.MultiheadAttention with LoRA adaptation on Q, K, V projections.
    
    This splits the fused in_proj_weight into separate Q, K, V projections and wraps
    them with LoRAParameter. The forward pass manually applies the LoRA-adapted
    projections before calling the attention mechanism.
    """
    
    def __init__(
        self,
        base_attn: nn.MultiheadAttention,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        adapt_q: bool = True,
        adapt_k: bool = True,
        adapt_v: bool = True,
        adapt_out: bool = True,
    ):
        """
        Args:
            base_attn: Base nn.MultiheadAttention module
            rank: Rank of LoRA decomposition
            alpha: LoRA scaling factor
            dropout: Dropout probability
            adapt_q: Whether to adapt Query projection
            adapt_k: Whether to adapt Key projection
            adapt_v: Whether to adapt Value projection
            adapt_out: Whether to adapt output projection
        """
        super().__init__()
        
        self.base_attn = base_attn
        self.embed_dim = base_attn.embed_dim
        self.num_heads = base_attn.num_heads
        self.head_dim = base_attn.head_dim
        self.adapt_q = adapt_q
        self.adapt_k = adapt_k
        self.adapt_v = adapt_v
        self.adapt_out = adapt_out
        
        # Check if this is a fused QKV attention
        if not hasattr(base_attn, 'in_proj_weight') or base_attn.in_proj_weight is None:
            raise ValueError("Base attention must have fused in_proj_weight")
        
        # Split the fused in_proj_weight
        in_proj_weight = base_attn.in_proj_weight.data
        assert in_proj_weight.shape[0] == 3 * self.embed_dim
        
        q_weight = in_proj_weight[:self.embed_dim, :]
        k_weight = in_proj_weight[self.embed_dim:2*self.embed_dim, :]
        v_weight = in_proj_weight[2*self.embed_dim:, :]
        
        # Split bias if present
        if base_attn.in_proj_bias is not None:
            in_proj_bias = base_attn.in_proj_bias.data
            q_bias = in_proj_bias[:self.embed_dim]
            k_bias = in_proj_bias[self.embed_dim:2*self.embed_dim]
            v_bias = in_proj_bias[2*self.embed_dim:]
        else:
            q_bias = k_bias = v_bias = None
        
        # Create LoRAParameter or frozen parameters for Q, K, V
        if adapt_q:
            self.q_proj = LoRAParameter(nn.Parameter(q_weight.clone()), rank, alpha, dropout)
        else:
            self.register_buffer('q_weight', q_weight.clone())
        
        if adapt_k:
            self.k_proj = LoRAParameter(nn.Parameter(k_weight.clone()), rank, alpha, dropout)
        else:
            self.register_buffer('k_weight', k_weight.clone())
        
        if adapt_v:
            self.v_proj = LoRAParameter(nn.Parameter(v_weight.clone()), rank, alpha, dropout)
        else:
            self.register_buffer('v_weight', v_weight.clone())
        
        # Register biases as buffers (frozen)
        if q_bias is not None:
            self.register_buffer('q_bias', q_bias.clone())
            self.register_buffer('k_bias', k_bias.clone())
            self.register_buffer('v_bias', v_bias.clone())
        else:
            self.q_bias = self.k_bias = self.v_bias = None
        
        # Wrap out_proj with LoRALinear if adapt_out is True
        if adapt_out and isinstance(base_attn.out_proj, nn.Linear):
            self.out_proj = LoRALinear(
                base_layer=base_attn.out_proj,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                bias="none",
            )
        else:
            self.out_proj = base_attn.out_proj
        
        # Copy other attributes from base attention
        self.batch_first = base_attn.batch_first
        self.dropout = base_attn.dropout
        self.add_zero_attn = base_attn.add_zero_attn
        self.bias_k = base_attn.bias_k
        self.bias_v = base_attn.bias_v
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
        need_weights: bool = True,
        attn_mask: torch.Tensor = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        """Forward pass with LoRA-adapted QKV projections."""
        # Apply Q, K, V projections (with or without LoRA)
        if self.adapt_q:
            q = self.q_proj(query)
            if self.q_bias is not None:
                q = q + self.q_bias
        else:
            q = torch.nn.functional.linear(query, self.q_weight, self.q_bias)
        
        if self.adapt_k:
            k = self.k_proj(key)
            if self.k_bias is not None:
                k = k + self.k_bias
        else:
            k = torch.nn.functional.linear(key, self.k_weight, self.k_bias)
        
        if self.adapt_v:
            v = self.v_proj(value)
            if self.v_bias is not None:
                v = v + self.v_bias
        else:
            v = torch.nn.functional.linear(value, self.v_weight, self.v_bias)
        
        # Reshape for multi-head attention
        if self.batch_first:
            # Input: [batch, seq_len, embed_dim]
            batch_size, seq_len, _ = query.shape
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            # Input: [seq_len, batch, embed_dim]
            seq_len, batch_size, _ = query.shape
            q = q.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
            k = k.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
            v = v.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
        
        # Compute attention scores
        # q, k, v: [batch, num_heads, seq_len, head_dim]
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [batch, num_heads, seq_len, seq_len]
        
        # Apply attention mask if provided
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        
        # Apply causal mask if requested
        if is_causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # Softmax to get attention weights
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        
        # Apply dropout if training
        if self.training and self.dropout > 0:
            attn_weights = torch.nn.functional.dropout(attn_weights, p=self.dropout)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [batch, num_heads, seq_len, head_dim]
        
        # Reshape back
        if self.batch_first:
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        else:
            attn_output = attn_output.transpose(1, 2).transpose(0, 1).contiguous().view(seq_len, batch_size, self.embed_dim)
        
        # Apply output projection (with LoRA)
        attn_output = self.out_proj(attn_output)
        
        if need_weights:
            # Average attention weights over heads if requested
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)
            return attn_output, attn_weights
        else:
            return attn_output, None


def adapt_multihead_attention_with_lora(
    attn_module: nn.MultiheadAttention,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    adapt_q: bool = True,
    adapt_k: bool = True,
    adapt_v: bool = True,
    adapt_out: bool = True,
) -> nn.Module:
    """
    Adapt nn.MultiheadAttention with LoRA by creating a wrapper module.
    
    For CLIP models where Q, K, V have the same dimensions, nn.MultiheadAttention 
    uses a fused in_proj_weight parameter of shape [3*embed_dim, embed_dim].
    This function:
    1. Splits in_proj_weight into separate Q, K, V projections
    2. Wraps selected projections with LoRAParameter for low-rank adaptation
    3. Returns a LoRAMultiheadAttention wrapper that handles the forward pass
    
    Args:
        attn_module: nn.MultiheadAttention module to adapt
        rank: Rank of LoRA decomposition
        alpha: LoRA scaling factor
        dropout: Dropout probability
        adapt_q: Whether to adapt Query projection
        adapt_k: Whether to adapt Key projection
        adapt_v: Whether to adapt Value projection
        adapt_out: Whether to adapt output projection
    
    Returns:
        LoRAMultiheadAttention wrapper module
    
    Note:
        Returns a new module that wraps the original attention module.
        Individual Q, K, V, and out_proj can be selectively adapted.
    
    Example:
        # Adapt only Q and V (as in some LoRA papers)
        lora_attn = adapt_multihead_attention_with_lora(
            attn, adapt_q=True, adapt_k=False, adapt_v=True, adapt_out=True
        )
    """
    # Check if this is a fused QKV attention
    if not hasattr(attn_module, 'in_proj_weight') or attn_module.in_proj_weight is None:
        # Already using separate projections - not supported yet
        print("Warning: MultiheadAttention doesn't have fused in_proj_weight, returning original module")
        return attn_module
    
    # Create and return wrapper
    return LoRAMultiheadAttention(
        base_attn=attn_module,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        adapt_q=adapt_q,
        adapt_k=adapt_k,
        adapt_v=adapt_v,
        adapt_out=adapt_out,
    )


class LoRAModel(nn.Module):
    """
    Wrapper that applies LoRA to a pretrained model.
    
    This automatically identifies target modules and replaces them with LoRA-adapted versions.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        lora_config: dict,
    ):
        """
        Args:
            base_model: Pretrained model to adapt
            lora_config: LoRA configuration with keys:
                - rank: LoRA rank
                - alpha: LoRA alpha scaling
                - dropout: Dropout probability
                - target_modules: List of module name patterns to apply LoRA
                - merge_weights: Whether to merge LoRA weights
                - bias: Bias handling ("none", "all", "lora_only")
                - fan_in_fan_out: Whether to use fan_in_fan_out convention
                - multihead_attention: Dict with adapt_q, adapt_k, adapt_v, adapt_out flags
        """
        super().__init__()
        
        self.base_model = base_model
        self.rank = lora_config.get("rank", 8)
        self.alpha = lora_config.get("alpha", 16.0)
        self.dropout = lora_config.get("dropout", 0.0)
        self.target_modules = lora_config.get("target_modules", [])
        self.merge_weights = lora_config.get("merge_weights", False)
        self.bias = lora_config.get("bias", "none")
        self.fan_in_fan_out = lora_config.get("fan_in_fan_out", False)
        
        # MultiheadAttention specific flags
        mha_config = lora_config.get("multihead_attention", {})
        self.adapt_q = mha_config.get("adapt_q", True)
        self.adapt_k = mha_config.get("adapt_k", True)
        self.adapt_v = mha_config.get("adapt_v", True)
        self.adapt_out = mha_config.get("adapt_out", True)
        
        # Track adapted modules
        self.lora_modules = set()
        
        # Apply LoRA to matching modules
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
            Applies to:
            - nn.Linear and its subclasses (e.g., NonDynamicallyQuantizableLinear)
            - nn.MultiheadAttention (QKV projections split and adapted)
        """
        # Check for MultiheadAttention
        if isinstance(module, nn.MultiheadAttention):
            for pattern in self.target_modules:
                if pattern in name:
                    return True
        
        # Check for Linear layers
        if isinstance(module, nn.Linear):
            for pattern in self.target_modules:
                if pattern in name:
                    return True
        
        return False
    
    def _apply_lora(self):
        """Apply LoRA to all matching modules in the model."""
        # Recursively replace matching modules with LoRA versions
        self._replace_modules_with_lora(self.base_model, "")
    
    def _replace_modules_with_lora(self, module: nn.Module, prefix: str = ""):
        """
        Recursively replace Linear and MultiheadAttention modules with LoRA-adapted versions.
        
        Args:
            module: Current module to process
            prefix: Name prefix for tracking module path
        """
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if self._should_apply_lora(full_name, child):
                if isinstance(child, nn.MultiheadAttention):
                    # Adapt MultiheadAttention by replacing with wrapper
                    # Use individual adaptation flags from config
                    lora_attn = adapt_multihead_attention_with_lora(
                        attn_module=child,
                        rank=self.rank,
                        alpha=self.alpha,
                        dropout=self.dropout,
                        adapt_q=self.adapt_q,
                        adapt_k=self.adapt_k,
                        adapt_v=self.adapt_v,
                        adapt_out=self.adapt_out,
                    )
                    setattr(module, name, lora_attn)
                    self.lora_modules.add(full_name)
                elif isinstance(child, nn.Linear):
                    # Replace Linear with LoRALinear
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
                self._replace_modules_with_lora(child, full_name)
    
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
        seen_modules = set()  # Track modules we've already processed
        
        for module in self.base_model.modules():
            # Skip if we've already processed this module
            if id(module) in seen_modules:
                continue
            
            if isinstance(module, LoRAMultiheadAttention):
                # Collect LoRA params from MultiheadAttention wrapper
                # Mark this module as seen
                seen_modules.add(id(module))
                
                if module.adapt_q and hasattr(module, 'q_proj') and isinstance(module.q_proj, LoRAParameter):
                    seen_modules.add(id(module.q_proj))  # Mark as seen
                    lora_params.extend([module.q_proj.lora_A, module.q_proj.lora_B])
                if module.adapt_k and hasattr(module, 'k_proj') and isinstance(module.k_proj, LoRAParameter):
                    seen_modules.add(id(module.k_proj))  # Mark as seen
                    lora_params.extend([module.k_proj.lora_A, module.k_proj.lora_B])
                if module.adapt_v and hasattr(module, 'v_proj') and isinstance(module.v_proj, LoRAParameter):
                    seen_modules.add(id(module.v_proj))  # Mark as seen
                    lora_params.extend([module.v_proj.lora_A, module.v_proj.lora_B])
                
                # Collect out_proj params if it's LoRALinear
                if isinstance(module.out_proj, LoRALinear):
                    seen_modules.add(id(module.out_proj))  # Mark as seen to avoid double-counting
                    lora_params.extend([module.out_proj.lora.lora_A, module.out_proj.lora.lora_B])
                    if module.out_proj._bias is not None:
                        lora_params.append(module.out_proj._bias)
                    if module.out_proj._lora_bias is not None:
                        lora_params.append(module.out_proj._lora_bias)
            
            elif isinstance(module, LoRALinear):
                seen_modules.add(id(module))
                lora_params.extend([module.lora.lora_A, module.lora.lora_B])
                if module._bias is not None:
                    lora_params.append(module._bias)
                if module._lora_bias is not None:
                    lora_params.append(module._lora_bias)
            
            elif isinstance(module, LoRAParameter):
                seen_modules.add(id(module))
                lora_params.extend([module.lora_A, module.lora_B])
        
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
