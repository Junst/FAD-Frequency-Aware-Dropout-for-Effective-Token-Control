# AdaptiveDropout LoRA Module for DiffSynth-Studio
# Implements rank dropout, module dropout, and neuron dropout for LoRA training
# Based on the AdaptiveDropout method proven effective on SDXL
# Reference: https://github.com/kohya-ss/sd-scripts

import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Set
import re


class AdaptiveDropoutLoRALinear(nn.Module):
    """
    LoRA layer with AdaptiveDropout support for Linear layers.
    
    Supports three types of dropout:
    - neuron_dropout: Standard dropout on intermediate activations
    - rank_dropout: Randomly drops entire rank dimensions
    - module_dropout: Randomly skips the entire LoRA module
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank
        self.scale = self.alpha / self.rank
        
        # Dropout settings
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        
        # LoRA layers
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        # Initialize weights (same as Microsoft's LoRA)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def _ensure_same_device(self, x: torch.Tensor):
        """Ensure LoRA weights are on the same device as input."""
        if self.lora_A.weight.device != x.device:
            self.lora_A = self.lora_A.to(x.device)
            self.lora_B = self.lora_B.to(x.device)
        if self.lora_A.weight.dtype != x.dtype and x.dtype in [torch.float16, torch.bfloat16, torch.float32]:
            # Only cast if trainable (requires_grad) to preserve training dtype
            if not self.lora_A.weight.requires_grad:
                self.lora_A = self.lora_A.to(x.dtype)
                self.lora_B = self.lora_B.to(x.dtype)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure LoRA weights are on the same device/dtype as input (for offloading compatibility)
        self._ensure_same_device(x)
        
        # Module dropout: skip entire LoRA
        if self.module_dropout is not None and self.training:
            if torch.rand(1, device=x.device) < self.module_dropout:
                return torch.zeros(
                    x.shape[:-1] + (self.out_features,),
                    dtype=x.dtype,
                    device=x.device
                )
        
        # Down projection
        lx = self.lora_A(x)
        
        # Neuron dropout
        if self.dropout is not None and self.training:
            lx = nn.functional.dropout(lx, p=self.dropout)
        
        # Rank dropout
        if self.rank_dropout is not None and self.training:
            mask = torch.rand((lx.size(0), self.rank), device=lx.device) > self.rank_dropout
            if len(lx.size()) == 3:
                mask = mask.unsqueeze(1)  # For sequence data [B, L, R]
            elif len(lx.size()) == 4:
                mask = mask.unsqueeze(-1).unsqueeze(-1)  # For Conv2d
            lx = lx * mask
            # Scale to maintain expected value
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))
        else:
            scale = self.scale
        
        # Up projection
        lx = self.lora_B(lx)
        
        return lx * scale
    
    def get_merged_weight(self) -> torch.Tensor:
        """Get the merged LoRA weight for inference."""
        return (self.lora_B.weight @ self.lora_A.weight) * self.scale


class AdaptiveDropoutLoRAConv2d(nn.Module):
    """
    LoRA layer with AdaptiveDropout support for Conv2d layers.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple = (1, 1),
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        rank: int = 4,
        alpha: float = 1.0,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank
        self.scale = self.alpha / self.rank
        
        # Dropout settings
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        
        # LoRA layers
        self.lora_A = nn.Conv2d(in_channels, rank, kernel_size, stride, padding, bias=False)
        self.lora_B = nn.Conv2d(rank, out_channels, (1, 1), (1, 1), (0, 0), bias=False)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def _ensure_same_device(self, x: torch.Tensor):
        """Ensure LoRA weights are on the same device as input."""
        if self.lora_A.weight.device != x.device:
            self.lora_A = self.lora_A.to(x.device)
            self.lora_B = self.lora_B.to(x.device)
        if self.lora_A.weight.dtype != x.dtype and x.dtype in [torch.float16, torch.bfloat16, torch.float32]:
            if not self.lora_A.weight.requires_grad:
                self.lora_A = self.lora_A.to(x.dtype)
                self.lora_B = self.lora_B.to(x.dtype)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure LoRA weights are on the same device/dtype as input (for offloading compatibility)
        self._ensure_same_device(x)
        
        # Module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1, device=x.device) < self.module_dropout:
                # Return zeros with proper output shape
                out_shape = list(x.shape)
                out_shape[1] = self.out_channels
                return torch.zeros(out_shape, dtype=x.dtype, device=x.device)
        
        # Down projection
        lx = self.lora_A(x)
        
        # Neuron dropout
        if self.dropout is not None and self.training:
            lx = nn.functional.dropout(lx, p=self.dropout)
        
        # Rank dropout
        if self.rank_dropout is not None and self.training:
            mask = torch.rand((lx.size(0), self.rank), device=lx.device) > self.rank_dropout
            mask = mask.unsqueeze(-1).unsqueeze(-1)  # [B, R, 1, 1]
            lx = lx * mask
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))
        else:
            scale = self.scale
        
        # Up projection
        lx = self.lora_B(lx)
        
        return lx * scale


class AdaptiveDropoutLoRAWrapper(nn.Module):
    """
    Wrapper that adds LoRA with AdaptiveDropout to an existing layer.
    Replaces the forward method while preserving the original weights.
    
    Compatible with DiffSynth's offloading mechanism (AutoWrappedLinear, etc.)
    """
    
    def __init__(
        self,
        original_module: nn.Module,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
    ):
        super().__init__()
        self.original_module = original_module
        self._is_auto_wrapped = hasattr(original_module, 'computation') or hasattr(original_module, 'offload')
        
        # Get the actual module properties for Linear/Conv2d detection
        # Handle AutoWrappedLinear and similar wrapped modules
        actual_module = original_module
        if hasattr(original_module, 'module'):
            actual_module = original_module.module
        
        if isinstance(actual_module, nn.Linear) or hasattr(original_module, 'in_features'):
            in_features = getattr(original_module, 'in_features', None) or actual_module.in_features
            out_features = getattr(original_module, 'out_features', None) or actual_module.out_features
            self.lora = AdaptiveDropoutLoRALinear(
                in_features=in_features,
                out_features=out_features,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                rank_dropout=rank_dropout,
                module_dropout=module_dropout,
            )
        elif isinstance(actual_module, nn.Conv2d) or hasattr(original_module, 'in_channels'):
            in_channels = getattr(original_module, 'in_channels', None) or actual_module.in_channels
            out_channels = getattr(original_module, 'out_channels', None) or actual_module.out_channels
            kernel_size = getattr(original_module, 'kernel_size', (1, 1)) or actual_module.kernel_size
            stride = getattr(original_module, 'stride', (1, 1)) or actual_module.stride
            padding = getattr(original_module, 'padding', (0, 0)) or actual_module.padding
            self.lora = AdaptiveDropoutLoRAConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                rank_dropout=rank_dropout,
                module_dropout=module_dropout,
            )
        else:
            raise ValueError(f"Unsupported module type: {type(original_module)}")
        
        # Freeze original module
        for param in self.original_module.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Call original module (may trigger offload/onload internally)
        base_output = self.original_module(x, *args, **kwargs)
        
        # Add LoRA output
        lora_output = self.lora(x)
        
        return base_output + lora_output
    
    # Proxy methods for offloading compatibility
    def offload(self):
        """Offload both original module and LoRA weights."""
        if hasattr(self.original_module, 'offload'):
            self.original_module.offload()
        # Keep LoRA on CPU when offloading
        self.lora = self.lora.to('cpu')
    
    def onload(self):
        """Onload original module (LoRA will move on forward)."""
        if hasattr(self.original_module, 'onload'):
            self.original_module.onload()
    
    def preparing(self):
        """Prepare original module."""
        if hasattr(self.original_module, 'preparing'):
            self.original_module.preparing()
    
    @property
    def weight(self):
        if hasattr(self.original_module, 'weight'):
            return self.original_module.weight
        return None
    
    @property
    def bias(self):
        if hasattr(self.original_module, 'bias'):
            return self.original_module.bias
        return None
    
    @property
    def in_features(self):
        return self.lora.in_features if hasattr(self.lora, 'in_features') else self.lora.in_channels
    
    @property
    def out_features(self):
        return self.lora.out_features if hasattr(self.lora, 'out_features') else self.lora.out_channels
    
    def __getattr__(self, name):
        """Proxy attribute access to original module for compatibility."""
        if name in ['original_module', 'lora', '_is_auto_wrapped', 'training']:
            return super().__getattr__(name)
        try:
            return super().__getattr__(name)
        except AttributeError:
            if hasattr(self, 'original_module'):
                return getattr(self.original_module, name)
            raise


def _is_lora_target_module(module: nn.Module) -> bool:
    """Check if a module is a valid LoRA target (Linear or Conv2d)."""
    # Check for standard PyTorch modules
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        return True
    
    # Check for DiffSynth's AutoWrappedLinear and similar wrapped modules
    # These have 'in_features'/'out_features' or 'in_channels'/'out_channels' attributes
    if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
        return True
    if hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
        return True
    
    # Check if it wraps a Linear/Conv2d
    if hasattr(module, 'module'):
        inner = module.module
        if isinstance(inner, (nn.Linear, nn.Conv2d)):
            return True
    
    return False


def inject_adaptive_dropout_lora(
    model: nn.Module,
    target_modules: Union[str, List[str]],
    rank: int = 4,
    alpha: float = 1.0,
    dropout: Optional[float] = None,
    rank_dropout: Optional[float] = None,
    module_dropout: Optional[float] = None,
) -> nn.Module:
    """
    Inject AdaptiveDropout LoRA layers into a model.
    
    Compatible with DiffSynth's offloading mechanism (AutoWrappedLinear, etc.)
    
    Args:
        model: The model to inject LoRA into
        target_modules: Module names to target (supports regex)
        rank: LoRA rank
        alpha: LoRA alpha for scaling
        dropout: Neuron dropout probability
        rank_dropout: Rank dropout probability
        module_dropout: Module dropout probability
    
    Returns:
        Modified model with LoRA layers
    """
    if isinstance(target_modules, str):
        target_modules = [target_modules]
    
    # Compile regex patterns
    patterns = [re.compile(pattern) for pattern in target_modules]
    
    # Find all modules to replace
    modules_to_replace = {}
    for name, module in model.named_modules():
        # Skip already wrapped modules
        if isinstance(module, AdaptiveDropoutLoRAWrapper):
            continue
            
        # Check if module is a valid LoRA target
        if _is_lora_target_module(module):
            for pattern in patterns:
                if pattern.search(name):
                    modules_to_replace[name] = module
                    break
    
    # Replace modules
    replaced_count = 0
    for name, module in modules_to_replace.items():
        try:
            # Navigate to parent module
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            
            # Create wrapper
            wrapper = AdaptiveDropoutLoRAWrapper(
                original_module=module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                rank_dropout=rank_dropout,
                module_dropout=module_dropout,
            )
            
            # Replace
            setattr(parent, parts[-1], wrapper)
            replaced_count += 1
        except Exception as e:
            print(f"Warning: Failed to inject LoRA into {name}: {e}")
    
    print(f"Injected AdaptiveDropout LoRA into {replaced_count} layers")
    print(f"  - Rank: {rank}, Alpha: {alpha}")
    print(f"  - Dropout: {dropout}, Rank Dropout: {rank_dropout}, Module Dropout: {module_dropout}")
    
    return model


def get_lora_state_dict(model: nn.Module, remove_prefix: Optional[str] = None) -> Dict[str, torch.Tensor]:
    """
    Extract LoRA weights from a model.
    
    Args:
        model: Model with LoRA layers
        remove_prefix: Optional prefix to remove from keys
    
    Returns:
        State dict containing only LoRA weights
    """
    state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, AdaptiveDropoutLoRAWrapper):
            lora_name = name
            if remove_prefix and lora_name.startswith(remove_prefix):
                lora_name = lora_name[len(remove_prefix):]
            
            state_dict[f"{lora_name}.lora_A.weight"] = module.lora.lora_A.weight
            state_dict[f"{lora_name}.lora_B.weight"] = module.lora.lora_B.weight
            state_dict[f"{lora_name}.alpha"] = torch.tensor(module.lora.alpha)
    
    return state_dict


def convert_to_opensource_format(state_dict: Dict[str, torch.Tensor], model_type: str = "flux") -> Dict[str, torch.Tensor]:
    """
    Convert AdaptiveDropout LoRA state dict to opensource format (compatible with civitai/kohya).
    
    Args:
        state_dict: LoRA state dict
        model_type: Type of model ("flux", "qwen_image", "z_image")
    
    Returns:
        Converted state dict
    """
    converted = {}
    
    for key, value in state_dict.items():
        # Convert lora_A/lora_B naming to lora_down/lora_up
        new_key = key.replace(".lora_A.weight", ".lora_down.weight")
        new_key = new_key.replace(".lora_B.weight", ".lora_up.weight")
        
        # Add model-specific prefix
        if model_type == "flux":
            if "blocks." in new_key:
                new_key = "lora_unet_double_" + new_key.replace(".", "_")
            elif "single_blocks." in new_key:
                new_key = "lora_unet_single_" + new_key.replace(".", "_")
        
        converted[new_key] = value
    
    return converted

