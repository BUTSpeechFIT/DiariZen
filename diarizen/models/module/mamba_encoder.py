from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import inspect

from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.block import Block
from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.ops.triton.layer_norm import RMSNorm


class MambaEncoder(nn.Module):
    """
    Mamba2-based encoder as a drop-in replacement for ConformerEncoder.
    
    Compatible interface with ConformerEncoder:
    - Input: (batch, time, features)
    - Output: (batch, time, features) or (batch, time, features*2) if bidirectional
    
    Using Mamba2 for improved performance and efficiency.
    """

    def __init__(
        self,
        attention_in: int = 256,
        num_layer: int = 4,
        d_state: int = 128,  # Mamba2 default
        d_conv: int = 4,
        expand: int = 2,  # Mamba2 default
        headdim: int = 64,  # Mamba2 parameter
        ngroups: int = 1,   # Mamba2 parameter
        rmsnorm_eps: float = 1e-5,
        bidirectional: bool = True,
        bidirectional_merging: str = "add",  # "concat", "add", "mul"
        output_activate_function: str = None,
    ):
        super().__init__()
        
        self.bidirectional = bidirectional
        self.bidirectional_merging = bidirectional_merging
        self.d_model = attention_in
        
        print(f"🔧 Initializing Mamba2Encoder: d_model={attention_in}, layers={num_layer}, bidirectional={bidirectional}")
        
        # Check Mamba2 and Block parameters
        mamba_params = inspect.signature(Mamba2).parameters
        supports_headdim = 'headdim' in mamba_params
        supports_ngroups = 'ngroups' in mamba_params
        
        block_params = inspect.signature(Block).parameters
        needs_mlp_cls = 'mlp_cls' in block_params
        
        print(f"   Block needs mlp_cls: {needs_mlp_cls}")
        
        # Create mixer_cls partial function - Block will pass dim as d_model
        def create_mamba_partial(layer_idx):
            kwargs = {
                'd_state': d_state,
                'd_conv': d_conv,
                'expand': expand,
                'layer_idx': layer_idx
            }
            if supports_headdim:
                kwargs['headdim'] = headdim
            if supports_ngroups:
                kwargs['ngroups'] = ngroups
            return partial(Mamba2, **kwargs)
        
        # Forward direction blocks
        self.forward_blocks = nn.ModuleList([])
        for i in range(num_layer):
            block_kwargs = {
                'dim': attention_in,
                'mixer_cls': create_mamba_partial(i),
                'norm_cls': partial(RMSNorm, eps=rmsnorm_eps),
                'fused_add_norm': False,
            }
            
            # Only add mlp_cls if Block expects it
            if needs_mlp_cls:
                # Use the simplest possible MLP that matches the interface
                def simple_mlp_fn(dim):
                    return nn.Identity()  # Even simpler than before
                block_kwargs['mlp_cls'] = simple_mlp_fn
            
            self.forward_blocks.append(Block(**block_kwargs))
            
        # Backward direction blocks (if bidirectional)
        if bidirectional:
            self.backward_blocks = nn.ModuleList([])
            for i in range(num_layer):
                block_kwargs = {
                    'dim': attention_in,
                    'mixer_cls': create_mamba_partial(i + num_layer),
                    'norm_cls': partial(RMSNorm, eps=rmsnorm_eps),
                    'fused_add_norm': False,
                }
                
                if needs_mlp_cls:
                    def simple_mlp_fn(dim):
                        return nn.Identity()
                    block_kwargs['mlp_cls'] = simple_mlp_fn
                
                self.backward_blocks.append(Block(**block_kwargs))

        # Output projection if bidirectional and concat
        if bidirectional and bidirectional_merging == "concat":
            self.output_proj = nn.Linear(attention_in * 2, attention_in)
        else:
            self.output_proj = None

        # Activation function layer (compatible with ConformerEncoder)
        if output_activate_function:
            if output_activate_function == "Tanh":
                self.activate_function = nn.Tanh()
            elif output_activate_function == "ReLU":
                self.activate_function = nn.ReLU()
            elif output_activate_function == "ReLU6":
                self.activate_function = nn.ReLU6()
            elif output_activate_function == "LeakyReLU":
                self.activate_function = nn.LeakyReLU()
            elif output_activate_function == "PReLU":
                self.activate_function = nn.PReLU()
            elif output_activate_function == "Sigmoid":
                self.activate_function = nn.Sigmoid()
            else:
                raise NotImplementedError(
                    f"Not implemented activation function {output_activate_function}"
                )
        self.output_activate_function = output_activate_function

        # Initialize weights
        self.apply(partial(_init_weights, n_layer=num_layer))
        
        print(f"✅ Mamba2Encoder initialized successfully with {sum(p.numel() for p in self.parameters()):,} parameters")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (#batch, time, features).
            
        Returns:
            torch.Tensor: Output tensor (#batch, time, features).
        """
        batch_size, seq_len, features = x.shape
        
        # Forward direction
        for_residual = None
        forward_f = x.clone()
        for block in self.forward_blocks:
            forward_f, for_residual = block(forward_f, for_residual, inference_params=None)
        forward_output = (forward_f + for_residual) if for_residual is not None else forward_f

        # Backward direction (if bidirectional)
        if self.bidirectional:
            back_residual = None
            backward_f = torch.flip(x, [1])  # Flip along time dimension
            for block in self.backward_blocks:
                backward_f, back_residual = block(backward_f, back_residual, inference_params=None)
            backward_output = (backward_f + back_residual) if back_residual is not None else backward_f
            backward_output = torch.flip(backward_output, [1])  # Flip back
            
            # Merge bidirectional outputs
            if self.bidirectional_merging == "concat":
                output = torch.cat([forward_output, backward_output], dim=-1)
                # Project back to original dimension
                if self.output_proj is not None:
                    output = self.output_proj(output)
            elif self.bidirectional_merging == "add":
                output = forward_output + backward_output
            elif self.bidirectional_merging == "mul":
                output = forward_output * backward_output
            else:
                raise ValueError(f"Invalid bidirectional_merging: {self.bidirectional_merging}")
        else:
            output = forward_output

        # Apply activation function if specified
        if self.output_activate_function:
            output = self.activate_function(output)
            
        return output