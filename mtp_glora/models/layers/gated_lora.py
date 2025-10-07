import math
import warnings
from typing import List

import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel


class GatedLoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
        lora_bias: bool,
        use_rslora: bool,
    ):
        """
        Gated LoRA Linear Layer.

        Args:
            base_layer: The base linear layer to wrap.
            rank: The rank of the LoRA layer.
            alpha: The alpha of the LoRA layer. Typically 2x rank.
            dropout: The dropout of the LoRA layer.
            lora_bias: Whether to use bias in the LoRA layer.
            use_rslora: Whether to use RS-LoRA. If True, the scaling factor will be set to alpha / sqrt(rank).
        """

        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.lora_scaling = alpha / math.sqrt(rank) if use_rslora else alpha / rank
        if lora_bias and (base_layer.bias is None):
            warnings.warn("lora_bias is True but base_linear.bias is None")

        if dropout > 0: self.dropout = nn.Dropout(p=dropout)
        else: self.dropout = nn.Identity()
        
        self.lora_A = nn.Linear(base_layer.in_features, rank, bias=False, dtype=base_layer.weight.dtype, device=base_layer.weight.device)
        self.lora_B = nn.Linear(rank, base_layer.out_features, bias=lora_bias, dtype=base_layer.weight.dtype, device=base_layer.weight.device)
        self._init_lora_weights()

    def _init_lora_weights(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor, gate_mask: torch.Tensor):
        out, lora_out = self.base_layer(x), self.lora_B(self.lora_A(self.dropout(x)))
        return out + self.lora_scaling * lora_out * gate_mask


def _freeze_base_and_enable_lora(model):
    for name, param in model.named_parameters():
        param.requires_grad = ('lora_' in name)


def apply_lora_to_model(
    model: PreTrainedModel,
    target_modules: List[str] = ['qkv_proj', 'o_proj', 'gate_up_proj', 'down_proj'],
    rank: int = 16,
    alpha: float = 32,
    dropout: float = 0.1,
    lora_bias: bool = False,
    use_rslora: bool = False,
):
    """
    Wrap nn.Linear with GatedLoRALinear Wrapper. Support fused qkv projection and gate-up projection.
    For fused projections, the rank will be 3x or 2x the original rank.

    Args:
        model: The model to apply LoRA to.
        target_modules: The target modules to apply LoRA to.
        rank: The rank of the LoRA layer.
        alpha: The alpha of the LoRA layer.
        dropout: The dropout of the LoRA layer.
        lora_bias: Whether to use bias in the LoRA layer.
        use_rslora: Whether to use RS-LoRA.
    """

    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            if any(target_module in name for target_module in target_modules):
                if 'qkv_proj' in name:
                    temp_rank = 3 * rank
                    temp_alpha = 3 * alpha
                elif 'gate_up_proj' in name:
                    temp_rank = 2 * rank
                    temp_alpha = 2 * alpha
                else:
                    temp_rank = rank
                    temp_alpha = alpha
                lora_layer = GatedLoRALinear(module, temp_rank, temp_alpha, dropout, lora_bias, use_rslora)
                setattr(model, name, lora_layer)
        else:
            apply_lora_to_model(module, target_modules, rank, alpha, dropout, lora_bias, use_rslora) # recursive search
    _freeze_base_and_enable_lora(model)