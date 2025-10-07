import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        x_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        y = x * torch.rsqrt(var + self.variance_epsilon)
        if y.dtype == self.weight.dtype: return (y * self.weight).to(x_dtype)
        else: return y.to(x_dtype) * self.weight


class SamplerHeadBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        eps: float = 1e-5,
    ):
        """
        Sampler Head Block.

        Args:
            in_features: The input feature size.
            out_features: The output feature size.
            eps: The epsilon for the RMSNorm.
        """

        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.norm = RMSNorm(out_features, eps=eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(F.silu(self.linear(x)))


class SamplerHead(nn.Module):
    def __init__(self, 
        config: PretrainedConfig,
        use_residual: bool = True,
    ):
        """
        Sampler Head.

        Args:
            config: The configuration of the model.
            use_residual: Whether to use residual connection.
        """

        super().__init__()

        hidden_size = config.hidden_size
        self.use_residual = use_residual

        self.layers = nn.ModuleList([
            SamplerHeadBlock(2 * hidden_size, hidden_size, eps=config.rms_norm_eps),
            SamplerHeadBlock(hidden_size, hidden_size, eps=config.rms_norm_eps),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers[0](x)
        y = self.layers[1](x)
        if self.use_residual: y = x + y
        return y
