"""Custom normalization layers — aligns with vLLM vllm/model_executor/layers/layernorm.py.

Architecture preserved: RMSNorm (CustomOp, residual path), GemmaRMSNorm, LayerNorm.

Engineering details skipped: CUDA kernel dispatch (always forward_native), vllm.ir.ops,
batch_invariant path, CustomOp registration decorator (registered as rms_norm).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from my_vllm.model_executor.custom_op import CustomOp


class RMSNorm(CustomOp):
    """Root mean square normalization — aligns with vLLM RMSNorm.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467

    Supports the fused_add_rms_norm residual path: when residual is not None,
    computes x = x + residual, then norm, and returns (normed, x).
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: int | None = None,
        has_weight: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.variance_size_override = (
            None if var_hidden_size == hidden_size else var_hidden_size
        )
        weight_dtype = dtype or torch.get_default_dtype()
        self.has_weight = has_weight
        weight = torch.ones(hidden_size, dtype=weight_dtype)
        if self.has_weight:
            self.weight = nn.Parameter(weight)
        else:
            self.register_buffer("weight", weight)
        self.pass_weight = self.has_weight
        self.pass_weight_add = self.has_weight

    def _rms_norm(
        self, x: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        """Pure PyTorch RMS norm — equivalent to vllm.ir.ops.rms_norm."""
        variance_size = self.variance_size_override or x.shape[-1]
        x_float = x.float()
        variance = x_float.pow(2).sum(dim=-1, keepdim=True) / variance_size
        x_normed = x_float * torch.rsqrt(variance + self.variance_epsilon)
        if self.has_weight:
            x_normed = x_normed * weight.float()
        return x_normed.to(x.dtype)

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self._rms_norm(x, self.weight.data if self.pass_weight else None)
        else:
            x_orig_dtype = x.dtype
            x = (x + residual).to(x_orig_dtype)
            normed = self._rms_norm(
                x, self.weight.data if self.pass_weight_add else None
            )
            return normed, x

    def extra_repr(self) -> str:
        w = self.weight.data if self.has_weight else self.weight
        s = f"hidden_size={w.size(0)}"
        s += f", eps={self.variance_epsilon}"
        return s


class GemmaRMSNorm(CustomOp):
    """RMS normalization for Gemma — aligns with vLLM GemmaRMSNorm.

    Differences from RMSNorm:
        1. x * (1 + w) instead of x * w.
        2. (x * w).to(orig_dtype) instead of x.to(orig_dtype) * w.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def _rms_norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_float * torch.rsqrt(variance + self.variance_epsilon)
        x_normed = x_normed * (1.0 + weight.float())
        return x_normed.to(x.dtype)

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self._rms_norm(x, self.weight)
        else:
            x = x + residual
            normed = self._rms_norm(x, self.weight)
            return normed, x


class LayerNorm(nn.Module):
    """Layer Normalization — aligns with vLLM LayerNorm."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.float(), (self.dim,), self.weight, self.bias, self.eps
        ).type_as(x)
