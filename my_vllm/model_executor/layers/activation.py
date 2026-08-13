"""Custom activation functions — aligns with vLLM vllm/model_executor/layers/activation.py.

Architecture preserved: SiluAndMul, GeluAndMul, GELU, NewGELU, ReLUSquaredActivation,
QuickGELU, FastGELU, SwigluOAIAndMul, SwigluStepAndMul, ScaledActivation,
get_act_fn(), get_act_and_mul_fn().

Engineering details skipped: CUDA kernel dispatch (always forward_native),
platform-specific code, Triton kernels, XIELU CUDA integration.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from my_vllm.distributed.parallel_state import (
    get_tp_rank,
    get_tp_world_size,
)
from my_vllm.model_executor.custom_op import CustomOp


def divide(n: int, d: int) -> int:
    """Divide n by d, asserting exact divisibility."""
    assert n % d == 0, f"{n} not divisible by {d}"
    return n // d


# ---------------------------------------------------------------------------
# Element-wise activation functions
# ---------------------------------------------------------------------------

class SiluAndMul(CustomOp):
    """An activation function for SwiGLU.

    The function computes x -> silu(x[:d]) * x[d:] where d = x.shape[-1] // 2.

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    def __init__(self, *, compile_native: bool = True):
        super().__init__(compile_native=compile_native)

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]


class GeluAndMul(CustomOp):
    """An activation function for GeGLU.

    The function computes x -> GELU(x[:d]) * x[d:] where d = x.shape[-1] // 2.

    Shapes:
        x: (batch_size, seq_len, 2 * d) or (num_tokens, 2 * d)
        return: (batch_size, seq_len, d) or (num_tokens, d)
    """

    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate
        if approximate not in ("none", "tanh"):
            raise ValueError(f"Unknown approximate mode: {approximate}")

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.gelu(x[..., :d], approximate=self.approximate) * x[..., d:]

    def extra_repr(self) -> str:
        return f"approximate={repr(self.approximate)}"


class SwigluOAIAndMul(CustomOp):
    """SwiGLU-OAI activation: silu(gate, scaled) * (up + 1)."""

    def __init__(self, alpha: float = 1.702, limit: float = 7.0):
        super().__init__()
        self.alpha = alpha
        self.limit = limit

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = x[..., ::2], x[..., 1::2]
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return gate * torch.sigmoid(gate * self.alpha) * (up + 1)

    def extra_repr(self) -> str:
        return f"alpha={repr(self.alpha)}, limit={repr(self.limit)}"


class SwigluStepAndMul(CustomOp):
    """SwiGLU with clamping: silu(gate).clamp(max) * up.clamp(-limit, limit)."""

    def __init__(self, limit: float = 7.0):
        super().__init__()
        if limit is None:
            raise ValueError("SwigluStepAndMul requires limit to be set.")
        self.limit = limit

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        gate = F.silu(gate).clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return gate * up

    def extra_repr(self) -> str:
        return f"limit={repr(self.limit)}"


class MulAndSilu(CustomOp):
    """SwiGLU (reversed order): x -> x[:d] * silu(x[d:])."""

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return x[..., :d] * F.silu(x[..., d:])


class GELU(CustomOp):
    """Plain GELU activation (approximate='none')."""

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate="none")


class FastGELU(CustomOp):
    """Fast GELU approximation."""

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            x * 0.7978845608 * (1.0 + 0.044715 * x * x)
        ))


class NewGELU(CustomOp):
    """New GELU activation (used by OPT)."""

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * torch.pow(x, 3.0))))


class QuickGELU(CustomOp):
    """Quick GELU: x * sigmoid(1.702 * x)."""

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ReLUSquaredActivation(CustomOp):
    """ReLU² activation."""

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        return torch.square(F.relu(x))


class ScaledActivation(nn.Module):
    """An activation function with post-scale parameters (used for AWQ quantization)."""

    def __init__(
        self,
        act_module: nn.Module,
        intermediate_size: int,
        input_is_parallel: bool = True,
        params_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.act = act_module
        self.input_is_parallel = input_is_parallel
        if input_is_parallel:
            tp_size = get_tp_world_size()
            intermediate_size_per_partition = divide(intermediate_size, tp_size)
        else:
            intermediate_size_per_partition = intermediate_size
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.scales = nn.Parameter(
            torch.empty(intermediate_size_per_partition, dtype=params_dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x) / self.scales

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if self.input_is_parallel:
            tp_rank = get_tp_rank()
            shard_size = param_data.shape[0]
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


# ---------------------------------------------------------------------------
# Activation registry
# ---------------------------------------------------------------------------

_ACTIVATION_REGISTRY: dict[str, type | nn.Module] = {
    "gelu": GELU,
    "gelu_fast": FastGELU,
    "gelu_new": NewGELU,
    "gelu_pytorch_tanh": lambda: nn.GELU(approximate="tanh"),
    "relu": nn.ReLU,
    "relu2": ReLUSquaredActivation,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "quick_gelu": QuickGELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


def get_act_fn(act_fn_name: str) -> nn.Module:
    """Get an activation function by name — aligns with vLLM get_act_fn()."""
    act_fn_name = act_fn_name.lower()

    if act_fn_name.startswith("torch.nn.modules."):
        activation_name = act_fn_name.split(".")[-1]
        if activation_name == "identity":
            return nn.Identity()
        act_fn_name = activation_name

    if act_fn_name not in _ACTIVATION_REGISTRY:
        raise ValueError(
            f"Activation function {act_fn_name!r} is not supported."
        )

    entry = _ACTIVATION_REGISTRY[act_fn_name]
    if isinstance(entry, type):
        return entry()
    elif callable(entry):
        return entry()
    return entry


_ACTIVATION_AND_MUL_REGISTRY = {
    "gelu": GeluAndMul,
    "gelu_pytorch_tanh": lambda: GeluAndMul(approximate="tanh"),
    "silu": SiluAndMul,
    "swish": SiluAndMul,
    "geglu": GeluAndMul,
    "swigluoai": SwigluOAIAndMul,
}


def get_act_and_mul_fn(act_fn_name: str, *, compile_native: bool = True) -> nn.Module:
    """Get an activation-and-mul function by name — aligns with vLLM."""
    act_fn_name = act_fn_name.lower()

    if not compile_native and act_fn_name in ("silu", "swish"):
        return SiluAndMul(compile_native=False)

    if act_fn_name not in _ACTIVATION_AND_MUL_REGISTRY:
        raise ValueError(
            f"Activation function {act_fn_name!r} is not supported."
        )

    entry = _ACTIVATION_AND_MUL_REGISTRY[act_fn_name]
    if isinstance(entry, type):
        return entry()
    elif callable(entry):
        return entry()
    return entry
