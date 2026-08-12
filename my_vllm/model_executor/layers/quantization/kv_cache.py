"""FP8 Key-Value cache quantization method.

In real vLLM, this manages per-layer KV cache scales for FP8 quantization
when kv_cache_dtype is set to "fp8" or "fp8_e4m3".
Scales are applied during reshape_and_cache (write) and gather (read).
"""

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .base_config import QuantizeMethodBase

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention import Attention


class KVCacheScaleParameter(nn.Parameter):
    """Per-layer KV cache scale parameter (1-D tensor).

    This is a torch.nn.Parameter that holds a single float32 scale value
    used to quantize/dequantize KV cache tensors for FP8 storage.
    """

    def __new__(cls, data: torch.Tensor | None = None):
        if data is None:
            data = torch.tensor([1.0], dtype=torch.float32)
        return super().__new__(cls, data, requires_grad=False)

    def dtype(self) -> torch.dtype:
        return torch.float32


class BaseKVCacheMethod(QuantizeMethodBase):
    """Base class for KV cache quantization methods.

    Tracks per-layer scales for key and value caches, and provides
    the interface for quantizing / dequantizing KV tensors.
    """

    def __init__(self, quant_config):
        self.quant_config = quant_config
        self.scales: list[KVCacheScaleParameter] = []

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        raise NotImplementedError

    def apply(self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError

    def get_kv_scale(self, layer_idx: int) -> torch.Tensor:
        """Get the KV cache scale for a given layer."""
        if layer_idx < len(self.scales):
            return self.scales[layer_idx]
        return torch.tensor([1.0], dtype=torch.float32)
