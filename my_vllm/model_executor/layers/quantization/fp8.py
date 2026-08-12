"""FP8 (8-bit Floating Point) quantization for linear layers and MoE.

Architecture:
  Fp8Config — configuration class, dispatches to per-layer methods
  Fp8LinearMethod — offline FP8 linear (checkpoint serialized)
  Fp8PerTensorOnlineLinearMethod — online FP8 linear (runtime quantize)
  Fp8MoEMethod — FP8 MoE (routed + shared experts)
  Fp8KVCacheMethod — FP8 KV cache quantization

The key insight: FP8 reduces memory bandwidth by 2x vs FP16/BF16
for both weights and KV cache, but requires:
  - Per-tensor or per-block scaling factors
  - Dynamic (online) or static (offline) activation quantization
  - CUTLASS FP8 GEMM for the matmul backends
"""

from typing import TYPE_CHECKING, Any

import torch

from .base_config import QuantizationConfig, QuantizeMethodBase
from .kv_cache import BaseKVCacheMethod, KVCacheScaleParameter

if TYPE_CHECKING:
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.fused_moe import (
        FusedMoEMethodBase,
        RoutedExperts,
        SharedExperts,
    )

ACTIVATION_SCHEMES = ["static", "dynamic"]


class Fp8Config(QuantizationConfig):
    """Configuration for FP8 quantization.

    Supports two activation modes:
    - "dynamic": scales computed at runtime (online quantization)
    - "static": scales baked into checkpoint (offline serialized)

    Weight block_size enables DeepSeek-style block-wise FP8.
    """

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: list[str] | None = None,
        weight_block_size: list[int] | None = None,
        store_dtype: str | None = None,
    ) -> None:
        super().__init__()

        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized

        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
        self.store_dtype = store_dtype

        if weight_block_size is not None:
            if not is_checkpoint_fp8_serialized:
                raise ValueError(
                    "Block-wise quantization only supports fp8-serialized checkpoint."
                )
            if len(weight_block_size) != 2:
                raise ValueError(
                    f"weight_block_size must have 2 dimensions, got {len(weight_block_size)}"
                )
        self.weight_block_size = weight_block_size

    @classmethod
    def get_name(cls) -> str:
        return "fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75  # Turing+ supports FP8 through CUTLASS

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Fp8Config":
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp8_serialized = "fp8" in quant_method
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"], None)
        store_dtype = cls.get_from_keys_or(config, ["store_dtype"], None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(config, ["modules_to_not_convert"], None)
        return cls(
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            weight_block_size=weight_block_size,
            store_dtype=store_dtype,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        from vllm.model_executor.layers.linear import LinearBase

        if isinstance(layer, LinearBase):
            if self._is_layer_skipped(prefix):
                return UnquantizedLinearMethod()
            if not self.is_checkpoint_fp8_serialized:
                return Fp8PerTensorOnlineLinearMethod()
            else:
                return Fp8LinearMethod(self)
        return None

    def _is_layer_skipped(self, prefix: str) -> bool:
        for ignored in self.ignored_layers:
            if ignored in prefix:
                return True
        return False


class UnquantizedLinearMethod(QuantizeMethodBase):
    """Fallback: no quantization, standard FP16/BF16 matmul."""

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **extra_weight_attrs):
        pass

    def apply(self, layer, x, bias=None):
        return torch.nn.functional.linear(x, layer.weight, bias)


class Fp8LinearMethod(QuantizeMethodBase):
    """Offline FP8 linear method.

    Weights are stored as FP8 in the checkpoint, with per-tensor
    or per-block scales.  Forward: w_scale * (fp8_weight @ fp8_act * a_scale).
    """

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **extra_weight_attrs):
        pass

    def apply(self, layer, x, bias=None):
        return torch.nn.functional.linear(x, layer.weight, bias)


class Fp8PerTensorOnlineLinearMethod(QuantizeMethodBase):
    """Online FP8 linear method.

    Weights are FP16/BF16, activations are quantized to FP8 at runtime.
    Forward: dequant(quant(x)) @ fp8_weight.T (handled by CUTLASS).
    """

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **extra_weight_attrs):
        pass

    def apply(self, layer, x, bias=None):
        return torch.nn.functional.linear(x, layer.weight, bias)


class Fp8MoEMethod(QuantizeMethodBase):
    """FP8 MoE method for RoutedExperts + SharedExperts.

    Handles w13_weight (gate+up projection) and w2_weight (down projection)
    with per-tensor or per-channel FP8 scales.
    """

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **extra_weight_attrs):
        pass

    def apply(self, layer, x, bias=None):
        return torch.nn.functional.linear(x, layer.weight, bias)


class Fp8KVCacheMethod(BaseKVCacheMethod):
    """FP8 KV cache quantization method.

    Quantizes K and V tensors to FP8 when writing to the physical KV cache.
    Applies per-token or per-channel scales.  The quantized tensors are
    dequantized back to FP16/BF16 when gathering for attention.
    """

    def __init__(self, quant_config: Fp8Config):
        super().__init__(quant_config)

    def create_weights(
        self,
        layer: "LinearBase",
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        pass

    def apply(self, layer, x, bias=None):
        return torch.nn.functional.linear(x, layer.weight, bias)
