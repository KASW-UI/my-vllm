"""Quantization utility helpers — package.

Mirrors vLLM V1 model_executor/layers/quantization/utils/.

C140: re-export `fp8_utils` for FP8 helpers (`is_fp8`,
`get_fp8_min_max`, `per_tensor_quantize`,
`per_tensor_dequantize`, `per_token_quant_fp8`).
"""

from my_vllm.model_executor.layers.quantization.utils.fp8_utils import (
    get_fp8_min_max,
    is_fp8,
    per_tensor_dequantize,
    per_tensor_quantize,
    per_token_quant_fp8,
)

__all__ = [
    "get_fp8_min_max",
    "is_fp8",
    "per_tensor_dequantize",
    "per_tensor_quantize",
    "per_token_quant_fp8",
]