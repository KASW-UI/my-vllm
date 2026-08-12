"""QuantizationConfig + backend routing — aligns with vLLM quantization layer.

JD alignment:
  - 异构推理: Triton vs CUDA 路径选择
  - GPU与CUDA能力: 量化方法的后端选择体现硬件理解

vLLM quantization landscape (per user intel):
  ┌──────────┬──────────────────────────┐
  │ Method   │ Backend                  │
  ├──────────┼──────────────────────────┤
  │ FP8 E4M3 │ Triton (fp8_utils.py)    │
  │ AWQ      │ Triton (AWQ kernel)      │
  │ GPTQ     │ CUDA/CUTLASS (Marlin)    │
  │ Marlin   │ CUDA only (SM80+)        │
  │ NVFP4    │ Triton emul / CUDA B200  │
  └──────────┴──────────────────────────┘

Design decision: quantization config is separated from model config because
different GPUs (H100 vs B200 vs L40S) support different quantization methods.
The VllmConfig validates compatibility between QuantizationConfig and the
available GPU hardware at init time.
"""

import enum
from dataclasses import dataclass, field


class QuantizationMethod(str, enum.Enum):
    """Supported quantization methods — aligns with vLLM quantization registry."""
    NONE = "none"
    FP8 = "fp8"
    AWQ = "awq"
    GPTQ = "gptq"
    MARLIN = "marlin"
    NVFP4 = "nvfp4"


@dataclass
class QuantizationConfig:
    """Quantization configuration with backend routing.

    Separates "what" (method) from "how" (backend), enabling heterogeneous
    hardware inference — different GPU types in the same cluster can use
    their optimal backend for the same quantized model.

    Aligns with vLLM vllm/config/quantization.py domain logic.
    """

    method: QuantizationMethod = QuantizationMethod.NONE
    kv_cache_dtype: str = "auto"
    weight_dtype: str = "auto"

    # Per-method parameters
    fp8_format: str = "e4m3"               # "e4m3" | "e5m2"
    awq_group_size: int = 128
    marlin_tile_size: int = 16

    @property
    def is_quantized(self) -> bool:
        return self.method != QuantizationMethod.NONE

    @property
    def triton_path(self) -> bool:
        """Whether this quantization uses the Triton backend.

        FP8, AWQ, and NVFP4 (emulation) all use Triton kernels.
        Marlin and GPTQ use CUDA/CUTLASS.
        """
        return self.method in {
            QuantizationMethod.FP8,
            QuantizationMethod.AWQ,
            QuantizationMethod.NVFP4,
        }

    @property
    def cuda_path(self) -> bool:
        """Whether this quantization uses the CUDA/CUTLASS backend."""
        return self.method in {
            QuantizationMethod.GPTQ,
            QuantizationMethod.MARLIN,
        }

    @property
    def kv_cache_bytes_per_element(self) -> int:
        """Return bytes per element in KV cache for memory planning."""
        if self.kv_cache_dtype == "fp8_e4m3":
            return 1
        if self.kv_cache_dtype == "auto" and self.method == QuantizationMethod.FP8:
            return 1
        return 2

    @property
    def memory_savings_ratio(self) -> float:
        """Estimated memory savings vs FP16 baseline."""
        ratios = {
            QuantizationMethod.NONE: 1.0,
            QuantizationMethod.FP8: 0.50,
            QuantizationMethod.AWQ: 0.25,
            QuantizationMethod.GPTQ: 0.25,
            QuantizationMethod.MARLIN: 0.25,
            QuantizationMethod.NVFP4: 0.125,
        }
        return ratios.get(self.method, 1.0)


def select_backend(
    method: QuantizationMethod,
    gpu_sm: int | None = None,
) -> str:
    """Select Triton vs CUDA backend based on method + GPU capability.

    Returns: "triton" | "cuda" | "fallback"

    Aligns with vLLM's automatic backend selection in:
      - model_executor/layers/quantization/compressed_tensors/
      - v1/worker/gpu_model_runner.py quantization section
    """
    if method == QuantizationMethod.MARLIN:
        if gpu_sm is not None and gpu_sm < 80:
            return "fallback"
        return "cuda"

    if method == QuantizationMethod.GPTQ:
        return "cuda"

    if method in (QuantizationMethod.FP8, QuantizationMethod.AWQ):
        if gpu_sm is not None and gpu_sm < 80:
            return "fallback"
        return "triton"

    if method == QuantizationMethod.NVFP4:
        if gpu_sm is not None and gpu_sm >= 100:
            return "cuda"       # B200 native
        return "triton"          # emulation

    return "none"
