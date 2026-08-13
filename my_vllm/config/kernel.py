"""KernelConfig — kernel selection / warmup configuration.

Mirrors vLLM V1 config/kernel.py:166.

C137: minimal subset — `moe_backend` and `linear_backend` literals
plus the `ir_op_priority` nested config. my-vllm does not yet wire
any of these to actual kernel dispatch (the engine is single-arch,
single-quantization, single-backend); the config exists for 1:1
file-tree alignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


MoEBackend = Literal[
    "auto",
    "triton",
    "deep_gemm",
    "deep_gemm_mega_moe",
    "cutlass",
    "flashinfer_trtllm",
    "flashinfer_cutlass",
    "flashinfer_cutedsl",
    "marlin",
    "humming",
    "triton_unfused",
    "aiter",
    "flydsl",
    "hpc",
    "emulation",
]


LinearBackend = Literal[
    "auto",
    "cutlass",
    "flashinfer_cutlass",
    "flashinfer_cutedsl",
    "flashinfer_trtllm",
    "marlin",
    "triton",
    "deep_gemm",
    "torch",
    "aiter",
    "machete",
    "fbgemm",
    "emulation",
    "xpu",
    "xpu_woq",
]


@dataclass
class IrOpPriorityConfig:
    """Priority list for vLLM-style IR op dispatch.

    Mirrors vLLM V1 config/kernel.py:IrOpPriorityConfig (L19).

    my-vllm does not yet have an IR-dispatch layer; the config exists
    for 1:1 alignment. A future refactor can wire these lists into
    `my_vllm.model_executor.layers.*` op selection.
    """

    rms_norm: list[str] = field(default_factory=list)
    """Priority list for vLLM-style rms_norm ops."""

    fused_add_rms_norm: list[str] = field(default_factory=list)
    """Priority list for vLLM-style fused_add_rms_norm ops."""


@dataclass
class KernelConfig:
    """Configuration for kernel selection and warmup.

    Mirrors vLLM V1 config/kernel.py:KernelConfig.
    """

    ir_op_priority: IrOpPriorityConfig = field(
        default_factory=IrOpPriorityConfig
    )
    """IR op priority config."""

    moe_backend: MoEBackend = "auto"
    """Backend for MoE expert computation kernels."""

    linear_backend: LinearBackend = "auto"
    """Backend for quantized linear-layer GEMM kernels."""

    def compute_hash(self) -> str:
        """Hash affected by kernel / op-priority selection."""
        return (
            f"KernelConfig(moe={self.moe_backend},"
            f" linear={self.linear_backend},"
            f" ir_op={tuple(self.ir_op_priority.rms_norm)})"
        )