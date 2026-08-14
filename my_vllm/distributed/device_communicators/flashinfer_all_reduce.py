"""FlashInfer AllReduce — flashinfer.comm-backed allreduce + norm/rms/quant fusion.

C244: closes the B-level "Quick AllReduce + FlashInfer AllReduce 笔记"
gap from ``phase3-distributed-runtime-plan.md``.

Mirrors ``vllm/distributed/device_communicators/flashinfer_all_reduce.py``
(381 lines → ~110 in my-vllm).

FlashInfer AllReduce (FlashInfer-AR) is a third-party allreduce library
shipped by FlashInfer (``flashinfer.comm``).  It provides:

* ``mnnvl`` backend — NVSwitch-multicast allreduce on multi-GPU nodes
  (lowest latency, requires NVSwitch hardware).
* ``trtllm`` backend — TensorRT-LLM's fused allreduce (single-node
  fallback when multicast is unavailable).
* Fused allreduce + RMSNorm / quantization patterns via
  ``AllReduceFusionPattern`` (``kAllReduce``,
  ``kARResidualRMSNorm``, ``kARResidualRMSNormQuant``).

The fusion paths let the allreduce write directly into the RMSNorm
output buffer, skipping an intermediate copy.  This is a meaningful
optimization for transformer prefill/decode where every layer ends
with ``x = residual + x; x = rms_norm(x)`` and the allreduce feeds the
residual.

Scope reductions vs vLLM:

* **flashinfer not vendored.** The ``import flashinfer.comm`` block
  sets ``fi_ar_available = False``; every codepath returns disabled.
* **No workspace allocation.** The ``get_fi_ar_workspace`` /
  ``get_fi_ar_quant_workspace`` functions are preserved as stubs that
  return ``None`` (matching vLLM's "creation failed" path).
* **No NVSwitch / CUDA detection.** vLLM's ``current_platform.is_cuda()``
  and ``get_node_count()`` calls aren't reached — the placeholder
  returns ``disabled=True`` before touching them.

This file ships:

* ``fi_ar_available`` flag (always False).
* ``FlashInferAllReduce`` skeleton with the lifecycle
  (``__init__`` / ``_ensure_workspace`` / ``should_use_fi_ar`` /
  ``all_reduce`` / ``destroy``).
* Module-level workspace globals (``_fi_ar_workspace``,
  ``_fi_ar_quant_workspace``) preserved as ``None`` for API shape.
"""

import atexit
import threading
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


# PDL = "Programmatic Dependent Launch"; FlashInfer's launch heuristic
# uses 16 tokens as the threshold below which the launch is conservative
# (waits for completion at the end of the kernel) vs above which it
# issues the next launch early to overlap.
PDL_ADVANCE_LAUNCH_TOKENS = 16

# flashinfer.comm is optional — without it, every FlashInfer-AR code
# path returns disabled.  In my-vllm the import fails by default, so
# ``fi_ar_available`` is False.
fi_ar_available: bool = False
"""Whether ``flashinfer.comm`` is importable and exposes
``allreduce_fusion``.  Always False in my-vllm (no flashinfer vendored)."""


# Workspace handles — preserved as module-level globals matching vLLM
# shape, but always None in my-vllm.
_fi_ar_workspace = None
_fi_ar_quant_workspace = None
_fi_ar_workspace_lock = threading.Lock()


def _create_workspace(
    backend: str,
    world_size: int,
    rank: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
    group: "ProcessGroup",
):
    """Create a FlashInfer AllReduce workspace.

    vLLM: ``flashinfer_comm.create_allreduce_fusion_workspace(backend,
    world_size, rank, max_token_num, hidden_dim, dtype, comm_backend,
    group)``.

    In my-vllm: returns ``None`` — the flashinfer library isn't
    vendored.
    """
    return None


def _resolve_fi_ar_backend() -> tuple[str, bool]:
    """Resolve the FlashInfer allreduce backend.

    vLLM reads ``VLLM_FLASHINFER_ALLREDUCE_BACKEND`` env var
    (``"auto"`` / ``"mnnvl"`` / ``"trtllm"``).  ``"auto"`` defaults
    to ``"mnnvl"`` for single- and multi-node setups (FlashInfer
    >= 0.6.12 fixed the single-node cudagraph hang), with a
    ``"trtllm"`` fallback for single-node topologies that lack
    NVSwitch multicast support.

    In my-vllm: returns ``("none", False)`` — neither backend is
    available.
    """
    return "none", False


def get_fi_ar_workspace(
    world_size: int,
    rank: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
    group: "ProcessGroup",
):
    """Return the allreduce workspace for non-quant patterns.

    vLLM: lazy-initializes and caches a module-level ``_fi_ar_workspace``
    using ``_create_workspace`` with the resolved backend.

    In my-vllm: returns ``None``.
    """
    return None


def get_fi_ar_quant_workspace(
    world_size: int,
    rank: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
    group: "ProcessGroup",
):
    """Return the allreduce workspace for quant-fusion patterns.

    vLLM: lazy-initializes and caches ``_fi_ar_quant_workspace``.

    In my-vllm: returns ``None``.
    """
    return None


def destroy_fi_ar_workspace() -> None:
    """Destroy both workspaces.

    vLLM: called by ``atexit`` + explicit ``FlashInferAllReduce.destroy``.
    Each workspace's ``destroy()`` releases the flashinfer-allocated
    GPU memory.

    In my-vllm: no-op.
    """
    global _fi_ar_workspace, _fi_ar_quant_workspace
    with _fi_ar_workspace_lock:
        _fi_ar_workspace = None
        _fi_ar_quant_workspace = None


# Register the destroy hook — preserved from vLLM even though the
# workspaces are always None.
atexit.register(destroy_fi_ar_workspace)


class FlashInferAllReduce:
    """FlashInfer-backed allreduce + AR/RMSNorm/quant fusion.

    Complements the CUDA CustomAllReduce (C214) and the ROCm
    QuickAllReduce (C244).  FlashInfer-AR is the third option in the
    dispatch chain when both are unavailable or when the fusion
    patterns (``AllReduceFusionPattern.kARResidualRMSNorm``,
    ``kARResidualRMSNormQuant``) are eligible.

    In my-vllm this is a skeleton — ``disabled = True`` always,
    ``all_reduce`` raises ``RuntimeError``.
    """

    def __init__(
        self,
        group: "ProcessGroup",
        device: int | str | torch.device,
    ) -> None:
        """Initialize the FlashInfer AllReduce wrapper.

        vLLM's full init checks:

        1. ``fi_ar_available`` — flashinfer installed.
        2. ``current_platform.is_cuda()`` — CUDA only.
        3. ``world_size != 1`` — no single-rank case.
        4. World-size-specific ``max_workspace_size`` from
           ``PassConfig.default_fi_allreduce_fusion_max_size_mb()``.

        In my-vllm: ``self.disabled = True`` unconditionally.
        """
        self.disabled = True
        self.group = group
        self.world_size = dist.get_world_size(self.group) if dist.is_initialized() else 1
        self.rank = dist.get_rank(self.group) if dist.is_initialized() else 0
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.max_workspace_size = 0
        self.max_num_tokens = 0

    # ------------------------------------------------------------------
    # vLLM-compatible lifecycle (all disabled in my-vllm)
    # ------------------------------------------------------------------

    def _ensure_workspace(self, hidden_dim: int, dtype: torch.dtype) -> bool:
        """Ensure the allreduce workspace is initialized.

        vLLM: computes ``max_num_tokens = max_workspace_size //
        (hidden_dim * element_size)`` and calls ``get_fi_ar_workspace``.

        In my-vllm: returns ``False`` and sets ``self.disabled = True``.
        """
        self.disabled = True
        return False

    def should_use_fi_ar(self, input_tensor: torch.Tensor) -> bool:
        """Return True if FlashInfer-AR should be used for ``input_tensor``.

        vLLM's heuristic checks:

        * ``not self.disabled``
        * ``input_tensor.is_cuda``
        * ``input_tensor.is_contiguous()``
        * ``len(input_tensor.shape) == 2`` (token-major 2D only).
        * ``num_tokens <= self.max_num_tokens``
        * Workspace creation succeeds (``_ensure_workspace``).

        In my-vllm: returns False unconditionally.
        """
        return False

    def all_reduce(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run a FlashInfer AllReduce.

        vLLM: ``flashinfer_comm.allreduce_fusion(input, workspace,
        pattern, launch_with_pdl=True, trigger_completion_at_end=
        num_tokens > PDL_ADVANCE_LAUNCH_TOKENS)``.

        In my-vllm: raises ``RuntimeError`` — the library isn't
        vendored.
        """
        raise RuntimeError(
            "FlashInferAllReduce is not available in my-vllm "
            "(flashinfer.comm not vendored)."
        )

    def destroy(self) -> None:
        """Destroy the FlashInfer AllReduce workspace.

        vLLM: ``destroy_fi_ar_workspace()`` — releases the module-level
        workspace handles (which may be aliased across quant / non-quant).

        In my-vllm: no-op (no workspace allocated).
        """
        # No-op; preserved for API compatibility.
        pass
