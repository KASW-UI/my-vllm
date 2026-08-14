"""Quick AllReduce — ROCm-only quantization-accelerated allreduce.

C244: closes the B-level "Quick AllReduce + FlashInfer AllReduce 笔记"
gap from ``phase3-distributed-runtime-plan.md``.

Mirrors ``vllm/distributed/device_communicators/quick_all_reduce.py``
(367 lines → ~140 in my-vllm).

Quick AllReduce (QuickReduce) is AMD's quantization-accelerated
allreduce — a complement to the CUDA-only Custom AllReduce (C214).
It quantizes the operand to INT8/INT6/INT4/INT3 (with a runtime-
selectable codec) before the ring-reduce-scatter, then dequantizes
after the all-gather.  Quantization reduces the bytes-per-rank on
the wire by 2× (INT8) to ~5× (INT3), at the cost of small numerical
drift.

Scope reductions vs vLLM:

* **No C extension vendored.** vLLM ships a CUDA/ROCm C kernel at
  ``vllm/csrc/quickreduce`` (``init_custom_qr`` / ``qr_get_handle`` /
  ``qr_open_handles`` / ``qr_all_reduce`` / ``qr_destroy``) — my-vllm
  keeps the Python surface but every codepath returns ``disabled=True``
  on a CUDA box (the ops aren't built, and QuickReduce is
  ROCm-MI300-only anyway).
* **No real ROCm MI300 detection.** We can't run on ROCm from the dev
  box, so ``_rocm_arch_available()`` always returns ``False``.
* **Quantization min-size table preserved verbatim.** vLLM's
  ``_QR_MIN_SIZE`` table is empirically tuned and useful as a
  reference even when the kernel isn't available.

This file therefore ships:

* ``QuickReduceRegime`` enum (FP / INT8 / INT6 / INT4 / INT3 / NONE).
* ``_QR_MIN_SIZE`` table for FP16/BF16 × world_size 2/4/8.
* ``QuickAllReduce`` skeleton: lifecycle (``__init__`` /
  ``init_quick_all_reduce`` / ``create_shared_buffer`` /
  ``close`` / ``__del__``), heuristic (``should_quick_allreduce``),
  and the call surface (``quick_all_reduce``).
"""

from enum import Enum
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


class QuickReduceRegime(Enum):
    """Quantization regime for the QuickReduce kernel.

    Mirrors vLLM ``QuickReduceRegime`` (``vllm/distributed/device_communicators/
    quick_all_reduce.py:30``); the integer ids are kept aligned with the C
    header ``csrc/quickreduce/quick_reduce.h`` so the same ``_QR_MIN_SIZE``
    table indexes correctly on either side of the FFI.
    """

    FP = 0
    INT8 = 1
    INT6 = 2
    INT4 = 3
    INT3 = 4
    NONE = 5


KB = 1024
MB = 1024 * KB


# Try the C extension at import time; my-vllm doesn't build it, so
# ``quick_ar`` is always False.  Kept as a module-level constant so
# downstream code can check it.
quick_ar: bool = False
"""Whether the QuickReduce C extension is available.  Always False in
my-vllm (no ``csrc/quickreduce`` vendored)."""


class QuickAllReduce:
    """ROCm-MI300-only quantization-accelerated allreduce.

    Complements the CUDA CustomAllReduce (C214).  QuickReduce
    quantizes the operand to a configurable integer codec (INT8 /
    INT6 / INT4 / INT3) before the ring-reduce-scatter, then
    dequantizes after the all-gather — trading small numerical drift
    for 2×–5× wire-byte reduction.

    Lifecycle in my-vllm:

    * ``__init__`` always sets ``self.disabled = True`` because the C
      extension isn't vendored and ROCm MI300 isn't available on the
      dev box.  All other paths (``should_quick_allreduce``,
      ``quick_all_reduce``) short-circuit on this flag.
    * ``close()`` is a no-op when disabled; in vLLM it calls
      ``ops.qr_destroy(self._ptr)``.

    vLLM contract (preserved verbatim):

    * Supported world sizes: 2, 4, 8 (per ``_SUPPORTED_WORLD_SIZES``).
    * Supported dtypes: float16, bfloat16 (per ``_SUPPORTED_DTYPES``).
    * ROCm MI300 series only (gfx94 / gfx95 GCN arch names).
    * Multi-node groups skip QuickReduce (no NVLink + ROCm combination
      in practice; the kernel is designed for fully-connected same-node).
    * World size > 2 requires NVLink (``is_fully_connected``) — PCIe-only
      configurations are skipped.
    * INT3 restricted to TP2 (world_size == 2) due to poor TP4/TP8
      kernel performance.
    """

    _SUPPORTED_WORLD_SIZES = [2, 4, 8]
    _SUPPORTED_DTYPES = [torch.float16, torch.bfloat16]
    # Per-dtype × world_size threshold for each quantization regime.
    # Order: [FP, INT8, INT6, INT4, INT3].
    # Empirical data from vLLM kernel benchmarks; preserved for reference.
    _QR_MIN_SIZE = {
        (torch.float16, 2): [1 * MB, 2 * MB, 2 * MB, 1 * MB, 1 * MB],
        (torch.float16, 4): [1 * MB, 16 * MB, 4 * MB, 2 * MB, 2 * MB],
        (torch.float16, 8): [16 * MB, 4 * MB, 4 * MB, 2 * MB, 2 * MB],
        (torch.bfloat16, 2): [2 * MB, 8 * MB, 8 * MB, 8 * MB, 8 * MB],
        (torch.bfloat16, 4): [8 * MB, 64 * MB, 64 * MB, 16 * MB, 16 * MB],
        (torch.bfloat16, 8): [
            16 * MB,
            2048 * MB,
            2048 * MB,
            2048 * MB,
            2048 * MB,
        ],
    }

    def __init__(
        self,
        group: "ProcessGroup",
        device: int | str | torch.device,
    ) -> None:
        """Initialize the QuickAllReduce wrapper.

        In my-vllm this always sets ``self.disabled = True`` and
        returns without touching the C extension.  vLLM's full init
        does the following checks (preserved as comments for
        reference):

        1. ROCm MI300 GCN arch detection (``_rocm_arch_available``).
        2. C extension availability (``quick_ar`` flag).
        3. Non-NCCL group (QuickReduce is layered on top of a
           ``GlooCommunicator``-style group, not NCCL).
        4. Single-node (cross-node spans aren't supported).
        5. ``world_size`` in ``[2, 4, 8]``.
        6. ``world_size > 2`` requires fully-connected GPUs (NVLink
           detection via ``current_platform.is_fully_connected``).

        Args:
            group: Process group (in my-vllm: not used; in vLLM: must
                be a non-NCCL single-node group of size 2/4/8).
            device: Device to bind (in my-vllm: not used; in vLLM:
                per-rank GPU ordinal).
        """
        self.disabled = True
        self.group = group
        self.device = device if isinstance(device, torch.device) else torch.device(device)

    # ------------------------------------------------------------------
    # vLLM-compatible lifecycle (all disabled in my-vllm)
    # ------------------------------------------------------------------

    def init_quick_all_reduce(self) -> None:
        """Resolve the quantization regime from environment variables.

        vLLM reads ``VLLM_ROCM_QUICK_REDUCE_QUANTIZATION`` to choose
        ``FP`` / ``INT8`` / ``INT6`` / ``INT4`` / ``INT3`` / ``NONE``,
        enforces the INT3-restricted-to-TP2 rule, then calls
        ``ops.init_custom_qr(rank, world_size, qr_max_size)``.

        In my-vllm this is a no-op (the kernel isn't vendored).
        """
        # No-op; preserved for API compatibility.
        self.disabled = True

    def create_shared_buffer(self) -> None:
        """Exchange QuickReduce handles across the group.

        vLLM: ``handle = ops.qr_get_handle(self._ptr)`` →
        ``dist.all_gather_object(handles, handle)`` →
        ``ops.qr_open_handles(self._ptr, handles)``.

        In my-vllm this is a no-op.
        """
        # No-op; preserved for API compatibility.

    def should_quick_allreduce(self, inp: torch.Tensor) -> bool:
        """Return True if QuickReduce should be used for ``inp``.

        Returns False unconditionally in my-vllm.  vLLM's heuristic
        checks:

        * ``not self.disabled``
        * ``inp.dtype in {fp16, bf16}``
        * ``inp.numel() * inp.element_size()`` is a multiple of 16
        * ``is_weak_contiguous(inp)``
        * ``min_size <= byte_size <= max_size`` from the
          ``_QR_MIN_SIZE`` table indexed by ``(dtype, world_size)``.

        Args:
            inp: Candidate tensor (not inspected in my-vllm).

        Returns:
            Always False — QuickReduce isn't available on this build.
        """
        return False

    def quick_all_reduce(
        self,
        inp: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run a QuickReduce allreduce.

        vLLM: ``ops.qr_all_reduce(self._ptr, inp, out, quant_level,
        use_fp16_kernels)``.

        In my-vllm raises ``RuntimeError`` to make accidental use loud
        — the kernel isn't built.
        """
        raise RuntimeError(
            "QuickAllReduce is not available in my-vllm (no C extension "
            "vendored; ROCm MI300 only)."
        )

    def close(self) -> None:
        """Destroy the QuickReduce instance.

        vLLM: ``ops.qr_destroy(self._ptr)``.  In my-vllm: no-op.
        """
        # No-op; preserved for API compatibility.
        self.disabled = True

    def __del__(self) -> None:
        """Destructor — closes the underlying kernel handle.

        In vLLM delegates to ``self.close()``.  In my-vllm it's a
        no-op (no kernel handle to release).
        """
        try:
            self.close()
        except Exception:
            # Never raise from __del__.
            pass


# ----------------------------------------------------------------------
# Helpers preserved verbatim from vLLM (env-var resolution shims).
# ----------------------------------------------------------------------


def _get_qr_min_size(qr_max_size: int | None) -> int | None:
    """Read ``VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB`` env var.

    vLLM uses this as a runtime override for the ``_QR_MIN_SIZE``
    table; preserved as a no-op stub in my-vllm (no environment
    variables honored without the C extension).
    """
    return None


def _get_qr_quantization_min_size() -> int | None:
    """Read ``VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB`` env var.

    vLLM uses this to force the FP regime below a per-message byte
    threshold; preserved as a no-op stub in my-vllm.
    """
    return None
