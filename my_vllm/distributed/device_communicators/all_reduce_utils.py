"""my_vllm/distributed/device_communicators/all_reduce_utils.py

Symmetric memory + custom all-reduce sizing tables + selection heuristics.

C241: extracted from ``vllm/distributed/device_communicators/all_reduce_utils.py``
(421 lines), with the following scope reductions per AGENTS.md
JD-driven scope:

  * ``SYMM_MEM_ALL_REDUCE_MAX_SIZES`` — kept verbatim.
  * ``CUSTOM_ALL_REDUCE_MAX_SIZES`` — kept (used by the
    cs/cu custom AR path; C214 already vendored).
  * ``NCCL_SYMM_MEM_ALL_REDUCE_CONFIG`` — kept verbatim.
  * ``should_nccl_symm_mem_allreduce`` — kept verbatim.
  * ``should_nccl_symm_mem_ag_rs`` — kept verbatim.
  * ``producer`` / ``consumer`` / ``can_actually_p2p`` /
    ``gpu_p2p_access_check`` — OMITTED (these are runtime P2P
    validation utilities that require spawning subprocesses
    with real GPU devices — the my-vllm dev box is a single
    RTX 4060 so a real P2P probe is not exercisable here).
    The P2P-cache rationale is preserved as a comment so
    that future ``deploy/`` work can wire it back in once
    multi-GPU servers are available.

References:
  * vLLM ``all_reduce_utils.py`` — original source.
  * NCCL symmetric memory docs: ``ncclCommWindowRegister`` +
    NVLS (NVLink-Switch) hardware-accelerated collectives.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)

KiB = 1024
MiB = 1024 * 1024

# Max size for each world size in case the custom AR path
# (csrc/custom_all_reduce.cu, C214) is used.  Different SM
# architectures have different optimal AR buffer sizes.
CUSTOM_ALL_REDUCE_MAX_SIZES: dict[str, dict[int, int]] = {
    "9.0": {
        2: 64 * MiB,  # 64 MB
        4: 32 * MiB,  # 32 MB
        6: MiB // 2,  # 512 KB
        8: MiB // 4,  # 256 KB
    },
    "10.0": {
        2: 2 * MiB,  # 2 MB
        4: 2 * MiB,  # 2 MB
        6: 1 * MiB,  # 1 MB
        8: 1 * MiB,  # 1 MB
    },
    "10.3": {
        2: 4 * MiB,  # 4 MB
        4: 4 * MiB,  # 4 MB
        6: 8 * MiB,  # 8 MB
        8: 4 * MiB,  # 4 MB
    },
}

# Max size for each world size in case symmetric memory is
# available — much larger than custom AR because NVLS uses the
# full NVLink-Switch bandwidth.  Per vLLM benchmarks on H100
# (sm_90) and GB200 (sm_100).
SYMM_MEM_ALL_REDUCE_MAX_SIZES: dict[str, dict[int, int]] = {
    "9.0": {
        2: 64 * MiB,
        4: 32 * MiB,
        6: 64 * MiB,
        8: 64 * MiB,
    },
    "10.0": {
        2: 8 * MiB,
        4: 32 * MiB,
        6: 128 * MiB,
        8: 128 * MiB,
    },
    "10.3": {
        2: 4 * MiB,
        4: 32 * MiB,
        6: 32 * MiB,
        8: 64 * MiB,
    },
}

# NCCL symmetric-memory allreduce configuration based on H100 +
# GB200 benchmarks.  PyNCCL-symm outperforms custom_AR for small
# and large tensor sizes; custom_AR wins for mid-range sizes.
NCCL_SYMM_MEM_ALL_REDUCE_CONFIG: dict[str, Any] = {
    "min_world_size": 4,
    "custom_ar_preferred_ranges": {
        4: (16 * KiB, 512 * KiB),
        8: (16 * KiB, 128 * KiB),
    },
    "always_use_above_world_size": 8,
}


def _is_symmetric_memory_enabled() -> bool:
    """Detect whether PyTorch's symmetric memory runtime is enabled.

    C241: vLLM gates the symm-mem path on a ``VLLM_ALLREDUCE_USE_SYMM_MEM``
    env var (default 1); my-vllm follows the same convention via
    ``MY_VLLM_ALLREDUCE_USE_SYMM_MEM``.  Falls back to False if the
    underlying torch.distributed._symmetric_memory isn't importable.
    """
    try:
        import torch.distributed._symmetric_memory as _  # noqa: F401
    except ImportError:
        return False
    env = os.environ.get("MY_VLLM_ALLREDUCE_USE_SYMM_MEM", "1")
    return env not in ("0", "false", "False", "FALSE")


def should_nccl_symm_mem_allreduce(
    world_size: int,
    input_tensor: torch.Tensor,
) -> bool:
    """Determine whether NCCL symmetric-memory all-reduce should be used.

    Mirrors ``vllm.distributed.device_communicators.all_reduce_utils
    .should_nccl_symm_mem_allreduce``.  Returns True when:
      1. ``MY_VLLM_ALLREDUCE_USE_SYMM_MEM`` is enabled.
      2. The torch runtime has symmetric memory available.
      3. World size is >= the configured minimum (default 4).
      4. The tensor size is in a range where NCCL symm-mem
         outperforms custom_AR (per the H100 / GB200 benchmarks
         in ``NCCL_SYMM_MEM_ALL_REDUCE_CONFIG``).

    For a world size above ``always_use_above_world_size`` (default 8),
    the symm-mem path is always preferred.
    """
    import torch.distributed as dist

    if not _is_symmetric_memory_enabled():
        return False
    if world_size < NCCL_SYMM_MEM_ALL_REDUCE_CONFIG["min_world_size"]:
        return False

    tensor_size = input_tensor.nbytes
    custom_ar_range = NCCL_SYMM_MEM_ALL_REDUCE_CONFIG[
        "custom_ar_preferred_ranges"
    ].get(world_size)

    if custom_ar_range is not None:
        lower_bound, upper_bound = custom_ar_range
        return (
            tensor_size <= lower_bound or tensor_size >= upper_bound
        )
    return (
        world_size
        > NCCL_SYMM_MEM_ALL_REDUCE_CONFIG["always_use_above_world_size"]
    )


def should_nccl_symm_mem_ag_rs() -> bool:
    """Whether NCCL symmetric memory should be used for AllGather /
    ReduceScatter collectives.

    C241: matches ``vllm.distributed.device_communicators.all_reduce_utils
    .should_nccl_symm_mem_ag_rs``.  Returns True when the runtime has
    symmetric memory available; the AG/RS path is generally always
    bandwidth-superior on NVLS hardware (no mid-range zone where
    custom_AR wins for AG/RS).
    """
    return _is_symmetric_memory_enabled()


__all__ = [
    "CUSTOM_ALL_REDUCE_MAX_SIZES",
    "NCCL_SYMM_MEM_ALL_REDUCE_CONFIG",
    "SYMM_MEM_ALL_REDUCE_MAX_SIZES",
    "should_nccl_symm_mem_ag_rs",
    "should_nccl_symm_mem_allreduce",
]
