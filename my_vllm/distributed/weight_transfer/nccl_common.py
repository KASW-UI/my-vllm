"""Shared NCCL initialization helpers for weight transfer engines.

C243: closes the A-level "跨节点 Weight Transfer" gap from
``phase3-distributed-runtime-plan.md``.

The dense (:class:`NCCLWeightTransferEngine`) and sparse
(:class:`SparseNCCLWeightTransferEngine`) backends are independent
engines that share *only* their process-group initialization.  That
common logic lives here so the sparse engine does not have to subclass
the dense one.

Scope reductions vs vLLM:

* Uses :func:`torch.distributed.init_process_group` (gloo backend on
  CPU, NCCL backend on GPU) directly instead of
  :class:`vllm.distributed.device_communicators.pynccl.PyNcclCommunicator`
  + :class:`vllm.distributed.utils.StatelessProcessGroup`.  vLLM's
  helper lets you run an isolated NCCL group without disturbing the
  global torch.distributed state — my-vllm's tests are isolated by
  ``torch.multiprocessing.spawn`` already, so the simpler
  ``init_process_group`` is sufficient.
* Drops the ``device=`` parameter plumbing of vLLM's helper; we rely
  on ``torch.cuda.set_device`` having been called by the caller.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from my_vllm.distributed.weight_transfer.base import WeightTransferInitInfo

if TYPE_CHECKING:
    from my_vllm.config.parallel import ParallelConfig


@dataclass
class NCCLWeightTransferInitInfo(WeightTransferInitInfo):
    """Initialization info for NCCL-based weight transfer backends.

    Mirrors :class:`vllm.distributed.weight_transfer.nccl_common.NCCLWeightTransferInitInfo`.
    """

    master_address: str
    master_port: int
    rank_offset: int
    world_size: int


def stateless_init_process_group(
    master_address: str,
    master_port: int,
    rank: int,
    world_size: int,
    backend: str = "gloo",
) -> dist.ProcessGroup:
    """Create a stateless process group via :func:`torch.distributed.init_process_group`.

    The trainer sets ``MASTER_ADDR`` / ``MASTER_PORT`` (or passes them
    explicitly to the worker), then both sides call
    :func:`torch.distributed.init_process_group`.  We use ``gloo`` by
    default because it's available on every platform without a GPU; on
    GPU hosts callers may pass ``"nccl"`` for true GPU-side
    :func:`torch.distributed.broadcast`.

    Args:
        master_address: Trainer rendezvous address (e.g. ``"127.0.0.1"``).
        master_port: Trainer rendezvous port.
        rank: This worker's rank in the trainer-worker group.
        world_size: Total number of trainer + worker ranks.
        backend: ``"gloo"`` (default, cross-platform) or ``"nccl"``
            (GPU-only).

    Returns:
        The newly-created :class:`dist.ProcessGroup` (``dist.group.WORLD``
        after :func:`init_process_group` returns).
    """
    import os

    os.environ["MASTER_ADDR"] = master_address
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    # `init_process_group` is idempotent only if it returns the same
    # group — calling it a second time in the same process raises.  We
    # guard with ``is_initialized`` so the same helper can be called
    # both at worker init (fresh) and at trainer init (after the
    # trainer's own init_process_group).
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )
    return dist.group.WORLD


def worker_init_process_group(
    init_info: NCCLWeightTransferInitInfo,
    parallel_config: "ParallelConfig",
    backend: str = "gloo",
) -> dist.ProcessGroup:
    """Create the trainer↔worker group on an inference worker.

    Computes a unique rank for this worker across all data-parallel
    groups and joins the process group with the trainer.

    Note: my-vllm's :class:`ParallelConfig` doesn't expose
    ``data_parallel_index`` / ``rank`` as properties (vLLM has them via
    engine runtime state); we use a lightweight convention: the worker
    rank is ``init_info.rank_offset`` itself (i.e. rank 0 is the
    trainer, rank ``rank_offset + i`` is the ``i``-th worker).  This is
    sufficient for single-DP deployments; multi-DP workers must add the
    per-DP offset externally before constructing the init info.
    """
    rank = init_info.rank_offset
    return stateless_init_process_group(
        init_info.master_address,
        init_info.master_port,
        rank,
        init_info.world_size,
        backend=backend,
    )


def trainer_init(
    init_info: NCCLWeightTransferInitInfo | dict,
    backend: str = "gloo",
) -> dist.ProcessGroup:
    """Initialize the trainer's process group.

    The trainer is always rank 0 in the trainer-worker process group.

    Args:
        init_info: Either a :class:`NCCLWeightTransferInitInfo` object or
            a dict with keys ``master_address``, ``master_port``,
            ``world_size``.
        backend: ``"gloo"`` (default) or ``"nccl"``.

    Returns:
        The :class:`dist.ProcessGroup` handle (``dist.group.WORLD``).
    """
    if isinstance(init_info, dict):
        master_address = init_info["master_address"]
        master_port = init_info["master_port"]
        world_size = init_info["world_size"]
    else:
        master_address = init_info.master_address
        master_port = init_info.master_port
        world_size = init_info.world_size

    return stateless_init_process_group(
        master_address,
        master_port,
        0,
        world_size,
        backend=backend,
    )
