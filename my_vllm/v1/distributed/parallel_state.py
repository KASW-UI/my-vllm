"""Minimal distributed state — world_size=1 by default (C013).

When world_size=1, all TP operations degrade to identity.
When world_size>1, NCCL init and collective ops activate.
"""

import torch

_tp_world_size: int = 1
_tp_rank: int = 0


def init_distributed(world_size: int = 1, rank: int = 0):
    global _tp_world_size, _tp_rank
    _tp_world_size = world_size
    _tp_rank = rank


def get_tp_world_size() -> int:
    return _tp_world_size


def get_tp_rank() -> int:
    return _tp_rank


def all_gather(input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """All-gather along dim.  Identity when world_size=1."""
    if _tp_world_size == 1:
        return input_
    # Real TP would use torch.distributed.all_gather
    return input_


def all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce.  Identity when world_size=1."""
    if _tp_world_size == 1:
        return input_
    # Real TP would use torch.distributed.all_reduce
    return input_


def split_tensor(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Return the local shard of a tensor."""
    if _tp_world_size == 1:
        return tensor
    chunk_size = tensor.shape[dim] // _tp_world_size
    start = _tp_rank * chunk_size
    return tensor.narrow(dim, start, chunk_size)
