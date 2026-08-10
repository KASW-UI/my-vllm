"""Minimal distributed state (C013 → C050).

C013: world_size=1 defaults, all TP ops degrade to identity.
C050: real torch.distributed all_gather/all_reduce when world_size>1.
      Assumes process group initialized externally by launcher (torchrun).
      No GroupCoordinator abstraction — reserved for C083.
"""

import os

import torch


_tp_world_size: int = 1
_tp_rank: int = 0
_pg_initialized: bool = False


def init_distributed(world_size: int = 1, rank: int = 0):
    global _tp_world_size, _tp_rank, _pg_initialized
    _tp_world_size = world_size
    _tp_rank = rank
    if not torch.distributed.is_initialized() and world_size > 1:
        _rank_env = int(os.environ.get("RANK", -1))
        _world_env = int(os.environ.get("WORLD_SIZE", -1))
        if _rank_env >= 0 and _world_env > 1:
            torch.distributed.init_process_group(backend="nccl")
            _pg_initialized = True
            _tp_world_size = torch.distributed.get_world_size()
            _tp_rank = torch.distributed.get_rank()


def get_tp_world_size() -> int:
    return _tp_world_size


def get_tp_rank() -> int:
    return _tp_rank


def all_gather(input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """All-gather along dim.  Identity when world_size=1."""
    if _tp_world_size == 1:
        return input_
    gathered = [torch.empty_like(input_) for _ in range(_tp_world_size)]
    torch.distributed.all_gather(gathered, input_)
    return torch.cat(gathered, dim=dim)


def all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce.  Identity when world_size=1."""
    if _tp_world_size == 1:
        return input_
    torch.distributed.all_reduce(input_)
    return input_


def split_tensor(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Return the local shard of a tensor."""
    if _tp_world_size == 1:
        return tensor
    chunk_size = tensor.shape[dim] // _tp_world_size
    start = _tp_rank * chunk_size
    return tensor.narrow(dim, start, chunk_size)
