"""Distributed parallel state — aligns with vLLM vllm/distributed/parallel_state.py.

C013: world_size=1 defaults, all TP ops degrade to identity.
C050: real torch.distributed all_gather/all_reduce when world_size>1.
C083: GroupCoordinator + initialize_model_parallel with NCCL process groups.
"""

import os

import torch
import torch.distributed as dist


_TP: "GroupCoordinator | None" = None
_PP: "GroupCoordinator | None" = None


class GroupCoordinator:
    """Wraps a PyTorch ProcessGroup for collective communication.

    Aligns with vLLM GroupCoordinator — minimal education version:
    creates NCCL device group + Gloo CPU group for the given ranks.
    """

    def __init__(
        self,
        group_ranks: list[list[int]],
        local_rank: int,
        backend: str,
        group_name: str = "anonymous",
    ) -> None:
        self.rank = dist.get_rank()
        self.local_rank = local_rank
        self.group_name = group_name

        self_device_group = None
        self_cpu_group = None

        for ranks in group_ranks:
            device_group = dist.new_group(ranks, backend=backend)
            cpu_group = dist.new_group(ranks, backend="gloo")
            if self.rank in ranks:
                self.ranks = ranks
                self.world_size = len(ranks)
                self.rank_in_group = ranks.index(self.rank)
                self_device_group = device_group
                self_cpu_group = cpu_group

        assert self_device_group is not None
        assert self_cpu_group is not None
        self.device_group = self_device_group
        self.cpu_group = self_cpu_group
        self.device = torch.device(f"cuda:{local_rank}")

    @property
    def first_rank(self) -> int:
        return self.ranks[0]

    @property
    def last_rank(self) -> int:
        return self.ranks[-1]

    @property
    def is_first_rank(self) -> bool:
        return self.rank == self.first_rank

    @property
    def is_last_rank(self) -> bool:
        return self.rank == self.last_rank

    @property
    def next_rank(self) -> int:
        return self.ranks[(self.rank_in_group + 1) % self.world_size]

    @property
    def prev_rank(self) -> int:
        return self.ranks[(self.rank_in_group - 1) % self.world_size]

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        dist.all_reduce(input_, group=self.device_group)
        return input_

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        gathered = [torch.empty_like(input_) for _ in range(self.world_size)]
        dist.all_gather(gathered, input_, group=self.device_group)
        return torch.cat(gathered, dim=dim)

    def broadcast(self, input_: torch.Tensor, src: int = 0) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        dist.broadcast(input_, src=self.ranks[src], group=self.device_group)
        return input_

    def broadcast_object(self, obj: object = None, src: int = 0) -> object:
        """Broadcast a Python object via CPU group."""
        if self.world_size == 1:
            return obj
        if self.rank_in_group == src:
            dist.broadcast_object_list([obj], src=self.ranks[src], group=self.cpu_group)
            return obj
        recv = [None]
        dist.broadcast_object_list(recv, src=self.ranks[src], group=self.cpu_group)
        return recv[0]

    def barrier(self) -> None:
        dist.barrier(group=self.cpu_group)

    def send(self, tensor: torch.Tensor, dst: int) -> None:
        """NCCL p2p send."""
        if self.world_size <= 1:
            return
        dist.send(tensor, dst=self.ranks[dst], group=self.device_group)

    def recv(
        self, size: torch.Size, dtype: torch.dtype, src: int
    ) -> torch.Tensor:
        """NCCL p2p recv."""
        if self.world_size <= 1:
            return torch.empty(size, dtype=dtype, device=self.device)
        tensor = torch.empty(size, dtype=dtype, device=self.device)
        dist.recv(tensor, src=self.ranks[src], group=self.device_group)
        return tensor

    def destroy(self) -> None:
        if hasattr(self, "device_group"):
            dist.destroy_process_group(self.device_group)
            del self.device_group
        if hasattr(self, "cpu_group"):
            dist.destroy_process_group(self.cpu_group)
            del self.cpu_group


def init_model_parallel_group(
    group_ranks: list[list[int]],
    local_rank: int,
    backend: str,
    group_name: str = "anonymous",
) -> GroupCoordinator:
    """Factory for GroupCoordinator — aligns with vLLM init_model_parallel_group."""
    return GroupCoordinator(
        group_ranks=group_ranks,
        local_rank=local_rank,
        backend=backend,
        group_name=group_name,
    )


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    backend: str = "nccl",
) -> None:
    """Initialize model parallel groups — aligns with vLLM initialize_model_parallel.

    C083: explicit init_process_group + TP GroupCoordinator creation.
    """
    global _TP
    assert _TP is None, "tensor model parallel group is already initialized"

    if not dist.is_initialized():
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    all_ranks = list(range(dist.get_world_size()))
    group_ranks = [all_ranks[i:i + tensor_model_parallel_size]
                    for i in range(0, len(all_ranks), tensor_model_parallel_size)]
    group_ranks = [gr for gr in group_ranks if len(gr) == tensor_model_parallel_size]

    _TP = init_model_parallel_group(
        group_ranks=group_ranks,
        local_rank=local_rank,
        backend=backend,
        group_name="tp",
    )


def get_tp_group() -> GroupCoordinator:
    assert _TP is not None, "tensor model parallel group is not initialized"
    return _TP


def get_pp_group() -> "GroupCoordinator | None":
    return _PP


def get_tp_world_size() -> int:
    if _TP is None:
        return 1
    return _TP.world_size


def get_tp_rank() -> int:
    if _TP is None:
        return 0
    return _TP.rank_in_group


def model_parallel_is_initialized() -> bool:
    return _TP is not None


def destroy_model_parallel() -> None:
    global _TP
    if _TP is not None:
        _TP.destroy()
        _TP = None
    if dist.is_initialized():
        dist.destroy_process_group()


# --- C050 backward-compat wrappers (used by linear.py + tests) ---

def init_distributed(world_size: int = 1, rank: int = 0):
    """Backward-compat: config world_size/rank without spawning processes.

    world_size=1: no-op (no NCCL needed).
    world_size>1: if not already initialized, set up the TP process group.
    """
    global _TP
    if world_size <= 1:
        return
    if not dist.is_initialized():
        os.environ.setdefault("RANK", str(rank))
        os.environ.setdefault("WORLD_SIZE", str(world_size))
    initialize_model_parallel(tensor_model_parallel_size=world_size)


# --- C050 backward-compat wrappers (used by linear.py) ---

def all_gather(input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if _TP is None:
        return input_
    return _TP.all_gather(input_, dim)


def all_reduce(input_: torch.Tensor) -> torch.Tensor:
    if _TP is None:
        return input_
    return _TP.all_reduce(input_)


def split_tensor(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    tp_size = get_tp_world_size() if _TP is not None else 1
    tp_rank = get_tp_rank() if _TP is not None else 0
    if tp_size == 1:
        return tensor
    chunk_size = tensor.shape[dim] // tp_size
    start = tp_rank * chunk_size
    return tensor.narrow(dim, start, chunk_size)
