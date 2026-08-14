"""Distributed parallel state — aligns with vLLM vllm/distributed/parallel_state.py.

C013: world_size=1 defaults, all TP ops degrade to identity.
C050: real torch.distributed all_gather/all_reduce when world_size>1.
C083: GroupCoordinator + initialize_model_parallel with NCCL process groups.
C150: canonical `init_distributed_environment(world_size, rank,
distributed_init_method, local_rank, backend)` entry point that
matches vLLM's. Spawned workers call this from `Worker
.init_device()` to set up NCCL before TP groups are created.
"""

import os

import torch
import torch.distributed as dist


_TP: "GroupCoordinator | None" = None
_PP: "GroupCoordinator | None" = None
_DP: "GroupCoordinator | None" = None
_CP: "GroupCoordinator | None" = None
_EP: "GroupCoordinator | None" = None


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

    def all_gatherv(
        self,
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Variable-size AllGather along ``dim``.

        C240: used by ``AgRsAll2AllManager.dispatch``.  When ``sizes``
        is uniform across ranks (or ``None``), falls through to the
        regular all_gather for a faster path; otherwise uses a
        gather-then-broadcast pattern (gather on rank 0, broadcast
        to all).  This is correct but not bandwidth-optimal — true
        all_gatherv uses NCCL ``ncclAllGatherv`` which my-vllm does
        not vendor at the C-layer; see Phase 5 plan for future
        acceleration work.

        ``input_`` may be a single tensor or a list of tensors
        (each gathered along the same ``dim``).
        """
        if self.world_size == 1:
            return input_

        if isinstance(input_, torch.Tensor):
            tensors = [input_]
        else:
            tensors = list(input_)

        if sizes is not None and all(s == sizes[0] for s in sizes):
            sizes = None

        results: list[torch.Tensor] = []
        for tensor in tensors:
            results.append(self._all_gatherv_single(tensor, dim, sizes))

        if isinstance(input_, torch.Tensor):
            return results[0]
        return results

    def _all_gatherv_single(
        self,
        input_: torch.Tensor,
        dim: int,
        sizes: list[int] | None,
    ) -> torch.Tensor:
        """Helper for a single-tensor AllGather."""
        if sizes is None:
            return self.all_gather(input_, dim=dim)

        assert len(sizes) == self.world_size, (
            f"len(sizes)={len(sizes)} != world_size={self.world_size}"
        )
        assert input_.size(dim) == sizes[self.rank_in_group], (
            f"local dim-{dim} size {input_.size(dim)} != "
            f"sizes[rank={self.rank_in_group}]={sizes[self.rank_in_group]}"
        )

        if dim == 0:
            return self._all_gatherv_dim0(input_, sizes)
        return self._all_gatherv_perm(input_, dim, sizes)

    def _all_gatherv_dim0(
        self, input_: torch.Tensor, sizes: list[int]
    ) -> torch.Tensor:
        out_size = (sum(sizes),) + tuple(input_.size()[1:])
        output = torch.empty(out_size, dtype=input_.dtype, device=input_.device)

        my_offset = sum(sizes[: self.rank_in_group])
        my_size = sizes[self.rank_in_group]
        output.narrow(0, my_offset, my_size).copy_(input_.contiguous())

        if self.rank_in_group == 0:
            for r in range(1, self.world_size):
                offset = sum(sizes[:r])
                recv_size = (sizes[r],) + tuple(input_.size()[1:])
                recv = torch.empty(recv_size, dtype=input_.dtype, device=input_.device)
                dist.recv(recv, src=self.ranks[r], group=self.device_group)
                output.narrow(0, offset, sizes[r]).copy_(recv)
        else:
            dist.send(
                input_.contiguous(),
                dst=self.ranks[0],
                group=self.device_group,
            )

        dist.broadcast(output, src=self.ranks[0], group=self.device_group)
        return output

    def _all_gatherv_perm(
        self, input_: torch.Tensor, dim: int, sizes: list[int]
    ) -> torch.Tensor:
        perm = input_.movedim(dim, 0).contiguous()
        out_perm = self._all_gatherv_dim0(perm, sizes)
        return out_perm.movedim(0, dim).contiguous()

    def reduce_scatterv(
        self,
        input_: torch.Tensor,
        dim: int = 0,
        sizes: list[int] | None = None,
    ) -> torch.Tensor:
        """Variable-size ReduceScatter (sum) along ``dim``.

        C240: used by ``AgRsAll2AllManager.combine``.  When ``sizes``
        is uniform across ranks (or ``None``), uses
        ``torch.distributed.reduce_scatter_tensor`` (NCCL
        ``ncclReduceScatter``).  Otherwise uses a gather-then-sum-
        then-scatter pattern via rank 0 (correct, not
        bandwidth-optimal — see Phase 5 plan).
        """
        if self.world_size == 1:
            return input_

        if dim < 0:
            dim += input_.dim()
        if sizes is not None and all(s == sizes[0] for s in sizes):
            sizes = None

        if sizes is None:
            return self._reduce_scatter_uniform(input_, dim)

        assert len(sizes) == self.world_size, (
            f"len(sizes)={len(sizes)} != world_size={self.world_size}"
        )
        assert input_.size(dim) == sum(sizes), (
            f"input dim-{dim} size {input_.size(dim)} != sum(sizes)={sum(sizes)}"
        )
        return self._reduce_scatterv_via_root(input_, dim, sizes)

    def _reduce_scatterv_via_root(
        self, input_: torch.Tensor, dim: int, sizes: list[int]
    ) -> torch.Tensor:
        chunk_size = sizes[self.rank_in_group]
        out_shape = (chunk_size,) + tuple(input_.size()[1:])
        if dim == 0:
            return self._reduce_scatterv_dim0(input_, sizes, out_shape)
        perm = input_.movedim(dim, 0).contiguous()
        perm_sizes = [s for s in sizes]
        perm_chunk_size = chunk_size
        perm_out_shape = (perm_chunk_size,) + tuple(perm.size()[1:])
        out_perm = self._reduce_scatterv_dim0(perm, perm_sizes, perm_out_shape)
        return out_perm.movedim(0, dim).contiguous()

    def _reduce_scatterv_dim0(
        self,
        input_: torch.Tensor,
        sizes: list[int],
        out_shape: torch.Size,
    ) -> torch.Tensor:
        if self.rank_in_group == 0:
            accumulator = input_.clone().contiguous()
            for r in range(1, self.world_size):
                recv = torch.empty_like(input_)
                dist.recv(recv, src=self.ranks[r], group=self.device_group)
                accumulator.add_(recv)
            output = accumulator.narrow(0, 0, sizes[0]).clone()
        else:
            dist.send(input_.contiguous(), dst=self.ranks[0], group=self.device_group)
            output = torch.empty(out_shape, dtype=input_.dtype, device=input_.device)

        dist.broadcast(output, src=self.ranks[0], group=self.device_group)
        return output

    def _reduce_scatter_uniform(
        self, input_: torch.Tensor, dim: int
    ) -> torch.Tensor:
        """Uniform-size reduce-scatter (sum) via torch primitive.

        Uses ``torch.distributed.reduce_scatter_tensor`` which maps
        to NCCL ``ncclReduceScatter`` on supported hardware.  This
        is the fast path: it materialises only the rank-local
        output slice.

        The ``input_`` is the WORLD-WIDE tensor (each rank holds
        the full ``world_size * chunk`` tensor); the output is the
        rank-local slice of shape ``(chunk, ...)``.
        """
        if dim < 0:
            dim += input_.dim()
        assert input_.size(dim) % self.world_size == 0, (
            f"input dim-{dim} size {input_.size(dim)} not divisible "
            f"by world_size={self.world_size}"
        )
        chunk = input_.size(dim) // self.world_size
        out_shape = (chunk,) + tuple(input_.size()[1:])

        if dim == 0:
            output = torch.empty(out_shape, dtype=input_.dtype, device=input_.device)
            dist.reduce_scatter_tensor(
                output, input_.contiguous(),
                op=dist.ReduceOp.SUM,
                group=self.device_group,
            )
            return output
        else:
            perm = input_.movedim(dim, 0).contiguous()
            out_perm = torch.empty(
                (chunk,) + tuple(perm.size()[1:]),
                dtype=input_.dtype,
                device=input_.device,
            )
            dist.reduce_scatter_tensor(
                out_perm, perm,
                op=dist.ReduceOp.SUM,
                group=self.device_group,
            )
            return out_perm.movedim(0, dim).contiguous()

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """C240 façade for EP/DP dispatch via router_logits.

        In world_size=1 (the my-vllm default), this is identity.  In
        a real EP deployment, this routes through ``DeviceCommunicator
        .dispatch_router_logits`` → ``All2AllManagerBase
        .dispatch_router_logits`` (e.g. ``AgRsAll2AllManager``).
        """
        if self.world_size == 1:
            if extra_tensors is not None:
                return hidden_states, router_logits, extra_tensors
            return hidden_states, router_logits
        device_communicator = getattr(self, "device_communicator", None)
        if device_communicator is not None:
            return device_communicator.dispatch_router_logits(
                hidden_states, router_logits, is_sequence_parallel, extra_tensors
            )
        return hidden_states, router_logits

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """C240 façade for EP/DP dispatch via topk routing.

        World_size=1: identity.  World_size>1: routes through
        ``DeviceCommunicator.dispatch`` → ``All2AllManagerBase
        .dispatch`` (e.g. ``AgRsAll2AllManager.dispatch``).
        """
        if self.world_size == 1:
            if extra_tensors is not None:
                return hidden_states, topk_weights, topk_ids, extra_tensors
            return hidden_states, topk_weights, topk_ids
        device_communicator = getattr(self, "device_communicator", None)
        if device_communicator is not None:
            return device_communicator.dispatch(
                hidden_states, topk_weights, topk_ids,
                is_sequence_parallel, extra_tensors,
            )
        return hidden_states, topk_weights, topk_ids

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        """C240 façade for EP/DP combine (reduce-scatter).

        World_size=1: identity.  World_size>1: routes through
        ``DeviceCommunicator.combine`` → ``All2AllManagerBase
        .combine`` (e.g. ``AgRsAll2AllManager.combine``).
        """
        if self.world_size == 1:
            return hidden_states
        device_communicator = getattr(self, "device_communicator", None)
        if device_communicator is not None:
            return device_communicator.combine(hidden_states, is_sequence_parallel)
        return hidden_states

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


def init_distributed_environment(
    world_size: int = -1,
    rank: int = -1,
    distributed_init_method: str = "env://",
    local_rank: int = -1,
    backend: str = "nccl",
) -> None:
    """Initialise the torch.distributed process group for NCCL.

    Mirrors vLLM V1 distributed/parallel_state.py:
    init_distributed_environment. C150: real implementation (was
    a no-op stub in C083). Spawned workers from
    `MultiprocExecutor._init_executor` call this as the first
    step in `Worker.init_device()` before the model is loaded.

    Reads `WORLD_SIZE` / `RANK` / `LOCAL_RANK` from env vars
    when the corresponding arg is `-1` (the vLLM convention).
    For `env://` init method, `MASTER_ADDR` / `MASTER_PORT`
    must also be set — `MultiprocExecutor._init_executor` does
    that when it spawns workers.
    """
    if dist.is_initialized():
        return

    if world_size == -1:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if rank == -1:
        rank = int(os.environ.get("RANK", "0"))
    if local_rank == -1:
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

    # NCCL requires GPU. Fall back to gloo for the rare
    # CPU-only test path (mirrors vLLM's runtime fallback).
    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"

    dist.init_process_group(
        backend=backend,
        init_method=distributed_init_method,
        world_size=world_size,
        rank=rank,
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


def get_dp_group() -> "GroupCoordinator | None":
    """Return the data-parallel group, or None if not initialized.

    Mirrors vLLM V1 distributed/parallel_state.py:get_dp_group.
    """
    return _DP


def get_ep_group() -> "GroupCoordinator | None":
    """Return the expert-parallel group, or None if not initialized.

    C240: mirrors vLLM V1 ``vllm/distributed/parallel_state.py:
    get_ep_group``.  The EP group is constructed from ``DP × TP``
    ranks such that each EP rank owns a disjoint slice of experts
    (when ``ep_size > 1``).  Returns ``None`` for dense models
    (no expert parallelism).
    """
    return _EP


def get_cp_group() -> "GroupCoordinator | None":
    """Return the context-parallel group, or None if not initialized.

    Mirrors vLLM V1 distributed/parallel_state.py:get_cp_group.
    """
    return _CP


def get_dp_world_size() -> int:
    return 1 if _DP is None else _DP.world_size


def get_dp_rank() -> int:
    return 0 if _DP is None else _DP.rank_in_group


def get_cp_world_size() -> int:
    return 1 if _CP is None else _CP.world_size


def get_cp_rank() -> int:
    return 0 if _CP is None else _CP.rank_in_group


def init_data_parallel(
    data_parallel_size: int,
    tensor_model_parallel_size: int = 1,
) -> None:
    """Initialize the data-parallel group.

    Mirrors vLLM V1 initialize_model_parallel (data-parallel subset).

    Splits the world into `data_parallel_size` replicas, each owning
    `tensor_model_parallel_size` ranks. For my-vllm's single-process
    default (world_size=1) this is a no-op; it requires a real
    multi-process NCCL setup.

    C141: stub — wires up a `_DP` group when dp_size > 1. Full
    rank-placement logic (TP × DP) is deferred to 阶段 4 follow-up.
    """
    global _DP
    if data_parallel_size <= 1 or not dist.is_initialized():
        _DP = None
        return
    # Construct replica groups: each DP replica owns TP-consecutive ranks.
    world_size = dist.get_world_size()
    dp_size = data_parallel_size
    tp_size = tensor_model_parallel_size
    assert world_size == dp_size * tp_size, (
        f"world_size ({world_size}) != dp_size ({dp_size}) * "
        f"tp_size ({tp_size})"
    )
    dp_group_ranks = [
        list(range(r * tp_size, (r + 1) * tp_size)) for r in range(dp_size)
    ]
    _DP = GroupCoordinator(
        group_ranks=dp_group_ranks,
        local_rank=dist.get_rank() % tp_size,
        backend="nccl",
        group_name="dp_group",
    )


def init_context_parallel(
    context_parallel_size: int,
    tensor_model_parallel_size: int = 1,
) -> None:
    """Initialize the context-parallel group.

    Mirrors vLLM V1 initialize_model_parallel (context-parallel subset).

    C142: stub — wires up a `_CP` group when cp_size > 1. Full
    rank-placement (TP × CP) is deferred to 阶段 4 follow-up.
    """
    global _CP
    if context_parallel_size <= 1 or not dist.is_initialized():
        _CP = None
        return
    world_size = dist.get_world_size()
    cp_size = context_parallel_size
    tp_size = tensor_model_parallel_size
    assert world_size == cp_size * tp_size, (
        f"world_size ({world_size}) != cp_size ({cp_size}) * "
        f"tp_size ({tp_size})"
    )
    cp_group_ranks = [
        list(range(r * tp_size, (r + 1) * tp_size)) for r in range(cp_size)
    ]
    _CP = GroupCoordinator(
        group_ranks=cp_group_ranks,
        local_rank=dist.get_rank() % tp_size,
        backend="nccl",
        group_name="cp_group",
    )


def model_parallel_is_initialized() -> bool:
    return _TP is not None


def init_expert_parallel(
    expert_parallel_size: int = 1,
    tensor_model_parallel_size: int = 1,
    backend: str = "nccl",
) -> None:
    """Initialise the expert-parallel group.

    C240: mirrors vLLM V1 ``init_expert_parallel``.  The EP group
    is constructed from ``TP × EP`` ranks such that each EP rank
    owns a disjoint slice of experts (when ``ep_size > 1``).

    For my-vllm's single-process default (``expert_parallel_size
    <= 1``), this is a no-op — the EP group is not needed for
    dense inference.

    For multi-process, the EP group is the same as the TP group
    when ``ep_size == world_size``; the vLLM convention is that
    each EP replica spans the full TP group.  This simplification
    matches the my-vllm C230 elastic EP rebalance contract (each
    "EP rank" is a full TP group).
    """
    global _EP
    if expert_parallel_size <= 1 or not dist.is_initialized():
        _EP = None
        return
    world_size = dist.get_world_size()
    tp_size = tensor_model_parallel_size
    assert world_size == expert_parallel_size * tp_size, (
        f"world_size ({world_size}) != ep_size ({expert_parallel_size}) "
        f"* tp_size ({tp_size})"
    )
    ep_group_ranks = [
        list(range(r * tp_size, (r + 1) * tp_size))
        for r in range(expert_parallel_size)
    ]
    _EP = GroupCoordinator(
        group_ranks=ep_group_ranks,
        local_rank=dist.get_rank() % tp_size,
        backend=backend,
        group_name="ep_group",
    )


def destroy_model_parallel() -> None:
    global _TP, _PP, _DP, _CP, _EP
    if _TP is not None:
        _TP.destroy()
        _TP = None
    if _PP is not None:
        _PP.destroy()
        _PP = None
    if _DP is not None:
        _DP.destroy()
        _DP = None
    if _CP is not None:
        _CP.destroy()
        _CP = None
    if _EP is not None:
        _EP.destroy()
        _EP = None
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
