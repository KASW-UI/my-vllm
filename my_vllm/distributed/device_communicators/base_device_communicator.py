"""my_vllm/distributed/device_communicators/base_device_communicator.py

Base abstract classes for per-device communication primitives.

C240: extracted from ``vllm/distributed/device_communicators/base_device_communicator.py``
(388 lines), with the following scope reductions per AGENTS.md JD-driven
scope:

  * ``Cache`` (weakref handle cache) — kept verbatim.
  * ``All2AllManagerBase`` ABC — kept verbatim, but ``support_fault_tolerance``
    is omitted (only the elastic EP path uses it; deferred to C247).
  * ``DeviceCommunicatorBase`` — kept with the
    ``all_gather``/``all_gatherv``/``reduce_scatter``/``reduce_scatterv``
    /``dispatch``/``combine`` surface.  The ``prepare_communication_buffer_for_model``
    method (which iterates MoE modules to call ``maybe_init_modular_kernel``)
    is omitted; my-vllm's MoE layer is still a single kernel without
    modular kernel wiring (deferred to C265).
  * The torch.distributed ``_world`` introspection is skipped (no need
    for stateless process-group support yet).

References:
  * vLLM base: ``vllm/distributed/device_communicators/base_device_communicator.py``
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Any
from weakref import WeakValueDictionary

import torch


class Cache:
    """Thread-safe weakref cache for handle objects keyed by hashable kwargs.

    Mirrors ``vllm.distributed.device_communicators.base_device_communicator
    .Cache``.  Used by All2All manager backends (e.g. DeepEP) that cache
    per-``(num_tokens, hidden_size)`` handle objects.
    """

    def __init__(self) -> None:
        self._cache: WeakValueDictionary = WeakValueDictionary()
        self._lock = threading.RLock()

    def get_or_create(self, kwargs: dict, func):
        key = tuple(sorted((k, v) for k, v in kwargs.items()))
        with self._lock:
            instance = self._cache.get(key)
            if instance is None:
                instance = func(**kwargs)
                self._cache[key] = instance
            return instance


class All2AllManagerBase(ABC):
    """Abstract base class for EP/DP all-to-all managers.

    Mirrors ``vllm.distributed.device_communicators.base_device_communicator
    .All2AllManagerBase``.  Subclasses implement one of the EP-path
    backends (DeepEP HT/LL/V2, NIXL EP, FlashInfer NVLink, Mori, AG+RS).
    """

    rank: int
    world_size: int

    def __init__(self, cpu_group: Any, tcp_store_group: Any = None) -> None:
        self.cpu_group = cpu_group
        self.tcp_store_group = tcp_store_group

        from my_vllm.distributed.parallel_state import (
            get_dp_group,
            get_tp_group,
        )

        self.dp_group = get_dp_group()
        self.tp_group = get_tp_group()

        self.dp_rank = (
            self.dp_group.rank_in_group if self.dp_group is not None else 0
        )
        self.dp_world_size = (
            self.dp_group.world_size if self.dp_group is not None else 1
        )
        self.rank = cpu_group.rank() if hasattr(cpu_group, "rank") else 0
        self.world_size = cpu_group.size() if hasattr(cpu_group, "size") else 1

    def get_handle(self, kwargs: dict) -> Any:
        """Return a backend-specific handle (e.g. DeepEP Buffer).

        Implementations cache the handle keyed by ``kwargs`` via
        :class:`Cache`.
        """
        raise NotImplementedError

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
        """Dispatch ``(hidden_states, router_logits[, extras])`` across EP/DP ranks."""
        raise NotImplementedError

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
        """Dispatch ``(hidden_states, topk_weights, topk_ids[, extras])`` across EP/DP ranks."""
        raise NotImplementedError

    def query_active_mask(self) -> torch.Tensor:
        raise NotImplementedError

    def query_fault(self) -> torch.Tensor:
        raise NotImplementedError

    def set_num_sms(self, num_sms: int) -> None:
        pass

    def max_sms_used(self) -> int | None:
        return None

    def combine(
        self,
        hidden_states: torch.Tensor,
        is_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        """Reduce-scatter ``hidden_states`` across EP/DP ranks."""
        raise NotImplementedError

    def destroy(self) -> None:
        pass


class DeviceCommunicatorBase:
    """Per-device communicator façade.

    C240: mirrors vLLM ``DeviceCommunicatorBase``.  Subclasses are
    expected to override ``all_gatherv`` and ``reduce_scatterv`` with
    the device-specific implementation (e.g. NCCL for CUDA, Gloo for
    CPU).  The default implementations delegate to the bound
    ``GroupCoordinator`` (see ``cuda_communicator.CudaCommunicator``).
    """

    def __init__(
        self,
        cpu_group: Any,
        device: torch.device | None = None,
        device_group: Any | None = None,
        unique_name: str = "",
    ) -> None:
        self.device = device or torch.device("cpu")
        self.cpu_group = cpu_group
        self.device_group = device_group
        self.unique_name = unique_name

        if cpu_group is not None and hasattr(cpu_group, "rank"):
            self.rank = cpu_group.rank()
            self.world_size = cpu_group.size()
        else:
            import torch.distributed as dist

            self.rank = dist.get_rank(cpu_group) if cpu_group else 0
            self.world_size = dist.get_world_size(cpu_group) if cpu_group else 1

        self.is_ep_communicator = unique_name.split(":")[0] == "ep"
        self.use_all2all = self.is_ep_communicator
        self.all2all_backend: str | None = None
        self.all2all_manager: All2AllManagerBase | None = None

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist
        dist.all_reduce(input_, group=self.device_group)
        return input_

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Uniform-size all-gather along ``dim``."""
        import torch.distributed as dist

        if dim < 0:
            dim += input_.dim()
        if self.world_size == 1:
            return input_

        input_size = input_.size()
        output_size = (input_size[0] * self.world_size,) + input_size[1:]
        output_tensor = torch.empty(
            output_size, dtype=input_.dtype, device=input_.device
        )
        dist.all_gather_into_tensor(output_tensor, input_, group=self.device_group)
        output_tensor = output_tensor.reshape((self.world_size,) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(
            input_size[:dim]
            + (self.world_size * input_size[dim],)
            + input_size[dim + 1:]
        )
        return output_tensor

    def all_gatherv(
        self,
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        raise NotImplementedError

    def reduce_scatter(
        self, input_: torch.Tensor, dim: int = -1
    ) -> torch.Tensor:
        """Uniform-size reduce-scatter (sum) along ``dim``."""
        import torch.distributed as dist

        if self.world_size == 1:
            return input_
        if dim < 0:
            dim += input_.dim()
        assert input_.size(dim) % self.world_size == 0
        chunk_size = input_.size(dim) // self.world_size
        output_shape = (chunk_size,) + tuple(input_.size()[1:])
        output_tensor = torch.empty(
            output_shape, dtype=input_.dtype, device=input_.device
        )
        moved = input_.movedim(0, dim).contiguous()
        moved_view = moved.view(
            self.world_size, chunk_size, *moved.size()[1:]
        )
        local_chunk = moved_view[self.rank].contiguous()
        torch.distributed.reduce_scatter_tensor(
            output_tensor, local_chunk,
            op=dist.ReduceOp.SUM,
            group=self.device_group,
        )
        return output_tensor.movedim(0, dim).contiguous()

    def reduce_scatterv(
        self,
        input_: torch.Tensor,
        dim: int = -1,
        sizes: list[int] | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None:
        """Gather to a single destination rank.

        NOTE: ``dst`` is the local rank-in-group index (mirrors vLLM
        convention).
        """
        import torch.distributed as dist

        if dim < 0:
            dim += input_.dim()
        ranks = (
            list(range(self.world_size))
            if self.device_group is None
            else dist.get_process_group_ranks(self.device_group)
        )
        if self.rank == dst:
            gather_list = [torch.empty_like(input_) for _ in range(self.world_size)]
        else:
            gather_list = None
        dist.gather(
            input_, gather_list, dst=ranks[dst], group=self.device_group
        )
        if self.rank == dst:
            return torch.cat(gather_list, dim=dim)
        return None

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        import torch.distributed as dist
        if self.world_size == 1:
            return tensor
        ranks = (
            list(range(self.world_size))
            if self.device_group is None
            else dist.get_process_group_ranks(self.device_group)
        )
        dist.broadcast(tensor, src=ranks[src], group=self.device_group)
        return tensor

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
        """Dispatch via ``all2all_manager``.  No-op when ``all2all_manager is None``."""
        if self.all2all_manager is None:
            if extra_tensors is not None:
                return hidden_states, router_logits, extra_tensors
            return hidden_states, router_logits
        return self.all2all_manager.dispatch_router_logits(
            hidden_states, router_logits, is_sequence_parallel, extra_tensors
        )

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
        """Dispatch via ``all2all_manager``.  No-op when ``all2all_manager is None``."""
        if self.all2all_manager is None:
            if extra_tensors is not None:
                return hidden_states, topk_weights, topk_ids, extra_tensors
            return hidden_states, topk_weights, topk_ids
        return self.all2all_manager.dispatch(
            hidden_states, topk_weights, topk_ids,
            is_sequence_parallel, extra_tensors=extra_tensors,
        )

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        """Combine via ``all2all_manager``.  No-op when ``all2all_manager is None``."""
        if self.all2all_manager is None:
            return hidden_states
        return self.all2all_manager.combine(hidden_states, is_sequence_parallel)

    def destroy(self) -> None:
        if self.all2all_manager is not None:
            self.all2all_manager.destroy()
            self.all2all_manager = None
