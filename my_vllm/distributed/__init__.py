"""my_vllm/distributed — Distributed runtime primitives.

C013-C050: world_size=1 defaults + TP group coordinator.
C083: real torch.distributed process groups (NCCL).
C150: ``init_distributed_environment`` (env-based init for multi-process).
C214: ``CustomAllreduce`` — same-host IPC ring all-reduce.
C230: ``elastic_ep`` rebalance controller.
C240: ``get_ep_group`` + ``init_expert_parallel`` for the EP all-to-all path.

Public surface (re-exports for downstream ``my_vllm.model_executor``
importers):

  * ``GroupCoordinator`` — wraps a ``torch.distributed`` process group.
  * ``init_distributed_environment`` / ``initialize_model_parallel`` —
    process-group bring-up.
  * ``init_data_parallel`` / ``init_context_parallel`` /
    ``init_expert_parallel`` — secondary group constructors.
  * ``get_tp_group`` / ``get_dp_group`` / ``get_cp_group`` /
    ``get_ep_group`` — group accessors.
  * ``model_parallel_is_initialized`` / ``destroy_model_parallel`` —
    lifecycle.
  * ``CustomAllreduce`` / ``get_local_ipc_handle`` / ``open_ipc_handle``
    — same-host multi-GPU custom all-reduce.
"""

from __future__ import annotations

from my_vllm.distributed.custom_all_reduce import (
    CustomAllreduce,
    get_local_ipc_handle,
    meta_size,
    open_ipc_handle,
)
from my_vllm.distributed.parallel_state import (
    GroupCoordinator,
    destroy_model_parallel,
    get_cp_group,
    get_cp_rank,
    get_cp_world_size,
    get_dp_group,
    get_dp_rank,
    get_dp_world_size,
    get_ep_group,
    get_pp_group,
    get_tp_group,
    get_tp_rank,
    get_tp_world_size,
    init_context_parallel,
    init_data_parallel,
    init_distributed_environment,
    init_expert_parallel,
    initialize_model_parallel,
    model_parallel_is_initialized,
)


__all__ = [
    "GroupCoordinator",
    "CustomAllreduce",
    "get_local_ipc_handle",
    "open_ipc_handle",
    "meta_size",
    "destroy_model_parallel",
    "get_cp_group",
    "get_cp_rank",
    "get_cp_world_size",
    "get_dp_group",
    "get_dp_rank",
    "get_dp_world_size",
    "get_ep_group",
    "get_pp_group",
    "get_tp_group",
    "get_tp_rank",
    "get_tp_world_size",
    "init_context_parallel",
    "init_data_parallel",
    "init_distributed_environment",
    "init_expert_parallel",
    "initialize_model_parallel",
    "model_parallel_is_initialized",
]
