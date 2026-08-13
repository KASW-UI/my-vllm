"""GPU-side DP (Data-Parallel) dispatch helpers.

Mirrors vLLM V1 v1/worker/gpu/dp_utils.py.

DP groups replicate the model and split the request batch across
the replicas. The GPU-side helper synchronizes CUDA-graph dispatch
across DP ranks: each rank picks a `BatchExecutionDescriptor` (CG
mode + uniform token count), and `sync_cudagraph_and_dp_padding`
all-reduces the rank's choice to a consensus so all DP replicas
agree on the same kernel launch.

C141: minimal subset — implements the all-reduce consensus loop
without the full `BatchExecutionDescriptor` plumbing
(`cudagraph_manager.dispatch(...)`, CG mode / uniform-token-count
negotiation). The function signature matches vLLM so a follow-up
can swap in the real `BatchExecutionDescriptor` without changing
callers.

For `dp_size == 1` (my-vllm's default) the helper returns the
local batch descriptor unchanged — no collective call needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


@dataclass
class BatchExecutionDescriptor:
    """Stand-in for vllm's BatchExecutionDescriptor (gpu/cudagraph_utils).

    Holds the three fields that DP dispatch needs to agree on:
    `cg_mode` (CUDA graph mode id, 0 = NONE), `num_tokens`, and
    `num_reqs`. The full vLLM dataclass adds `num_active_loras`
    and `uniform_token_count`; C141 uses these as plain ints.
    """

    cg_mode: int
    num_tokens: int
    num_reqs: int
    num_active_loras: int = 0
    uniform_token_count: int | None = None


def sync_cudagraph_and_dp_padding(
    cudagraph_manager: Any | None,
    desired_batch_desc: BatchExecutionDescriptor,
    num_tokens: int,
    num_reqs: int,
    uniform_token_count: int | None,
    dp_size: int,
    dp_rank: int,
    num_active_loras: int = 0,
) -> tuple[BatchExecutionDescriptor, torch.Tensor | None]:
    """Coordinate batch descriptor + DP padding across ranks.

    Mirrors vLLM V1 v1/worker/gpu/dp_utils.py:sync_cudagraph_and_dp_padding.

    All-reduces a 3-row tensor `(num_tokens, cg_mode, uniform_token_count_or_0)`
    across the DP group, then:
      - if all ranks have num_tokens == 0 → returns an empty batch
        descriptor and None.
      - else picks the minimum CG mode (any rank running eager
        forces everyone to run eager).
      - else picks the max num_tokens (DP padding) and the uniform
        token count if all ranks agree.

    For `dp_size == 1` the helper short-circuits and returns
    `(desired_batch_desc, None)` unchanged.
    """
    if dp_size == 1:
        return desired_batch_desc, None

    from my_vllm.distributed.parallel_state import get_dp_group
    group = get_dp_group().cpu_group if get_dp_group() is not None else None
    if group is None:
        return desired_batch_desc, None

    tensor = torch.zeros(3, dp_size, dtype=torch.int32)
    tensor[0][dp_rank] = num_tokens
    tensor[1][dp_rank] = desired_batch_desc.cg_mode
    tensor[2][dp_rank] = uniform_token_count or 0
    dist.all_reduce(tensor, group=group)

    num_tokens_across_dp = tensor[0]
    cg_mode_across_dp = tensor[1]
    uniform_token_counts_across_dp = tensor[2]

    if int(num_tokens_across_dp.max().item()) == 0:
        return (
            BatchExecutionDescriptor(cg_mode=0, num_tokens=0, num_reqs=0),
            None,
        )

    synced_cg_mode = int(cg_mode_across_dp.min().item())
    synced_num_tokens = int(num_tokens_across_dp.max().item())
    synced_uniform_token_count = int(uniform_token_counts_across_dp[0].item())
    if synced_uniform_token_count == 0 or not bool(
        torch.all(uniform_token_counts_across_dp == synced_uniform_token_count).item()
    ):
        synced_uniform_token_count = None

    # Dispatch via cudagraph_manager if available; otherwise the
    # bare (cg_mode, num_tokens, num_reqs) triple suffices.
    if cudagraph_manager is not None and synced_cg_mode != 0:
        synced_desc = cudagraph_manager.dispatch(
            num_reqs=num_reqs,
            num_tokens=synced_num_tokens,
            uniform_token_count=synced_uniform_token_count,
            num_active_loras=num_active_loras,
        )
    else:
        synced_desc = BatchExecutionDescriptor(
            cg_mode=synced_cg_mode,
            num_tokens=synced_num_tokens,
            num_reqs=num_reqs,
            num_active_loras=num_active_loras,
            uniform_token_count=synced_uniform_token_count,
        )
    num_tokens_across_dp[:] = synced_desc.num_tokens
    return synced_desc, num_tokens_across_dp


def dispatch_cg_and_sync_dp(
    cudagraph_manager: Any | None,
    num_reqs: int,
    num_tokens: int,
    uniform_token_count: int | None,
    dp_size: int,
    dp_rank: int,
    need_eager: bool = False,
    num_active_loras: int = 0,
) -> tuple[BatchExecutionDescriptor, torch.Tensor | None]:
    """Top-level DP dispatch helper.

    Mirrors vLLM V1 v1/worker/gpu/dp_utils.py:dispatch_cg_and_sync_dp.
    Picks a per-rank batch descriptor (or forces eager), then
    all-reduces across the DP group to reach consensus.
    """
    if need_eager or cudagraph_manager is None:
        batch_desc = BatchExecutionDescriptor(
            cg_mode=0,  # NONE
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_active_loras=num_active_loras,
            uniform_token_count=uniform_token_count,
        )
    else:
        batch_desc = cudagraph_manager.dispatch(
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            uniform_token_count=uniform_token_count,
            num_active_loras=num_active_loras,
        )

    if dp_size == 1:
        return batch_desc, None

    return sync_cudagraph_and_dp_padding(
        cudagraph_manager,
        batch_desc,
        num_tokens,
        num_reqs,
        uniform_token_count,
        dp_size,
        dp_rank,
        num_active_loras=num_active_loras,
    )


__all__ = [
    "BatchExecutionDescriptor",
    "dispatch_cg_and_sync_dp",
    "sync_cudagraph_and_dp_padding",
]