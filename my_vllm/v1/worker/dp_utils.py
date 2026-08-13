"""Worker-level DP (Data-Parallel) helpers.

Mirrors vLLM V1 v1/worker/dp_utils.py.

The worker-level helpers sit above the GPU-side `gpu/dp_utils.py`
and orchestrate DP at the request-batch level: they handle the
ubatch split (whether to split a step's work into multiple
micro-batches across DP ranks) and the cross-rank all-reduce
that decides DP padding + CUDA-graph mode.

vLLM's `prepare_dp_padding` / `decide_dp_ubatch` /
`gather_dp_attention` helpers integrate with vLLM's
`UbatchInfo` / `MicroBatchScheduler`. my-vllm does not yet have a
micro-batch scheduler (DP is wired for shape alignment, not
end-to-end execution).

C141: minimal subset — `gather_dp_outputs(...)` is the only fully
implemented helper (it does a cross-rank all_gather of per-rank
sample logprobs). The ubatch / padding decision helpers are
stubs that return the input unchanged (no-op for dp_size == 1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from my_vllm.config.parallel import ParallelConfig


def gather_dp_outputs(
    sample_logprobs: torch.Tensor | None,
    parallel_config: "ParallelConfig",
) -> torch.Tensor | None:
    """Gather per-rank sample-logprobs across the DP group.

    Mirrors vLLM V1 v1/worker/dp_utils.py (the gather_dp family of
    helpers, simplified). For `dp_size == 1` returns the input
    unchanged.

    Returns a tensor of shape `(dp_size * num_local_tokens, ...)`
    on rank 0; ranks > 0 return None.
    """
    dp_size = getattr(parallel_config, "data_parallel_size", 1)
    if dp_size == 1:
        return sample_logprobs

    from my_vllm.distributed.parallel_state import get_dp_group
    group = get_dp_group()
    if group is None:
        return sample_logprobs

    if sample_logprobs is None:
        # All ranks must participate in the collective.
        gathered = [torch.empty(0, dtype=torch.float32) for _ in range(dp_size)]
        torch.distributed.all_gather(gathered, torch.empty(0), group=group.device_group)
    else:
        gathered = [torch.empty_like(sample_logprobs) for _ in range(dp_size)]
        torch.distributed.all_gather(
            gathered, sample_logprobs, group=group.device_group
        )

    if group.rank_in_group == 0:
        return torch.cat(gathered, dim=0)
    return None


def decide_dp_ubatch(
    orig_num_tokens: int,
    padded_num_tokens: int,
    parallel_config: "ParallelConfig",
) -> bool:
    """Decide whether to split the current step into DP ubatches.

    Mirrors vLLM V1 v1/worker/dp_utils.py:_post_process_ubatch.

    C141: stub — always returns False (no ubatching). Wiring
    requires my-vllm's MicroBatchScheduler (out of scope).
    """
    return False


def prepare_dp_padding(
    orig_num_tokens: int,
    padded_num_tokens: int,
    parallel_config: "ParallelConfig",
) -> torch.Tensor:
    """Compute the per-rank `num_tokens` after DP padding.

    Mirrors vLLM V1 v1/worker/dp_utils.py:_post_process_dp_padding.

    C141: stub — returns a single-rank `num_tokens` tensor when
    dp_size == 1; for dp_size > 1, all-reduces the padded counts
    and returns the max for padding.
    """
    dp_size = getattr(parallel_config, "data_parallel_size", 1)
    dp_rank = getattr(parallel_config, "data_parallel_rank", 0)
    if dp_size == 1:
        return torch.tensor([padded_num_tokens], dtype=torch.int32)

    from my_vllm.distributed.parallel_state import get_dp_group
    group = get_dp_group()
    if group is None:
        return torch.tensor([padded_num_tokens], dtype=torch.int32)

    local = torch.zeros(2, dp_size, dtype=torch.int32)
    local[0][dp_rank] = orig_num_tokens
    local[1][dp_rank] = padded_num_tokens
    torch.distributed.all_reduce(local, group=group.cpu_group)
    max_padded = int(local[1].max().item())
    return torch.full((dp_size,), max_padded, dtype=torch.int32)


__all__ = [
    "decide_dp_ubatch",
    "gather_dp_outputs",
    "prepare_dp_padding",
]