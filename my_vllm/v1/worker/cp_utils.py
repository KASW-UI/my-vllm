"""Worker-level CP (Context-Parallel) helpers.

Mirrors vLLM V1 v1/worker/cp_utils.py.

vLLM's `cp_utils.py` (worker-side) provides:
  - `check_attention_cp_compatibility(...)` — model-specific
    checks (GQA groups divide cp_size, etc.).
  - `gather_cp_kv_cache_blocks(...)` — collects KV blocks from
    each CP rank into the local rank's block table.
  - `dispatch_cp_kv_cache_blocks(...)` — distributes local blocks
    to each CP rank.

C142: minimal subset — `check_attention_cp_compatibility` (model-
generic power-of-two check + a stub for GQA-group divisibility),
plus `dispatch_cp_kv_cache_blocks` and `gather_cp_kv_cache_blocks`
as stubs that all-gather / all-to-all the block-id tensors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from my_vllm.config.parallel import ParallelConfig


def check_attention_cp_compatibility(
    parallel_config: "ParallelConfig",
    num_kv_heads: int,
    num_attention_heads: int | None = None,
) -> None:
    """Validate the model supports the configured CP size.

    Mirrors vLLM V1 v1/worker/cp_utils.py:check_attention_cp_compatibility.

    Generic checks (cp_size is a power of two) run first. Model-
    specific checks (e.g. GQA groups divide cp_size) are stubs
    that raise NotImplementedError when non-trivial.
    """
    cp_size = getattr(parallel_config, "context_parallel_size", 1)
    if cp_size <= 1:
        return
    if cp_size & (cp_size - 1) != 0:
        raise ValueError(
            f"context_parallel_size must be a power of 2, got {cp_size}"
        )
    # GQA-group divisibility: only matters when num_attention_heads
    # is provided (i.e. model exposes the kv-head/attention-head
    # split). my-vllm's OPT is dense MHA (num_kv_heads ==
    # num_attention_heads), so this is trivially satisfied.
    if num_attention_heads is not None and num_kv_heads > 0:
        ratio = num_attention_heads // num_kv_heads
        if ratio % cp_size != 0:
            raise NotImplementedError(
                f"GQA group size {ratio} must divide cp_size {cp_size}; "
                f"non-trivial GQA + CP is out of scope for my-vllm's "
                f"Dense-OPT subset."
            )


def gather_cp_kv_cache_blocks(
    local_blocks: torch.Tensor,
    parallel_config: "ParallelConfig",
) -> torch.Tensor:
    """Gather per-CP-rank block-id tensors into the local rank.

    Mirrors vLLM V1 v1/worker/cp_utils.py:gather_cp_kv_cache_blocks.

    C142: stub — all-gathers along dim 0 (rows = ranks). For
    cp_size == 1 returns the input unchanged.
    """
    cp_size = getattr(parallel_config, "context_parallel_size", 1)
    if cp_size == 1:
        return local_blocks

    from my_vllm.distributed.parallel_state import get_cp_group
    group = get_cp_group()
    if group is None:
        return local_blocks

    gathered = [torch.empty_like(local_blocks) for _ in range(cp_size)]
    torch.distributed.all_gather(gathered, local_blocks, group=group.device_group)
    return torch.cat(gathered, dim=0)


def dispatch_cp_kv_cache_blocks(
    blocks: torch.Tensor,
    parallel_config: "ParallelConfig",
) -> torch.Tensor:
    """Distribute a stacked blocks tensor to each CP rank.

    C142: stub — splits along dim 0 by `cp_size`. For cp_size == 1
    returns the input unchanged.
    """
    cp_size = getattr(parallel_config, "context_parallel_size", 1)
    cp_rank = getattr(parallel_config, "context_parallel_rank", 0)
    if cp_size == 1:
        return blocks

    chunk_size = blocks.shape[0] // cp_size
    return blocks[cp_rank * chunk_size : (cp_rank + 1) * chunk_size]


__all__ = [
    "check_attention_cp_compatibility",
    "dispatch_cp_kv_cache_blocks",
    "gather_cp_kv_cache_blocks",
]