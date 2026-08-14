"""KV cache offloading — normalized configuration.

Mirrors vLLM V1 vllm/v1/kv_offload/config.py.

C255: a frozen-dataclass config tree consumed by
`OffloadingSpec` and its concrete backend implementations.

Sub-configs:
- `OffloadingGroupConfig` — per-group token span + layer names.
- `OffloadingModelConfig` — model identifier + KV dtype.
- `OffloadingCacheConfig` — tokens-per-hash + blocks-per-chunk.
- `OffloadingParallelConfig` — rank / world_size / TP / PP / PCP / DCP
  + data-parallel index + parallelism-agnostic flag.

Aggregated into `OffloadingConfig` with per-block worker-KV bytes
+ enable_kv_cache_events + extra_config mapping + engine_id.

my-vllm NOTE: `OffloadingConfig` is constructed directly in C255
tests. Production wiring (extracting these fields from a real
`VllmConfig`) lands with C256 (CPU/NVMe tiering) when the
concrete specs land.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OffloadingGroupConfig:
    """Per-group offload configuration.

    `tokens_per_block` is the total token span covered by one
    block across all workers (accounts for context parallelism).
    """

    tokens_per_block: int
    layer_names: tuple[str, ...]


@dataclass(frozen=True)
class OffloadingModelConfig:
    """Model identifier + KV cache data type."""

    name: str
    dtype: str


@dataclass(frozen=True)
class OffloadingCacheConfig:
    """Tokens per block hash + blocks coalesced into one offload chunk."""

    tokens_per_hash: int
    blocks_per_chunk: int


@dataclass(frozen=True)
class OffloadingParallelConfig:
    """Parallelism layout + worker identity.

    `is_parallelism_agnostic=True` means concatenating a block's
    data across all workers yields the same result regardless of
    parallelism configuration (used by `FileMapper` to collapse
    different TP/PP/PCP/DCP layouts into the same folder).
    """

    rank: int
    world_size: int
    tp_size: int
    pp_size: int
    pcp_size: int
    dcp_size: int
    data_parallel_index: int
    is_parallelism_agnostic: bool


@dataclass(frozen=True)
class OffloadingConfig:
    """Aggregated config consumed by an `OffloadingSpec`.

    Mirrors vLLM V1 vllm/v1/kv_offload/config.py:OffloadingConfig.
    """

    groups: tuple[OffloadingGroupConfig, ...]
    worker_kv_bytes_per_block: int
    enable_kv_cache_events: bool
    extra_config: Mapping[str, Any]
    engine_id: str
    model: OffloadingModelConfig
    cache: OffloadingCacheConfig
    parallel: OffloadingParallelConfig


__all__ = [
    "OffloadingCacheConfig",
    "OffloadingConfig",
    "OffloadingGroupConfig",
    "OffloadingModelConfig",
    "OffloadingParallelConfig",
]