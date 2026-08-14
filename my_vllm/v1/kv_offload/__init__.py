"""KV cache offloading — facade.

C255: re-exports the offload abstractions from `base.py`,
`config.py`, `factory.py`, and `file_mapper.py` so callers can
`from my_vllm.v1.kv_offload import OffloadingManager,
OffloadingSpec, OffloadingConfig, FileMapper, OffloadingSpecFactory`.

The package layout:

    my_vllm/v1/kv_offload/
        ├── __init__.py        (this file — facade)
        ├── base.py            (OffloadingManager / Worker / Spec ABCs +
                                 LoadStoreSpec + enums + canonical cache types)
        ├── config.py          (OffloadingConfig sub-configs)
        ├── factory.py         (OffloadingSpecFactory registry)
        └── file_mapper.py     (FileMapper — disk layout primitive)

Future C256 (KV Offload tiering) adds `cpu/`, `nvme/`, `tiering/`
subpackages with concrete specs. C257 (Simple KV Offload) adds a
manager-only lightweight path.
"""

from my_vllm.v1.kv_offload.base import (
    BlockIDsLoadStoreSpec,
    CanonicalKVCacheRef,
    CanonicalKVCacheTensor,
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    Locality,
    LookupResult,
    OffloadKey,
    OffloadPolicy,
    OffloadingCounterMetadata,
    OffloadingEvent,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
    OffloadingKVEventsConfig,
    OffloadingManager,
    OffloadingMetricMetadata,
    OffloadingSpec,
    OffloadingWorker,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
    TransferResult,
    get_offload_block_hash,
    get_offload_group_idx,
    make_offload_key,
)
from my_vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingGroupConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from my_vllm.v1.kv_offload.factory import OffloadingSpecFactory
from my_vllm.v1.kv_offload.file_mapper import FileMapper
# C256 CPU + Tiering re-exports.
from my_vllm.v1.kv_offload.cpu import (
    ARCCachePolicy,
    BlockStatus,
    CachePolicy,
    CPUOffloadingManager,
    CPUOffloadingMetrics,
    CPUOffloadingSpec,
    CPUOffloadingWorker,
    CPULoadStoreSpec,
    LRUCachePolicy,
)
from my_vllm.v1.kv_offload.tiering import (
    CPUPrimaryTierOffloadingManager,
    ExampleSecondaryTierManager,
    JobMetadata,
    JobResult,
    ParentManager,
    SecondaryTierFactory,
    SecondaryTierManager,
    TieringOffloadingManager,
    TieringOffloadingSpec,
)


__all__ = [
    # Base types
    "BlockIDsLoadStoreSpec",
    "CanonicalKVCacheRef",
    "CanonicalKVCacheTensor",
    "CanonicalKVCaches",
    "GPULoadStoreSpec",
    "LoadStoreSpec",
    "Locality",
    "LookupResult",
    "OffloadKey",
    "OffloadPolicy",
    "OffloadingCounterMetadata",
    "OffloadingEvent",
    "OffloadingGaugeMetadata",
    "OffloadingHistogramMetadata",
    "OffloadingKVEventsConfig",
    "OffloadingManager",
    "OffloadingMetricMetadata",
    "OffloadingSpec",
    "OffloadingWorker",
    "PrepareStoreOutput",
    "ReqContext",
    "RequestOffloadingContext",
    "ScheduleEndContext",
    "TransferResult",
    "get_offload_block_hash",
    "get_offload_group_idx",
    "make_offload_key",
    # Config
    "OffloadingCacheConfig",
    "OffloadingConfig",
    "OffloadingGroupConfig",
    "OffloadingModelConfig",
    "OffloadingParallelConfig",
    # Factory
    "OffloadingSpecFactory",
    # FileMapper
    "FileMapper",
    # CPU backend (C256)
    "ARCCachePolicy",
    "BlockStatus",
    "CachePolicy",
    "CPUOffloadingManager",
    "CPUOffloadingMetrics",
    "CPUOffloadingSpec",
    "CPUOffloadingWorker",
    "CPULoadStoreSpec",
    "LRUCachePolicy",
    # Tiering (C256)
    "CPUPrimaryTierOffloadingManager",
    "ExampleSecondaryTierManager",
    "JobMetadata",
    "JobResult",
    "ParentManager",
    "SecondaryTierFactory",
    "SecondaryTierManager",
    "TieringOffloadingManager",
    "TieringOffloadingSpec",
]