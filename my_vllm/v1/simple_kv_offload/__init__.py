# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SimpleCPUOffload — pluggable CPU offload KV connector (C257).

Mirrors ``vllm/v1/simple_kv_offload/__init__.py`` (empty upstream).

C257: re-exports the manager + worker + metadata surface so callers
can do ``from my_vllm.v1.simple_kv_offload import (
    SimpleCPUOffloadScheduler, SimpleCPUOffloadWorker,
    SimpleCPUOffloadMetadata, SimpleCPUOffloadWorkerMetadata,
)``.

Note: my-vllm's ``simple_kv_offload`` is a **standalone connector**
that wires directly into the C251 ``KVConnectorBase_V1`` surface.
It is intentionally NOT based on the C255/C256 ``OffloadingSpec``
plug-in pattern — the simple connector is a separate, lighter
implementation that doesn't need the spec / manager / worker ABCs.
"""

from my_vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler
from my_vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)
from my_vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker

__all__ = [
    "SimpleCPUOffloadMetadata",
    "SimpleCPUOffloadScheduler",
    "SimpleCPUOffloadWorker",
    "SimpleCPUOffloadWorkerMetadata",
]
