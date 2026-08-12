"""KV Transfer — distributed KV cache transport layer.

Real vLLM provides multiple backends:
  - NIXL (NVIDIA In-Xfer Library) — RDMA-based, GPU-direct
  - Mooncake — RDMA-based, optimized for MoE
  - Moriio — RDMA-based
  - LMCache — CPU/disk caching with shared filesystem
  - HF3FS — 3FS filesystem-based offload
  - OffloadingConnector — CPU/disk swap
  - SimpleCPUOffloadConnector — naive CPU offload

Each backend implements KVConnectorBase_V1 (see v1/core/kv_connector.py)
with scheduler-side + worker-side interfaces.
"""

__all__ = []
