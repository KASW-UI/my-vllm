"""NIXL connector utilities.

C250: extracted from ``vllm/distributed/kv_transfer/kv_connector/v1/nixl/utils.py``
(68 lines).  Scope reductions:

  * ``zmq_ctx`` + ``make_zmq_socket`` — **omitted** (ZMQ is not vendored;
    my-vllm uses the C242 SHM broadcast primitive for out-of-band
    handshake messages, see ``my_vllm/distributed/device_communicators/
    shm_broadcast.py``).
  * ``current_platform.get_nixl_supported_devices`` — **omitted**
    (NIXL is not vendored; the dev box is single-platform CUDA).
  * ``regex`` import — replaced with stdlib ``re`` (regex not vendored).
  * ``get_representative_spec_type`` — kept (pure helper).
  * ``get_base_request_id`` — kept verbatim.

References:
  * vLLM utils: ``vllm/distributed/kv_transfer/kv_connector/v1/nixl/utils.py``.
"""

from __future__ import annotations

import re
from typing import Any

# Trailing 8-hex randomization suffix appended by
# ``input_processor.assign_request_id`` as ``-{random_uuid():.8}``.
_RANDOM_SUFFIX_RE = re.compile(r"-[0-9a-f]{8}$", re.IGNORECASE)


def get_base_request_id(request_id: str) -> str:
    """Strip the per-request ``-<8 hex>`` randomization suffix, if present.

    Mirrors ``vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils
    .get_base_request_id``.  Used when matching producer / consumer
    request IDs across the PD handshake — the producer and consumer
    generate their own random suffix, but the base ID is stable.
    """
    return _RANDOM_SUFFIX_RE.sub("", request_id)


def get_representative_spec_type(spec: Any) -> type:
    """Return the representative ``KVCacheSpec`` type.

    Mirrors ``vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils
    .get_representative_spec_type``.  When ``spec`` is a
    ``UniformTypeKVCacheSpecs`` (which holds multiple inner specs of
    the same type), we pick the first one; otherwise we return
    ``type(spec)`` directly.  This is used in TP mapping to decide
    whether the local TP group reads from one or multiple remote
    TP ranks.
    """
    inner = getattr(spec, "kv_cache_specs", None)
    if inner:
        first = next(iter(inner.values()))
        return type(first)
    return type(spec)


__all__ = ["get_base_request_id", "get_representative_spec_type"]
