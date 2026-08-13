"""ExampleConnector — minimal reference KV connector.

Mirrors vLLM V1 distributed/kv_transfer/kv_connector/v1/example_connector.py.

vLLM's `ExampleConnector` is a 444-line debug connector that uses
disk-based KV storage to demonstrate the connector lifecycle without
requiring NIXL / RDMA. It overrides every primitive
(`start_load_kv`, `save_kv_layer`, `wait_for_layer_load`,
`wait_for_save`, `get_num_new_matched_tokens`, `request_finished`,
`build_connector_worker_meta`, etc.) and stores bundles as pickle
files under a configurable path.

C137: minimal subset — `ExampleConnector` is implemented as a
thin alias of `NixlConnector`. Both backends in my-vllm use the
same shared-memory transfer mechanism (the vLLM NIXL RDMA backend
is out of scope — my-vllm does not have GPU-direct RDMA libs).

The alias preserves the vLLM-canonical class name
(`ExampleConnector`) so `KVConnectorFactory.register_connector(
"ExampleConnector", ...)` matches vLLM's registration pattern.
Adding a disk-based debug connector can be done by subclassing
`NixlConnector` and overriding `save_kv_layer` to write
`kv_layer.cpu().numpy().tofile(...)`.
"""

from __future__ import annotations

from my_vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlConnector,
)


class ExampleConnector(NixlConnector):
    """Debug reference KV connector — alias of NixlConnector in my-vllm.

    Mirrors vLLM V1 distributed/kv_transfer/kv_connector/v1/
    example_connector.py:ExampleConnector.

    In vLLM, this connector writes bundles to disk under
    `shared_storage_path` for offline debugging. In my-vllm it
    shares the shared-memory transfer backend with `NixlConnector`
    — both serve as ≥1 real backend for PD disaggregation testing.

    The `method` field in `KVTransferConfig` selects which connector
    the `KVConnectorFactory` instantiates; `"NixlConnector"` and
    `"ExampleConnector"` both resolve to this class today.
    """

    pass


__all__ = ["ExampleConnector"]