"""Default NIXL worker façade.

C250: extracted from ``vllm/distributed/kv_transfer/kv_connector/v1/nixl/
worker.py`` (13 lines).  This is a trivial re-export; the real
work lives in ``pull_worker.py`` / ``push_worker.py``.

The default is the pull-mode variant (backward-compatible with
vLLM's ``NixlConnector`` alias).
"""

from my_vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
)


NixlConnectorWorker = NixlPullConnectorWorker


__all__ = ["NixlConnectorWorker"]
