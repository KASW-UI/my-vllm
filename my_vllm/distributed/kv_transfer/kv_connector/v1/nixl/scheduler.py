"""Default NIXL scheduler façade.

C250: extracted from ``vllm/distributed/kv_transfer/kv_connector/v1/nixl/
scheduler.py`` (12 lines).  Re-exports the pull-mode scheduler;
the real work lives in ``pull_scheduler.py`` / ``push_scheduler.py``.
"""

from my_vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
    NixlPullConnectorScheduler,
)


NixlConnectorScheduler = NixlPullConnectorScheduler


__all__ = ["NixlConnectorScheduler"]
