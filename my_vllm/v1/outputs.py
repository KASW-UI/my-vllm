"""Output data structures — mirrors vLLM V1 v1/outputs.py.

C015: minimal placeholder definitions. Will be extended in C016-C017.
EngineCoreOutput / EngineCoreOutputs moved to v1/engine/__init__.py (L7 alignment).

C257: KVConnectorOutput dataclass added — connector scheduler-side hook
returns this from ``update_connector_output()`` so the scheduler can
route finished loads back to the request lifecycle and aggregate
worker-side store completions.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SamplerOutput:
    """Output from the Sampler."""

    sampled_token_ids: list[int]  # [num_reqs] — one token per request


@dataclass
class ModelRunnerOutput:
    """Output from ModelRunner after executing a batch.

    C015 minimal definition. Uses index-aligned parallel lists
    matching real vLLM's IPC-optimized layout.
    """

    req_ids: list[str]
    req_id_to_index: dict[str, int]
    sampled_token_ids: list[list[int]]


@dataclass
class KVConnectorOutput:
    """Output emitted by KV connectors per scheduler step.

    Mirrors vLLM V1 v1/outputs.py:42 KVConnectorOutput.

    ``finished_sending`` — request ids whose stores to the external
    cache have completed (returned by the connector's ``get_finished``
    on the worker).
    ``finished_recving`` — request ids whose loads from the external
    cache have completed (returned by the connector's ``get_finished``
    on the worker).
    ``kv_connector_worker_meta`` — connector-specific aggregate data
    from each worker (e.g. completed-store-event counts for the
    C257 SimpleCPUOffloadScheduler).
    ``preempted_req_ids`` — request ids preempted this step, used by
    connectors that need to flush in-flight transfers (C257 sets
    ``SimpleCPUOffloadMetadata.need_flush`` based on this).
    """

    finished_sending: set[str] | None = None
    finished_recving: set[str] | None = None
    kv_connector_worker_meta: Any = None
    preempted_req_ids: set[str] | None = None

    def __post_init__(self) -> None:
        if self.finished_sending is None:
            self.finished_sending = set()
        if self.finished_recving is None:
            self.finished_recving = set()
        if self.preempted_req_ids is None:
            self.preempted_req_ids = set()
