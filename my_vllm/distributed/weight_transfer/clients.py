"""Built-in :class:`VLLMWeightSyncClient` implementations.

C243: closes the A-level "跨节点 Weight Transfer" gap from
``phase3-distributed-runtime-plan.md``.

Mirrors ``vllm/distributed/weight_transfer/clients.py`` (114 lines →
~70 in my-vllm).

The trainer-side weight transfer engines drive a four-method control
plane on the inference side via the :class:`VLLMWeightSyncClient`
protocol.  This module ships one concrete client used by the unit
tests + an end-to-end smoke harness:

* :class:`InProcessVLLMWeightSyncClient` — wraps a single
  :class:`WeightTransferEngine` instance and routes
  ``init_weight_transfer_engine`` /
  ``start_weight_update`` / ``update_weights`` /
  ``finish_weight_update`` calls straight to the engine's methods.
  No serialization, no transport — same process, same Python objects.

Scope reductions vs vLLM:

* **No :class:`HTTPVLLMWeightSyncClient`** — my-vllm has no RLHF HTTP
  routes; the OpenAI API server was deleted in C151.
* **No :class:`RayVLLMWeightSyncClient`** — my-vllm has no Ray.
"""

from typing import Any

from my_vllm.distributed.weight_transfer.base import (
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)


class InProcessVLLMWeightSyncClient:
    """In-process :class:`VLLMWeightSyncClient` adapter.

    Wraps a single :class:`WeightTransferEngine` instance and routes
    the four control-plane methods to the corresponding engine
    methods, deserializing dict payloads to the backend's typed
    dataclasses via ``parse_init_info`` / ``parse_update_info``.

    Useful for unit tests, end-to-end single-process harnesses, and
    embedded deployment where the trainer and worker share an
    address space (e.g. co-located RLHF inside one training job).
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def init_weight_transfer_engine(self, init_info: dict[str, Any]) -> None:
        typed = self.engine.parse_init_info(init_info)
        self.engine.init_transfer_engine(typed)

    def start_weight_update(self) -> None:
        self.engine.start_weight_update()

    def update_weights(self, update_info: dict[str, Any]) -> None:
        typed = self.engine.parse_update_info(update_info)
        self.engine.receive_weights(typed)
        # Worker-side synchronization happens in
        # ``WeightTransferEngine.update_weights`` (which wraps
        # ``receive_weights``); this client only routes the call.

    def finish_weight_update(self) -> None:
        self.engine.finish_weight_update()
