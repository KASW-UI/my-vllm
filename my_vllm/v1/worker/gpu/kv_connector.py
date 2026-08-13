"""Worker-side KVConnector wrapper.

Mirrors vLLM V1 v1/worker/gpu/kv_connector.py:29.

The vLLM file defines a `KVConnector` interface used by
`GPUModelRunner`:
  - `pre_forward(scheduler_output)`  — call before model forward.
  - `post_forward(finished_req_ids, wait_for_save)` — call after.
  - `no_forward(scheduler_output)` — drain when there's no work.
  - `set_disabled(disabled)` — toggle for graph capture.

`ActiveKVConnector` is the concrete impl that wraps the
`KVConnectorBase_V1` instance and registers the KV caches.

C137: minimal subset — both classes exist for 1:1 file-tree
alignment. The default `KVConnector()` instance is a no-op
singleton (`NO_OP_KV_CONNECTOR`); `ActiveKVConnector` defers
all real work to `KVConnectorBase_V1` via the existing
`v1/engine/core.py` plumbing. The `GPUModelRunner` hot path
does not yet call `pre_forward` / `post_forward` — that's a
follow-up refactor (阶段 3 worker-side wiring).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from my_vllm.config.vllm import VllmConfig
    from my_vllm.v1.core.sched.output import SchedulerOutput
    from my_vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput


class KVConnector:
    """KVConnector interface used by GPUModelRunner.

    Mirrors vLLM V1 v1/worker/gpu/kv_connector.py:KVConnector.
    All methods are no-ops by default; `ActiveKVConnector` overrides
    them to delegate to `KVConnectorBase_V1`.
    """

    def pre_forward(self, scheduler_output: "SchedulerOutput") -> None:
        pass

    def post_forward(
        self,
        finished_req_ids: set[str],
        wait_for_save: bool = True,
    ) -> "KVConnectorOutput | None":
        return None

    def no_forward(
        self, scheduler_output: "SchedulerOutput"
    ) -> "ModelRunnerOutput":
        from my_vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
        return EMPTY_MODEL_RUNNER_OUTPUT

    def set_disabled(self, disabled: bool) -> None:
        pass


class ActiveKVConnector(KVConnector):
    """Concrete wrapper that delegates to a `KVConnectorBase_V1`.

    Mirrors vLLM V1 v1/worker/gpu/kv_connector.py:ActiveKVConnector.

    C137: minimal — constructor stores the `VllmConfig` and the
    `KVConnectorBase_V1` instance; `pre_forward` /
    `post_forward` are no-ops today because the GPUModelRunner
    hot path does not yet call them. A follow-up refactor (阶段 3)
    will wire `GPUModelRunner.execute_model` to call these
    hooks in the right order.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        kv_connector: Any,
        kv_caches_dict: dict[str, Any] | None = None,
    ):
        self.vllm_config = vllm_config
        self.kv_connector = kv_connector
        self.kv_caches_dict = kv_caches_dict or {}
        self._disabled = False


# Module-level no-op singleton (matches vLLM).
NO_OP_KV_CONNECTOR = KVConnector()


def get_kv_connector(
    vllm_config: "VllmConfig",
    kv_caches_dict: dict[str, Any] | None = None,
    kv_connector: Any | None = None,
) -> KVConnector:
    """Return the active KVConnector wrapper for the runner.

    Mirrors vLLM V1 v1/worker/gpu/kv_connector.py:get_kv_connector.
    Returns `NO_OP_KV_CONNECTOR` when no connector is configured;
    otherwise returns an `ActiveKVConnector` wrapping the supplied
    `kv_connector` (typically a `KVConnectorBase_V1` instance from
    `EngineCore._kv_connector_init`).
    """
    if kv_connector is None:
        return NO_OP_KV_CONNECTOR
    return ActiveKVConnector(
        vllm_config=vllm_config,
        kv_connector=kv_connector,
        kv_caches_dict=kv_caches_dict,
    )


__all__ = [
    "ActiveKVConnector",
    "KVConnector",
    "NO_OP_KV_CONNECTOR",
    "get_kv_connector",
]