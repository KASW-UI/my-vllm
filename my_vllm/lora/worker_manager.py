"""LoRAWorkerManager — per-worker adapter dispatch.

Mirrors vLLM V1 lora/worker_manager.py:LoRAWorkerManager.

vLLM's `LoRAWorkerManager` handles the worker-side adapter
lifecycle in a multi-worker setup:
  - `add_lora(...)` / `remove_lora(...)` broadcast to all
    workers (or only the local rank for fully-sharded LoRAs).
  - `set_lora(lora_int_id)` propagates the active-adapter id to
    every worker's `LoRAModel`.
  - `list_adapters()` returns the union across all workers.

C146: minimal subset — class skeleton with the public API.
For my-vllm's single-process default (`world_size == 1`),
`add_lora` / `remove_lora` are local calls; the broadcast is
a no-op. A follow-up wires real multi-worker dispatch.
"""

from __future__ import annotations

import logging
from typing import Any

from my_vllm.lora.model_manager import LoRAModelManager
from my_vllm.lora.request import LoRARequest

logger = logging.getLogger(__name__)


class LoRAWorkerManager:
    """Per-worker adapter dispatch.

    Mirrors vLLM V1 lora/worker_manager.py:LoRAWorkerManager (subset).

    C146: skeleton — single-process default. `add_lora` /
    `remove_lora` / `set_lora` delegate to the local
    `LoRAModelManager`. Multi-worker broadcast is a no-op
    (placeholder for a follow-up).
    """

    def __init__(
        self,
        model_manager: LoRAModelManager,
        device: Any | None = None,
    ):
        self.model_manager = model_manager
        self.device = device

    def add_lora(self, lora_request: LoRARequest) -> bool:
        """Add an adapter locally; broadcast to workers (no-op stub)."""
        return self.model_manager.add_lora(lora_request)

    def remove_lora(self, lora_int_id: int) -> bool:
        """Remove an adapter locally; broadcast to workers (no-op stub)."""
        return self.model_manager.remove_lora(lora_int_id)

    def set_lora(self, lora_int_id: int) -> None:
        """Set the active adapter locally; broadcast to workers."""
        self.model_manager.activate_adapter(lora_int_id)

    def list_adapters(self) -> list[LoRARequest]:
        return self.model_manager.list_adapters()

    def load_adapter(self, lora_request: LoRARequest) -> bool:
        """Load + register + activate an adapter from a path."""
        loaded = self.model_manager.load_adapter(lora_request)
        if loaded:
            self.add_lora(lora_request)
            self.set_lora(lora_request.lora_int_id)
        else:
            # Even without weights loaded, register the request
            # so the scheduler knows the id is valid.
            self.add_lora(lora_request)
        return loaded


__all__ = ["LoRAWorkerManager"]