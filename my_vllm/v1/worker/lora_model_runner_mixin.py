"""LoRA mixin for GPUModelRunner — adapter hooks.

Mirrors vLLM V1 v1/worker/lora_model_runner_mixin.py:
LoRAModelRunnerMixin.

vLLM's `LoRAModelRunnerMixin` is mixed into `GPUModelRunner`
to provide:
  - `add_lora(lora_request)` / `remove_lora(lora_int_id)` —
    lifecycle hooks called from the scheduler output.
  - `set_lora(lora_int_id)` — per-step active-adapter swap.
  - `list_loras()` — debugging helper.
  - `update_lora_modules()` — called after model load to wrap
    registered modules with `BaseLayer` subclasses.

C146: minimal subset — class skeleton with the public API.
The `update_lora_modules()` helper walks `model.named_modules()`
and wraps each `Linear` / `ColumnParallelLinear` /
`RowParallelLinear` / `MergedColumnParallelLinear` / `QKVParallelLinear`
with the corresponding LoRA subclass. For my-vllm's default
(no LoRA active) the wrappers are dormant — they only get
invoked when an adapter is set via `set_lora(id)`.
"""

from __future__ import annotations

import logging
from typing import Any

from my_vllm.lora.lora_model import LoRAModel
from my_vllm.lora.request import LoRARequest
from my_vllm.lora.worker_manager import LoRAWorkerManager

logger = logging.getLogger(__name__)


class LoRAModelRunnerMixin:
    """Mix into `GPUModelRunner` to add LoRA support.

    Mirrors vLLM V1 v1/worker/lora_model_runner_mixin.py:
    LoRAModelRunnerMixin (subset).

    C146: skeleton — `add_lora` / `remove_lora` / `set_lora` /
    `list_loras` delegate to the worker_manager. The actual
    per-step active-adapter swap is not wired into
    `execute_model` (follow-up).
    """

    def init_lora(
        self,
        model: Any,
        max_loras: int,
        max_lora_rank: int,
        device: Any | None = None,
    ) -> None:
        """Initialise the LoRA model + managers.

        Called from `GPUModelRunner.load_model()` after the base
        model is on-device.
        """
        self.lora_model = LoRAModel(
            model=model,
            max_loras=max_loras,
            max_lora_rank=max_lora_rank,
        )
        # Lazy import to avoid a circular import on
        # my_vllm.lora.model_manager → my_vllm.lora.lora_model.
        from my_vllm.lora.model_manager import LoRAModelManager
        self.lora_model_manager = LoRAModelManager(model=self.lora_model)
        self.lora_worker_manager = LoRAWorkerManager(
            model_manager=self.lora_model_manager,
            device=device,
        )
        logger.info(
            "LoRA init: max_loras=%d max_lora_rank=%d",
            max_loras, max_lora_rank,
        )

    def update_lora_modules(self) -> None:
        """Walk the model and wrap registered Linear modules.

        C146: stub — no-op. Real impl walks
        `self.model.named_modules()`, finds the
        `target_modules`-named layers, and replaces them with
        the corresponding LoRA subclass. Deferred until
        my-vllm's `OPTForCausalLM` is wired to use a
        `lora_target_modules` config.
        """
        return None

    def add_lora(self, lora_request: LoRARequest) -> bool:
        """Add a LoRA adapter (called from the scheduler)."""
        return self.lora_worker_manager.add_lora(lora_request)

    def remove_lora(self, lora_int_id: int) -> bool:
        """Remove a LoRA adapter."""
        return self.lora_worker_manager.remove_lora(lora_int_id)

    def set_lora(self, lora_int_id: int) -> None:
        """Set the active adapter for the next forward step."""
        self.lora_worker_manager.set_lora(lora_int_id)

    def list_loras(self) -> list[LoRARequest]:
        """Return all registered adapters."""
        return self.lora_worker_manager.list_adapters()


__all__ = ["LoRAModelRunnerMixin"]