"""LoRAModelManager — adapter loading + registration.

Mirrors vLLM V1 lora/model_manager.py:LoRAModelManager.

vLLM's `LoRAModelManager` handles:
  - Loading a HuggingFace PEFT checkpoint from a path
    (`from_local_checkpoint(...)`) and mapping the per-module
    weights to the registered `lora_modules` (`BaseLayer`s).
  - Adapting the rank + alpha to each module (some modules may
    have a different rank than the checkpoint's nominal rank).
  - Tracking the active adapter for each request.

C146: minimal subset — class skeleton with the public API
(`add_lora`, `remove_lora`, `load_adapter`, `activate_adapter`).
The actual weight-mapping from a HF checkpoint is out of scope
for my-vllm's minimal Dense-Linear subset (the runner can
construct adapters via `BaseLinearLayer.create_lora_weights`
directly).
"""

from __future__ import annotations

import logging
from typing import Any

from my_vllm.lora.lora_model import LoRAModel
from my_vllm.lora.request import LoRARequest

logger = logging.getLogger(__name__)


class LoRAModelManager:
    """Adapter lifecycle + checkpoint loading.

    Mirrors vLLM V1 lora/model_manager.py:LoRAModelManager (subset).

    C146: skeleton — `add_lora` / `remove_lora` register /
    unregister adapters. `load_adapter` is a stub that
    acknowledges the path and returns; the real HF-Peft
    weight mapping is out of scope per docs/alignment-plan.md
    §8 ("Dense Linear 的一条 LoRA 前向路径即可证明理解，
    不追求全 backend").
    """

    def __init__(self, model: LoRAModel):
        self.model = model

    def add_lora(self, lora_request: LoRARequest) -> bool:
        """Register an adapter with the model."""
        self.model.add_lora(lora_request)
        logger.info(
            "LoRA adapter registered: id=%d name=%s",
            lora_request.lora_int_id, lora_request.lora_name,
        )
        return True

    def remove_lora(self, lora_int_id: int) -> bool:
        """Remove an adapter from the model."""
        self.model.remove_lora(lora_int_id)
        logger.info("LoRA adapter removed: id=%d", lora_int_id)
        return True

    def load_adapter(self, lora_request: LoRARequest) -> bool:
        """Load an adapter from `lora_request.path`.

        C146: stub — acknowledge the path and return False
        (no weights loaded). Real impl parses the HF PEFT
        checkpoint and populates each registered LoRA module's
        `create_lora_weights` / per-module A/B tensors. Out of
        scope for the alignment-phase minimal subset.
        """
        if not lora_request.path:
            logger.warning(
                "LoRA %s has no path; skipping load",
                lora_request.lora_name,
            )
            return False
        logger.info(
            "LoRA %s load_adapter(%s): stub — HF PEFT mapping deferred",
            lora_request.lora_name, lora_request.path,
        )
        return False

    def activate_adapter(self, lora_int_id: int) -> None:
        """Mark `lora_int_id` as the active adapter on every module."""
        self.model.set_lora(lora_int_id)

    def list_adapters(self) -> list[LoRARequest]:
        return self.model.list_loras()


__all__ = ["LoRAModelManager"]