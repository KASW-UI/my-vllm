"""LoRAModel — wraps a base model with LoRA-augmented layers.

Mirrors vLLM V1 lora/lora_model.py:LoRAModel.

vLLM's `LoRAModel` is the runtime wrapper that holds:
  - the base model's `nn.Module`
  - a `lora_modules: dict[str, BaseLayer]` mapping the base
    module name → the LoRA-augmented wrapper (e.g. `"q_proj" →
    ColumnParallelLinearLoRA(base=q_proj, ...)`).
  - `add_lora(...)` / `remove_lora(...)` / `get_lora(...)` for
    adapter lifecycle.
  - `set_lora(lora_int_id)` / `reset_lora(lora_int_id)` to swap
    the active adapter on all registered modules.

C146: minimal subset — class skeleton + the lifecycle methods.
The actual weight loading from a HF LoRA checkpoint is deferred
to `LoRAModelManager` (next file). The single dense-Linear
forward path is end-to-end runnable via
`BaseLinearLayer.create_lora_weights + set_lora + forward`.
"""

from __future__ import annotations

import logging
from typing import Any

from my_vllm.lora.layers.base import BaseLayer
from my_vllm.lora.request import LoRARequest

logger = logging.getLogger(__name__)


class LoRAModel:
    """Runtime wrapper holding base model + LoRA-augmented layers.

    Mirrors vLLM V1 lora/lora_model.py:LoRAModel (subset).
    """

    def __init__(
        self,
        model: Any,
        max_loras: int = 1,
        max_lora_rank: int = 16,
    ):
        self.model = model
        self.max_loras = max_loras
        self.max_lora_rank = max_lora_rank

        # base_module_name → LoRA-augmented wrapper.
        self.lora_modules: dict[str, BaseLayer] = {}

        # lora_int_id → per-adapter metadata (LoRARequest).
        self._adapters: dict[int, LoRARequest] = {}

    def register_module(
        self,
        base_module_name: str,
        lora_module: BaseLayer,
    ) -> None:
        """Register a LoRA-augmented layer under `base_module_name`."""
        self.lora_modules[base_module_name] = lora_module

    def add_lora(self, lora_request: LoRARequest) -> None:
        """Register an adapter (id → LoRARequest).

        For every already-wrapped `BaseLinearLayer` in
        `self.lora_modules`, also allocate the per-adapter A/B
        weight tensors via `create_lora_weights(...)`. This is
        what makes subsequent `set_lora(id)` succeed (each
        wrapper checks `lora_int_id in self._adapters`).

        C147: real implementation (C146 only stored the
        request; the wrappers' `_adapters` dict stayed empty,
        so `set_lora` would raise `KeyError`).
        """
        if lora_request.lora_int_id in self._adapters:
            logger.info(
                "LoRA adapter %d (%s) already registered",
                lora_request.lora_int_id, lora_request.lora_name,
            )
            return
        if len(self._adapters) >= self.max_loras:
            raise RuntimeError(
                f"max_loras={self.max_loras} exceeded; cannot register "
                f"adapter {lora_request.lora_int_id}"
            )
        self._adapters[lora_request.lora_int_id] = lora_request

        # Allocate per-wrapper A/B tensors so `set_lora` succeeds.
        for wrapper in self.lora_modules.values():
            base = wrapper.base_layer
            if base is None:
                continue
            in_features = getattr(base, "input_size", None)
            out_features = getattr(base, "output_size", None)
            if in_features is None or out_features is None:
                logger.debug(
                    "LoRA: skipping weight alloc for %s; cannot "
                    "infer in/out features",
                    type(base).__name__,
                )
                continue
            wrapper.create_lora_weights(
                lora_int_id=lora_request.lora_int_id,
                rank=self.max_lora_rank,
                in_features=in_features,
                out_features=out_features,
                alpha=1.0,
            )

    def remove_lora(self, lora_int_id: int) -> None:
        """Drop an adapter and its per-module weights."""
        self._adapters.pop(lora_int_id, None)
        for module in self.lora_modules.values():
            module.reset_lora(lora_int_id)

    def set_lora(self, lora_int_id: int) -> None:
        """Mark `lora_int_id` as the active adapter on every module."""
        if lora_int_id not in self._adapters:
            raise KeyError(
                f"LoRA adapter {lora_int_id} not registered"
            )
        for module in self.lora_modules.values():
            module.set_lora(lora_int_id)

    def get_lora(self, lora_int_id: int) -> LoRARequest | None:
        return self._adapters.get(lora_int_id)

    def list_loras(self) -> list[LoRARequest]:
        return list(self._adapters.values())

    @property
    def active_lora_int_id(self) -> int | None:
        """The currently-active adapter id (None if no adapter set)."""
        # All modules share the same active adapter id (single-
        # adapter path; multi-adapter SGMV deferred).
        for module in self.lora_modules.values():
            return module.active_lora_int_id
        return None


__all__ = ["LoRAModel"]