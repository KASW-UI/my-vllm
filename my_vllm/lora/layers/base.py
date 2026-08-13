"""LoRA base layer — weight container + reset utilities.

Mirrors vLLM V1 lora/layers/base.py.

vLLM's `BaseLayer` (and its `LoRAMapping` / `BaseLayerWithLoRA`
subclasses) provide:
  - Per-adapter weight containers (one set of A/B matrices per
    `lora_int_id`).
  - `set_lora(id)` / `reset_lora(id)` methods that swap the active
    adapter in / out.
  - `create_lora_weights(...)` helper that allocates the A/B
    parameter tensors.
  - `reset_lora_weights(...)` for unloader-side cleanup.

C144: minimal subset — `BaseLayer` with `set_lora` /
`reset_lora` / `create_lora_weights` (returns a dict of
parameters) / `get_lora_weights` (returns the current adapter's
weights). Per-adapter weight storage is via a `dict[int,
LoRAWeights]` keyed by `lora_int_id`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class LoRAWeights:
    """One adapter's LoRA weight pair.

    Mirrors vLLM V1 lora/layers/base.py:LoRAWeights (subset).

    Fields:
        lora_a: A matrix, shape `[rank, in_features]`.
        lora_b: B matrix, shape `[out_features, rank]`.
        scaling: `lora_alpha / lora_rank` (the standard LoRA scale).
    """

    lora_a: torch.Tensor
    lora_b: torch.Tensor
    scaling: float

    def to(self, device: torch.device | str) -> "LoRAWeights":
        return LoRAWeights(
            lora_a=self.lora_a.to(device),
            lora_b=self.lora_b.to(device),
            scaling=self.scaling,
        )


class BaseLayer:
    """Base for all LoRA-augmented layers.

    Mirrors vLLM V1 lora/layers/base.py:BaseLayerWithLoRA.

    C144: minimal — the active adapter is stored as an integer
    in `self.lora_id_to_loaded_id`. Per-adapter weights live in
    `self._adapters: dict[int, LoRAWeights]` keyed by
    `lora_int_id`. Subclasses override `apply_lora` / `apply_base`
    to do the actual forward.

    Lifecycle:
      1. `create_lora_weights(lora_int_id, rank, in_features,
         out_features)` allocates + registers an adapter's A/B
         parameters.
      2. `set_lora(lora_int_id)` marks the adapter as active.
      3. `forward(x)` (in subclasses) does base + LoRA forward
         using the active adapter's weights.
      4. `reset_lora(lora_int_id)` drops the adapter.
    """

    def __init__(
        self,
        base_layer: Any | None = None,
        max_loras: int = 1,
        max_lora_rank: int = 16,
    ):
        self.base_layer = base_layer
        self.max_loras = max_loras
        self.max_lora_rank = max_lora_rank

        # `lora_id_to_loaded_id` maps a request's `lora_int_id`
        # to the local "slot id" (offset into the packed weights
        # buffer in vLLM; my-vllm uses a simple dict instead).
        self.lora_id_to_loaded_id: dict[int, int] = {}

        # `lora_slot_id_to_adapter_id` is the inverse map.
        self.lora_slot_id_to_adapter_id: dict[int, int] = {}

        # Per-adapter weights, keyed by `lora_int_id`.
        self._adapters: dict[int, LoRAWeights] = {}

    def create_lora_weights(
        self,
        lora_int_id: int,
        rank: int,
        in_features: int,
        out_features: int,
        alpha: float = 1.0,
    ) -> LoRAWeights:
        """Allocate A/B parameters for one adapter.

        Returns the `LoRAWeights` dataclass so callers can also
        populate the values from a checkpoint.
        """
        if lora_int_id in self._adapters:
            return self._adapters[lora_int_id]
        if len(self._adapters) >= self.max_loras:
            raise RuntimeError(
                f"max_loras={self.max_loras} exceeded; cannot register "
                f"adapter {lora_int_id}"
            )
        # A initialized to small random values (Kaiming); B zero
        # (so the initial forward is the base linear + 0).
        a = torch.randn(rank, in_features, dtype=torch.float32) * 0.01
        b = torch.zeros(out_features, rank, dtype=torch.float32)
        weights = LoRAWeights(lora_a=a, lora_b=b, scaling=alpha / rank)
        self._adapters[lora_int_id] = weights
        return weights

    def get_lora_weights(self, lora_int_id: int) -> LoRAWeights | None:
        return self._adapters.get(lora_int_id)

    def set_lora(self, lora_int_id: int) -> None:
        """Mark `lora_int_id` as the active adapter for the next forward."""
        if lora_int_id not in self._adapters:
            raise KeyError(
                f"LoRA adapter {lora_int_id} not registered"
            )
        self.lora_id_to_loaded_id.setdefault(lora_int_id, lora_int_id)
        self.lora_slot_id_to_adapter_id.setdefault(lora_int_id, lora_int_id)

    def reset_lora(self, lora_int_id: int) -> None:
        """Drop an adapter's weights from this layer."""
        self._adapters.pop(lora_int_id, None)
        self.lora_id_to_loaded_id.pop(lora_int_id, None)
        self.lora_slot_id_to_adapter_id.pop(lora_int_id, None)

    @property
    def active_lora_int_id(self) -> int | None:
        """The currently-active adapter id (None if no adapter set)."""
        if not self.lora_id_to_loaded_id:
            return None
        return next(iter(self.lora_id_to_loaded_id))


__all__ = ["BaseLayer", "LoRAWeights"]