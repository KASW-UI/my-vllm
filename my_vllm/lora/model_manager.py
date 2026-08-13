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
C147: adds `create_lora_modules()` — walks `self.model.model
.named_modules()` and replaces target Linear layers with LoRA
wrappers (mirrors vLLM's `_create_lora_modules`). The actual
weight-mapping from a HF checkpoint is out of scope for my-vllm's
minimal Dense-Linear subset (the runner can construct adapters
via `BaseLinearLayer.create_lora_weights` directly).
"""

from __future__ import annotations

import logging
from typing import Any

from my_vllm.lora.layers.column_parallel_linear import ColumnParallelLinearLoRA
from my_vllm.lora.layers.row_parallel_linear import RowParallelLinearLoRA
from my_vllm.lora.lora_model import LoRAModel
from my_vllm.lora.request import LoRARequest
from my_vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)

logger = logging.getLogger(__name__)


# Base-layer-type → LoRA-wrapper class.
#
# Checked in order (most-specific first). QKVParallelLinear and
# MergedColumnParallelLinear both extend ColumnParallelLinear and
# all three currently route to ColumnParallelLinearLoRA (the
# delta is applied in the same partitioned output space as the
# host). RowParallelLinear → RowParallelLinearLoRA (adds the TP
# all_reduce on the LoRA delta).
LORA_LAYER_TYPE_MAP: list[tuple[type, type]] = [
    (QKVParallelLinear, ColumnParallelLinearLoRA),
    (MergedColumnParallelLinear, ColumnParallelLinearLoRA),
    (ColumnParallelLinear, ColumnParallelLinearLoRA),
    (RowParallelLinear, RowParallelLinearLoRA),
]


class LoRAModelManager:
    """Adapter lifecycle + checkpoint loading.

    Mirrors vLLM V1 lora/model_manager.py:LoRAModelManager (subset).

    C146: skeleton — `add_lora` / `remove_lora` register /
    unregister adapters.
    C147: adds `create_lora_modules()` — wraps every target
    Linear module on `self.model.model` (the host `nn.Module`)
    with the matching LoRA subclass, then registers each
    wrapper in `self.model.lora_modules`.
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

    def create_lora_modules(self) -> int:
        """Wrap Linear target modules with LoRA wrappers.

        Mirrors vLLM V1 lora/model_manager.py:LoRAModelManager.
        `_create_lora_modules` (subset). Walks
        `self.model.model.named_modules()` (the host `nn.Module`
        wrapped by `LoRAModel`) and, for every module whose last
        name-component is in `host.lora_target_modules`, replaces
        it in its parent with the matching `BaseLinearLayer`
        subclass.

        The wrapped layers are registered in
        `self.model.lora_modules` under their fully-qualified
        module name so subsequent `set_lora(id)` /
        `forward(x)` calls apply the active adapter's delta
        to the base output.

        Returns:
            Number of modules wrapped.
        """
        host: Any = self.model.model
        if not getattr(host, "supports_lora", False):
            raise RuntimeError(
                f"{host.__class__.__name__} does not declare "
                "`supports_lora=True`; cannot wrap LoRA modules."
            )
        targets: set[str] = set(getattr(host, "lora_target_modules", set()))
        if not targets:
            logger.warning(
                "Model %s has empty lora_target_modules; no LoRA wrapping",
                host.__class__.__name__,
            )
            return 0

        max_loras = self.model.max_loras
        max_lora_rank = self.model.max_lora_rank
        n_wrapped = 0

        # Snapshot named_modules() before mutating — setattr on
        # the parent would invalidate the iterator otherwise.
        for module_name, module in list(host.named_modules()):
            last = module_name.rsplit(".", 1)[-1]
            if last not in targets:
                continue
            wrapper_cls = next(
                (
                    wcls
                    for base_cls, wcls in LORA_LAYER_TYPE_MAP
                    if isinstance(module, base_cls)
                ),
                None,
            )
            if wrapper_cls is None:
                logger.debug(
                    "LoRA: no wrapper for %s (%s); skipping",
                    module_name, type(module).__name__,
                )
                continue
            wrapper = wrapper_cls(
                base_layer=module,
                max_loras=max_loras,
                max_lora_rank=max_lora_rank,
            )
            # Replace in parent. PyTorch tracks `_modules` on
            # setattr — this is the canonical swap pattern.
            if "." in module_name:
                parent_name, attr = module_name.rsplit(".", 1)
                parent = host.get_submodule(parent_name)
            else:
                parent, attr = host, module_name
            setattr(parent, attr, wrapper)
            self.model.register_module(module_name, wrapper)
            n_wrapped += 1

        logger.info(
            "LoRA: wrapped %d target modules on %s (targets=%s)",
            n_wrapped, host.__class__.__name__, sorted(targets),
        )
        return n_wrapped


__all__ = ["LoRAModelManager"]