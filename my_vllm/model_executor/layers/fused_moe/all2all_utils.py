"""my_vllm/model_executor/layers/fused_moe/all2all_utils.py

Factory + helpers for picking the EP/DP PrepareAndFinalize.

C240: extracted from ``vllm/model_executor/layers/fused_moe/all2all_utils.py``
(361 lines), with the following scope reductions per AGENTS.md
JD-driven scope:

  * ``maybe_make_prepare_finalize`` keeps the AG+RS branch (when
    ``use_ag_rs_all2all_kernels``) and the NoDPEP branch (default).
  * All other backends (DeepEP HT/LL/V2, NIXL EP, FlashInfer NVLink,
    MoRI, Batched) are short-circuited to ``None`` with a log
    message — callers must explicitly opt into them when those
    external kernels are vendored.
  * ``get_ep_all2all_manager`` is kept (returns the
    ``All2AllManagerBase`` attached to the EP group's device
    communicator).
  * ``maybe_roundup_layer_hidden_size`` — kept (the round-up
    helper for FP8 per-tensor scales).

References:
  * vLLM ``all2all_utils.py:59-360``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from my_vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEPrepareAndFinalize,
)
from my_vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
    make_moe_prepare_and_finalize_naive_dp_ep,
)
from my_vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    make_moe_prepare_and_finalize_no_dp_ep,
)

logger = logging.getLogger(__name__)


def get_ep_all2all_manager(eep_stage: bool = False) -> Any | None:
    """Return the ``All2AllManagerBase`` attached to the EP group's device communicator.

    C240: mirrors ``vllm/model_executor/layers/fused_moe/all2all_utils
    .get_ep_all2all_manager``.  Returns ``None`` when no EP group is
    initialized (the my-vllm default for single-process / dense
    inference).

    When ``eep_stage=True``, returns the elastic-EP stage manager
    if available.  The elastic-EP manager itself is a C230 concern;
    my-vllm wires it up in C247.
    """
    try:
        from my_vllm.distributed import get_ep_group
    except ImportError:
        return None
    ep_group = get_ep_group()
    if ep_group is None:
        return None
    device_communicator = getattr(ep_group, "device_communicator", None)
    if device_communicator is None:
        return None
    return device_communicator.all2all_manager


def maybe_roundup_layer_hidden_size(
    hidden_size: int,
    act_dtype: torch.dtype,
    moe_parallel_config: Any,
) -> int:
    """Round up ``hidden_size`` to a multiple required by FP8 / INT8 quant.

    C240: mirrors ``vllm/model_executor/layers/fused_moe/all2all_utils
    .maybe_roundup_layer_hidden_size``.  Per-token quantization
    kernels often require ``hidden_size % block == 0``.  For
    my-vllm the only quant is FP8 (C220) which requires alignment
    to 128; without per-token scale hooks yet, this is a no-op.
    """
    if not getattr(moe_parallel_config, "use_all2all_kernels", False):
        return hidden_size
    block_size = 128
    return ((hidden_size + block_size - 1) // block_size) * block_size


def maybe_make_prepare_finalize(
    moe: Any,
    quant_config: Any | None = None,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    allow_new_interface: bool = False,
    use_monolithic: bool = False,
    eep_stage: bool = False,
) -> FusedMoEPrepareAndFinalize | None:
    """Pick the EP/DP PrepareAndFinalize for ``moe``.

    C240: extracted from ``vllm/model_executor/layers/fused_moe/all2all_utils
    .maybe_make_prepare_finalize``.  The simplified dispatch:

      1. If ``moe.moe_parallel_config.use_all2all_kernels`` is False
         (or the EP group is not initialized), return the NoDPEP
         PrepareAndFinalize (identity, no comm).
      2. If ``moe.moe_parallel_config.use_ag_rs_all2all_kernels`` is
         True (the default for the AG+RS backend), return the
         ``NaiveDPEP`` PrepareAndFinalize (AllGather dispatch +
         ReduceScatter combine).
      3. Otherwise, return ``None`` and log a warning (the user
         has selected a backend that my-vllm does not vendor).

    Parameters:
      moe: ``FusedMoEConfig`` (we only read ``.moe_parallel_config``).
      quant_config: ``FusedMoEQuantConfig | None`` — reserved for
        future quantization dispatch.
      routing_tables: reserved for DeepEP / NIXL EP backend selection.
      allow_new_interface: unused (the vLLM compat flag).
      use_monolithic: True for monolithic kernels; False for Modular.
      eep_stage: True for the elastic-EP path (deferred to C247).

    Returns:
      ``FusedMoEPrepareAndFinalize`` instance, or ``None`` when no
      backend applies.
    """
    moe_parallel_config = getattr(moe, "moe_parallel_config", None)
    use_all2all_kernels = bool(
        getattr(moe_parallel_config, "use_all2all_kernels", False)
    )
    use_ag_rs = bool(
        getattr(moe_parallel_config, "use_ag_rs_all2all_kernels", False)
    )

    if not use_all2all_kernels:
        return make_moe_prepare_and_finalize_no_dp_ep(
            use_monolithic=use_monolithic
        )

    if use_ag_rs:
        return make_moe_prepare_and_finalize_naive_dp_ep(
            use_monolithic=use_monolithic,
        )

    backend = getattr(moe_parallel_config, "all2all_backend", "none")
    logger.warning(
        "All2All backend '%s' requested but not vendored in my-vllm "
        "(see AGENTS.md JD-driven scope).  Falling back to NoDPEP "
        "(no all-to-all).", backend,
    )
    return make_moe_prepare_and_finalize_no_dp_ep(
        use_monolithic=use_monolithic
    )


__all__ = [
    "get_ep_all2all_manager",
    "maybe_make_prepare_finalize",
    "maybe_roundup_layer_hidden_size",
]
