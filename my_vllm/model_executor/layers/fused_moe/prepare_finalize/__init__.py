"""my_vllm/model_executor/layers/fused_moe/prepare_finalize — EP/DP Prepare+Finalize.

C240: package init.  Re-exports the ``Naive`` (AG+RS-based) and
``NoDPEP`` (single-rank / no all-to-all) PrepareAndFinalize
implementations + their factories.

Subpackages:
  * ``naive_dp_ep`` — AllGather + ReduceScatter (the reference
    implementation).  Used when ``FusedMoEParallelConfig.use_ag_rs_all2all_kernels``
    is True.
  * ``no_dp_ep`` — identity (no comm).  Used when ``dp_size == 1``
    and ``ep_size == 1`` (the my-vllm default for dense models).

Other backends (DeepEP HT/LL/V2, NIXL EP, FlashInfer NVLink,
MoRI, batched) are not vendored — they live in the vLLM source as
external-library wrappers that my-vllm does not require.
"""

from __future__ import annotations

from my_vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
    MoEPrepareAndFinalizeNaiveDPEPMonolithic,
    MoEPrepareAndFinalizeNaiveDPEPModular,
    make_moe_prepare_and_finalize_naive_dp_ep,
)
from my_vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    MoEPrepareAndFinalizeNoDPEPMonolithic,
    MoEPrepareAndFinalizeNoDPEPModular,
    make_moe_prepare_and_finalize_no_dp_ep,
)


__all__ = [
    "MoEPrepareAndFinalizeNaiveDPEPMonolithic",
    "MoEPrepareAndFinalizeNaiveDPEPModular",
    "MoEPrepareAndFinalizeNoDPEPMonolithic",
    "MoEPrepareAndFinalizeNoDPEPModular",
    "make_moe_prepare_and_finalize_naive_dp_ep",
    "make_moe_prepare_and_finalize_no_dp_ep",
]
