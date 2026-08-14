"""TP mapping computation for NIXL KV cache transfers.

C250: extracted verbatim from ``vllm/distributed/kv_transfer/kv_connector/
v1/nixl/tp_mapping.py`` (142 lines).  This module is the **most
algorithmically rich** piece of the NIXL connector and has **no NIXL
runtime dependency** — it's pure data manipulation over
``TransferTopology`` and ``KVCacheSpec`` instances.

Scope decisions:

  * ``numpy`` is a soft import — the module works without numpy
    (falls back to pure Python ``range`` / ``set`` ops) so tests
    on CPU-only envs can exercise the math.
  * ``vllm.distributed.kv_transfer.kv_connector.utils.BlockIds`` /
    ``TransferTopology`` are referenced via duck-typing; my-vllm
    defines minimal stand-ins in this file (``BlockIds`` is
    ``list[int] | np.ndarray``, ``TransferTopology`` is a dataclass).
  * ``vllm.v1.kv_cache_interface.AttentionSpec`` /
    ``MambaSpec`` / ``KVCacheSpec`` — duck-typed; my-vllm only uses
    ``issubclass`` to detect SSM specs (MambaSpec) and the rest
    falls through to AttentionSpec.

References:
  * vLLM tp_mapping: ``vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Soft numpy import — fallback to pure Python when numpy is unavailable.
try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False
    np = None  # type: ignore[assignment]


# BlockIds: list[int] | np.ndarray — duck-typed.
BlockIds = Any


@dataclass(frozen=True)
class TransferTopology:
    """Topology context for TP mapping.

    Mirrors ``vllm.distributed.kv_transfer.kv_connector.utils.TransferTopology``.

    Attributes:
        tp_rank: Local TP rank (0-indexed within the TP group).
        tp_size: Local TP size (number of local TP ranks).
        total_num_kv_heads: Total number of KV heads across all TP ranks
            (= num_kv_heads per rank × tp_size for dense; == num_kv_heads
            for GQA where heads are shared).
        is_mla: Whether this engine uses MLA (Multi-head Latent Attention).
            When True, every TP rank holds a duplicate of the latent
            cache, so reading from any single remote rank suffices.
    """

    tp_rank: int
    tp_size: int
    total_num_kv_heads: int
    is_mla: bool = False


@dataclass(frozen=True)
class ReadSpec:
    """Specification for a single remote block read operation.

    Mirrors ``vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping.ReadSpec``.

    Attributes:
        remote_rank: Remote TP rank to read from.
        local_block_ids: Local block IDs that will receive the read data.
        remote_block_ids: Remote block IDs to read from (same length
            as ``local_block_ids``).
    """

    remote_rank: int
    local_block_ids: BlockIds
    remote_block_ids: BlockIds


@dataclass(frozen=True)
class TPMapping:
    """Complete local-to-remote TP mapping for one remote engine.

    Generated once per remote engine during handshake.  Mirrors
    ``vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping.TPMapping``.

    Attributes:
        source_ranks_per_group: Per KV-cache-group list of remote TP
            ranks to read from (position = local piece index).
        all_source_ranks: Union of all source ranks across groups.
        rank_to_attention_slot: Maps each source rank to its FA head
            slot index (used for cross-attention placement).
        rank_offset_factor: FA head offset factor for hetero-TP
            (D_TP > P_TP).  Zero when not hetero-TP.
    """

    source_ranks_per_group: tuple[tuple[int, ...], ...]
    all_source_ranks: tuple[int, ...]
    rank_to_attention_slot: dict[int, int]
    rank_offset_factor: int


def _is_attention_spec(spec_type: type) -> bool:
    """Whether ``spec_type`` is an attention spec (vs SSM/Mamba)."""
    # Duck-type: any subclass of an ``AttentionSpec``-like base is
    # treated as attention; ``MambaSpec`` is the only known SSM
    # exception in vLLM.
    name = spec_type.__name__
    if "Mamba" in name:
        return False
    return True


def _is_ssm_spec(spec_type: type) -> bool:
    return not _is_attention_spec(spec_type)


def compute_tp_mapping(
    transfer_topology: TransferTopology,
    remote_tp_size: int,
    group_spec_types: tuple[type, ...],
) -> TPMapping:
    """Build the complete local-to-remote TP mapping.

    Mirrors ``vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping
    .compute_tp_mapping``.  Computes source ranks, head slot
    assignments, and the rank offset factor in a single pass.

    The math is independent of any NIXL runtime — it's purely a
    function of ``(local_tp_rank, local_tp_size, remote_tp_size,
    total_num_kv_heads, is_mla)`` and the attention vs SSM
    classification of each KV cache group.

    Args:
        transfer_topology: Local TP topology + KV-head count.
        remote_tp_size: Remote TP size (how many TP ranks the
            producer engine has).
        group_spec_types: One ``KVCacheSpec`` type per local
            KV cache group.  Used to decide which groups need
            SSM-specific TP splitting (Mamba) vs uniform attention.

    Returns:
        ``TPMapping`` with source ranks, attention slots, and the
        hetero-TP rank offset factor.
    """
    tp_rank = transfer_topology.tp_rank
    tp_size = transfer_topology.tp_size
    total_num_kv_heads = transfer_topology.total_num_kv_heads

    # --- Attention source ranks ---
    if transfer_topology.is_mla or tp_size >= remote_tp_size:
        # D (local TP) > P (remote TP): multiple local ranks read
        # different chunks from one remote rank (different KV heads).
        # For MLA, we only need one remote since cache is duplicated.
        attn_ranks = [tp_rank * remote_tp_size // tp_size]
    else:
        # P (remote TP) > D (local TP): one local rank reads from
        # multiple remote ranks.  GQA dedup: when K < remote_tp_size,
        # several remote ranks hold the same KV head.  np.unique
        # keeps only the first rank per unique head.
        abs_tp = remote_tp_size // tp_size
        start = tp_rank * abs_tp
        if _HAS_NUMPY:
            heads = np.arange(start, start + abs_tp) * total_num_kv_heads // remote_tp_size
            _, unique_idx = np.unique(heads, return_index=True)
            attn_ranks = (start + np.sort(unique_idx)).tolist()
        else:
            # Pure-Python fallback: dedup by head index.
            heads = [
                (start + i) * total_num_kv_heads // remote_tp_size
                for i in range(abs_tp)
            ]
            seen: set[int] = set()
            unique: list[int] = []
            for i, h in enumerate(heads):
                if h not in seen:
                    seen.add(h)
                    unique.append(start + i)
            attn_ranks = unique

    # --- SSM source ranks ---
    has_ssm = any(_is_ssm_spec(t) for t in group_spec_types)
    if has_ssm:
        if tp_size < remote_tp_size:
            abs_tp = remote_tp_size // tp_size
            ssm_ranks = list(range(tp_rank * abs_tp, (tp_rank + 1) * abs_tp))
        else:
            ssm_ranks = list(attn_ranks)
    else:
        ssm_ranks = []

    all_ranks = sorted(set(attn_ranks) | set(ssm_ranks))

    # --- Per-group ordered source ranks ---
    source_ranks_per_group = tuple(
        tuple(ssm_ranks) if _is_ssm_spec(t) else tuple(attn_ranks)
        for t in group_spec_types
    )

    # --- Attention head slots ---
    head_to_slot: dict[int, int] = {}
    for i, r in enumerate(attn_ranks):
        head_to_slot[r * total_num_kv_heads // remote_tp_size] = i
    rank_to_attention_slot = {
        r: head_to_slot.get(r * total_num_kv_heads // remote_tp_size, 0)
        for r in all_ranks
    }

    # --- Rank offset factor ---
    if transfer_topology.is_mla or tp_size <= remote_tp_size:
        # We don't index into remote for reading, no offset needed.
        rank_offset_factor = 0
    elif tp_size > total_num_kv_heads:
        local_head = tp_rank * total_num_kv_heads // tp_size
        p_start = attn_ranks[0] * total_num_kv_heads // remote_tp_size
        rank_offset_factor = local_head - p_start
    else:
        # D TP > P TP: we index into remote to read different heads
        # depending on rank.
        rank_offset_factor = tp_rank % (tp_size // remote_tp_size)

    return TPMapping(
        source_ranks_per_group=source_ranks_per_group,
        all_source_ranks=tuple(all_ranks),
        rank_to_attention_slot=rank_to_attention_slot,
        rank_offset_factor=rank_offset_factor,
    )


__all__ = [
    "BlockIds",
    "ReadSpec",
    "TPMapping",
    "TransferTopology",
    "compute_tp_mapping",
]
