"""CommonAttentionMetadata — shared per-batch attention metadata.

Mirrors vLLM V1 v1/attention/backend.py:412.

C022: combines fields from prior commits into a single struct
for the attention backend interface.
"""

from dataclasses import dataclass

import torch


@dataclass
class CommonAttentionMetadata:
    """Per-batch metadata shared across all attention layers.

    Fields assembled from:
      C017 InputBatch  → query_start_loc, seq_lens, positions
      C019 BlockTables → block_table_tensor, slot_mapping
      C020 RequestState → num_reqs, num_actual_tokens
    """

    query_start_loc: torch.Tensor
    """(batch_size + 1,) start location of each request in the query."""
    query_start_loc_cpu: torch.Tensor
    """CPU copy of query_start_loc."""

    seq_lens: torch.Tensor
    """(batch_size,) number of computed tokens per request."""
    seq_lens_cpu: torch.Tensor
    """CPU copy of seq_lens."""

    num_reqs: int

    num_actual_tokens: int
    """Total number of tokens in the batch."""

    max_query_len: int
    """Longest query in the batch."""

    max_seq_len: int
    """Longest context length."""

    block_table_tensor: torch.Tensor
    """[num_reqs, max_blocks] int32 — paged block table."""

    slot_mapping: torch.Tensor
    """[num_tokens] int64 — slot_id = block_id * block_size + offset."""

    positions: torch.Tensor | None = None
    """(num_actual_tokens,) token positions."""

    causal: bool = True
