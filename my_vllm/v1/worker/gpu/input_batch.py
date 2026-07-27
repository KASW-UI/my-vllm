"""InputBatch — GPU tensor batch from SchedulerOutput.

Mirrors vLLM V1 v1/worker/gpu/input_batch.py structure.

C017: base fields only (req_ids, input_ids, positions, seq_lens).
Extended in later commits:
  C019 → +slot_mapping, +block_table_tensor
  C022 → +query_start_loc, +is_prefilling, +num_computed_tokens_np
"""

from dataclasses import dataclass, field

import torch


@dataclass
class InputBatch:
    req_ids: list[str] = field(default_factory=list)
    num_reqs: int = 0
    num_tokens: int = 0
    input_ids: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None
