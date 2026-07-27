"""InputBatch — GPU tensor batch from SchedulerOutput.

Mirrors vLLM V1 v1/worker/gpu/input_batch.py structure.

C019: +block_table_tensor, +slot_mapping fields from BlockTables.
"""

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class InputBatch:
    req_ids: list[str] = field(default_factory=list)
    num_reqs: int = 0
    num_tokens: int = 0
    input_ids: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None
    block_table_tensor: torch.Tensor | None = None
    slot_mapping: torch.Tensor | None = None
    query_start_loc: torch.Tensor | None = None
