from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SpecDecodeMetadata:
    draft_token_ids: torch.Tensor       # [num_tokens] flattened draft tokens
    num_draft_tokens: list[int]         # [batch_size] per-request draft count
    cu_num_draft_tokens: torch.Tensor   # [batch_size] cumulative draft tokens
    cu_num_sampled_tokens: torch.Tensor # [batch_size] cumulative sampled (draft+1)
    target_logits_indices: torch.Tensor # [num_tokens] logits to score
    bonus_logits_indices: torch.Tensor  # [batch_size] bonus token logits
    logits_indices: torch.Tensor        # [num_tokens + batch_size] all logits

    def __post_init__(self):
        self.max_spec_len = max(self.num_draft_tokens)

    @classmethod
    def make_dummy(
        cls,
        draft_token_ids: list[list[int]],
        device: torch.device,
    ) -> "SpecDecodeMetadata":
        batch_size = len(draft_token_ids)
        num_draft_tokens = [len(ids) for ids in draft_token_ids]
        num_sampled_tokens = [len(ids) + 1 for ids in draft_token_ids]
        flattened = sum(draft_token_ids, [])
        num_tokens = len(flattened)

        draft_tensor = torch.tensor(flattened, dtype=torch.int32, device=device)
        cu_draft = torch.from_numpy(np.cumsum(num_draft_tokens, dtype=np.int32)).to(device)
        cu_sampled = torch.from_numpy(np.cumsum(num_sampled_tokens, dtype=np.int32)).to(device)
        target_indices = torch.zeros(num_tokens, dtype=torch.int32, device=device)
        bonus_indices = torch.zeros(batch_size, dtype=torch.int32, device=device)
        logits_indices = torch.zeros(num_tokens + batch_size, dtype=torch.int32, device=device)

        return cls(
            draft_token_ids=draft_tensor,
            num_draft_tokens=num_draft_tokens,
            cu_num_draft_tokens=cu_draft,
            cu_num_sampled_tokens=cu_sampled,
            target_logits_indices=target_indices,
            bonus_logits_indices=bonus_indices,
            logits_indices=logits_indices,
        )
