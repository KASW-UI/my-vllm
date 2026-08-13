"""RequestState — GPU-side per-request state management.

Mirrors vLLM V1 v1/worker/gpu/states.py:9.

C020: replaces Python dicts (_steps, _next_token, _req_index) with
GPU tensor-backed state.  Uses regular torch.Tensor (no StagedWrite/UVA)
since enforce_eager=True.
"""

import numpy as np
import torch


class RequestState:
    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        device: torch.device,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.device = device

        self.req_id_to_index: dict[str, int] = {}
        self.index_to_req_id: dict[int, str] = {}
        self.free_indices = list(range(max_num_reqs))

        # Per-request LoRA adapter id (C148). Set when the
        # request is added in `_update_states` from
        # `NewRequestData.lora_int_id`; cleared on
        # `remove_request`. The runner uses this to decide which
        # LoRA adapter to activate per step.
        self.req_id_to_lora_id: dict[str, int | None] = {}

        # [max_num_reqs, max_model_len] int32 — all token IDs per request
        self.all_token_ids = torch.zeros(
            max_num_reqs, max_model_len,
            dtype=torch.int32, device=device,
        )

        # [max_num_reqs] int32 — number of computed tokens per request
        self.num_computed_tokens_cpu = np.zeros(
            max_num_reqs, dtype=np.int32
        )

        # [max_num_reqs, 1] int64 — last sampled token per request
        self.last_sampled_tokens = torch.zeros(
            max_num_reqs, 1, dtype=torch.int64, device=device,
        )

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    def add_request(
        self,
        req_id: str,
        prompt_token_ids: list[int],
    ) -> int:
        assert len(self.free_indices) > 0, "No free indices"
        idx = self.free_indices.pop()
        self.req_id_to_index[req_id] = idx
        self.index_to_req_id[idx] = req_id

        plen = len(prompt_token_ids)
        self.all_token_ids[idx, :plen] = torch.tensor(
            prompt_token_ids, dtype=torch.int32, device=self.device
        )
        self.num_computed_tokens_cpu[idx] = plen
        self.last_sampled_tokens[idx, 0] = 0
        return idx

    def append_output_token(
        self, req_id: str, token_id: int
    ) -> None:
        """Append a new output token and increment computed count."""
        idx = self.req_id_to_index[req_id]
        pos = int(self.num_computed_tokens_cpu[idx])
        self.all_token_ids[idx, pos] = token_id
        self.num_computed_tokens_cpu[idx] = pos + 1

    def set_last_sampled_token(
        self, req_id: str, token_id: int
    ) -> None:
        idx = self.req_id_to_index[req_id]
        self.last_sampled_tokens[idx, 0] = token_id

    def get_last_token(self, req_id: str) -> int:
        idx = self.req_id_to_index.get(req_id)
        if idx is None:
            return 0
        return int(self.last_sampled_tokens[idx, 0].item())

    def get_step(self, req_id: str) -> int:
        idx = self.req_id_to_index.get(req_id)
        if idx is None:
            return 0
        return int(self.num_computed_tokens_cpu[idx])

    def remove_request(self, req_id: str) -> int | None:
        idx = self.req_id_to_index.pop(req_id, None)
        if idx is None:
            return None
        self.index_to_req_id.pop(idx, None)
        self.free_indices.append(idx)
        self.all_token_ids[idx].zero_()
        self.num_computed_tokens_cpu[idx] = 0
        self.last_sampled_tokens[idx].zero_()
        # C148: drop any LoRA mapping for this request.
        self.req_id_to_lora_id.pop(req_id, None)
        return idx

    def get_index(self, req_id: str) -> int:
        return self.req_id_to_index[req_id]
