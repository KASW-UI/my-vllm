import torch
import torch.nn as nn

from my_vllm.generation.sampler import greedy_sample
from my_vllm.v1.request import Request, RequestStatus


class BatchExecutor:
    """Static batch executor.

    Locks N requests together: pads prompts to max length, then runs
    decode in lockstep.  Finished requests keep consuming GPU until
    the *last* request stops—this is the fundamental waste of static
    batching (fixed by dynamic batching in C006).
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

    def execute(
        self,
        requests: list[Request],
        max_new_tokens: int = 50,
        eos_token_id: int = 2,
    ):
        n = len(requests)
        active = [True] * n
        output_tokens: list[list[int]] = [[] for _ in range(n)]

        for req in requests:
            req.status = RequestStatus.RUNNING

        # ── Prefill: right-pad to max prompt length ──
        max_len = max(len(r.prompt_token_ids) for r in requests)
        batch_ids = []
        batch_mask = []
        prompt_lens = []
        for req in requests:
            plen = len(req.prompt_token_ids)
            prompt_lens.append(plen)
            ids = req.prompt_token_ids + [eos_token_id] * (max_len - plen)
            mask = [1] * plen + [0] * (max_len - plen)
            batch_ids.append(ids)
            batch_mask.append(mask)

        input_ids = torch.tensor(batch_ids, device=self.device)
        attn_mask = torch.tensor(batch_mask, device=self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                use_cache=True,
            )
        past_key_values = outputs.past_key_values

        # Sample first token for each request (last non-pad position)
        logits = outputs.logits
        cur_tokens = []
        for i in range(n):
            last_pos = prompt_lens[i] - 1
            tok = greedy_sample(
                logits[i : i + 1, last_pos : last_pos + 1, :]
            ).item()
            cur_tokens.append(tok)

        # ── Decode loop (lockstep) ──
        for step in range(1, max_new_tokens):
            batch_input = torch.tensor([cur_tokens], device=self.device).T

            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = outputs.past_key_values

            for i in range(n):
                if not active[i]:
                    continue
                tok = greedy_sample(
                    outputs.logits[i : i + 1, -1:, :]
                ).item()
                cur_tokens[i] = tok

                if tok == eos_token_id:
                    active[i] = False
                    requests[i].status = RequestStatus.FINISHED_STOPPED
                output_tokens[i].append(tok)

            if not any(active):
                break

        # ── Finalize ──
        for i in range(n):
            if active[i]:
                requests[i].status = RequestStatus.FINISHED_LENGTH_CAPPED
            requests[i].output_token_ids = output_tokens[i]
