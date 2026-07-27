"""GPUModelRunner — batched model execution.

Mirrors vLLM V1 v1/worker/gpu_model_runner.py core structure.

C017: replaces per-request prefill/decode with batched prefill
and centralized decode.  EngineCore calls execute_model() once
per step instead of looping over individual requests.

C019: adds BlockTables GPU tensor management.  New requests'
block IDs are written to the BlockTable; slot_mappings are
computed and attached to InputBatch.

C020: RequestState replaces Python dicts for per-request state
(_steps, _next_token, _req_index).  GPU tensors store token
history, computed token counts, and last sampled tokens.

C022: builds CommonAttentionMetadata from InputBatch +
BlockTables fields. Metadata not yet consumed by model forward
(HF attention → C023 PagedAttention).
"""

import torch
import torch.nn as nn

from my_vllm.cache.kv_cache import KVCache
from my_vllm.generation.sampler import greedy_sample
from my_vllm.v1.attention.metadata import CommonAttentionMetadata
from my_vllm.v1.attention.selector import get_attn_backend
from my_vllm.v1.core.sched.output import SchedulerOutput
from my_vllm.v1.outputs import ModelRunnerOutput
from my_vllm.v1.worker.gpu.block_table import BlockTables
from my_vllm.v1.worker.gpu.input_batch import InputBatch
from my_vllm.v1.worker.gpu.states import RequestState


class GPUModelRunner:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        eos_token_id: int = 2,
        block_size: int = 16,
        max_num_reqs: int = 64,
        max_num_blocks: int = 256,
        max_num_batched_tokens: int = 4096,
        max_model_len: int = 4096,
    ):
        self.model = model
        self.device = device
        self.eos_token_id = eos_token_id
        self.block_size = block_size

        self._kv_caches: dict[str, KVCache] = {}

        self.request_state = RequestState(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            device=device,
        )

        self.block_tables = BlockTables(
            block_size=block_size,
            max_num_reqs=max_num_reqs,
            max_num_blocks=max_num_blocks,
            max_num_batched_tokens=max_num_batched_tokens,
            device=device,
        )

        self.attn_backend_cls = get_attn_backend()
        self.attn_metadata: CommonAttentionMetadata | None = None

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput:
        new_req_ids = {d.req_id for d in scheduler_output.scheduled_new_reqs}
        sampled_tokens: dict[str, list[int]] = {}

        # Register new requests with RequestState and BlockTables
        prefill_datas = []
        for d in scheduler_output.scheduled_new_reqs:
            if d.req_id not in new_req_ids:
                continue
            if d.req_id not in self.request_state.req_id_to_index:
                self.request_state.add_request(
                    d.req_id, d.prompt_token_ids
                )
            idx = self.request_state.get_index(d.req_id)
            self.block_tables.append_block_ids(idx, d.block_ids)
            prefill_datas.append(d)

        # 1. Batch prefill
        if prefill_datas:
            prefill_results = self._prefill_batch(prefill_datas)
            for rid, token in prefill_results.items():
                self.request_state.set_last_sampled_token(rid, token)
                sampled_tokens.setdefault(rid, []).append(token)

        # 2. Decode — per-request with HF past_key_values
        for rid in scheduler_output.scheduled_cached_reqs.req_ids:
            if rid not in self._kv_caches:
                continue
            token = self._decode_one(rid)
            if token is not None:
                self.request_state.append_output_token(rid, token)
                sampled_tokens.setdefault(rid, []).append(token)

        return ModelRunnerOutput(
            req_ids=list(sampled_tokens.keys()),
            sampled_token_ids=sampled_tokens,
        )

    def _prefill_batch(
        self, new_datas: list
    ) -> dict[str, int]:
        results: dict[str, int] = {}
        if not new_datas:
            return results

        max_len = max(len(d.prompt_token_ids) for d in new_datas)
        batch_size = len(new_datas)

        input_ids = torch.zeros(
            batch_size, max_len, dtype=torch.long, device=self.device
        )
        attention_mask = torch.zeros(
            batch_size, max_len, dtype=torch.long, device=self.device
        )
        positions = torch.zeros(
            batch_size, max_len, dtype=torch.long, device=self.device
        )

        req_ids = []
        for i, d in enumerate(new_datas):
            prompt = d.prompt_token_ids
            plen = len(prompt)
            input_ids[i, :plen] = torch.tensor(
                prompt, dtype=torch.long, device=self.device
            )
            attention_mask[i, :plen] = 1
            positions[i, :plen] = torch.arange(
                plen, dtype=torch.long, device=self.device
            )
            req_ids.append(d.req_id)

        # Compute slot_mappings
        query_start = torch.tensor(
            [0] + [len(d.prompt_token_ids) for d in new_datas],
            dtype=torch.long, device=self.device,
        ).cumsum(0)
        flat_positions = positions.flatten()[:max_len * batch_size]
        num_tokens = batch_size * max_len
        self.block_tables.compute_slot_mappings(
            query_start, flat_positions, num_tokens
        )

        self.attn_metadata = self._build_attn_metadata(
            num_reqs=batch_size,
            query_start_loc=query_start,
            positions=flat_positions,
            num_tokens=num_tokens,
            max_query_len=max_len,
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

        logits = outputs.logits
        past_kvs = outputs.past_key_values

        for i, rid in enumerate(req_ids):
            prompt_len = len(new_datas[i].prompt_token_ids)
            per_req_kv = []
            for layer_k, layer_v in past_kvs:
                per_req_kv.append((layer_k[i:i + 1], layer_v[i:i + 1]))
            cache = KVCache()
            cache._cached = tuple(per_req_kv)
            self._kv_caches[rid] = cache

            token_logits = logits[i, prompt_len - 1, :]
            token = torch.argmax(token_logits).item()
            results[rid] = token

        return results

    def _decode_one(self, request_id: str) -> int | None:
        cache = self._kv_caches.get(request_id)
        if cache is None:
            return None

        token = self.request_state.get_last_token(request_id)
        if token == 0 and self.request_state.get_step(request_id) > 0:
            return None

        next_input = torch.tensor([token], device=self.device)
        logits = cache.decode(self.model, next_input)
        next_token = greedy_sample(logits).item()
        self.request_state.set_last_sampled_token(request_id, next_token)

        if next_token == self.eos_token_id:
            return None

        return next_token

    def _build_attn_metadata(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        num_tokens: int,
        max_query_len: int,
    ) -> CommonAttentionMetadata:
        """Build CommonAttentionMetadata for the current batch."""
        seq_lens = torch.tensor(
            [self.request_state.get_step(rid) for rid in
             self.request_state.req_id_to_index],
            dtype=torch.int32, device=self.device,
        )[:num_reqs]

        return CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc.cpu(),
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.cpu(),
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            max_query_len=max_query_len,
            max_seq_len=max_query_len,
            block_table_tensor=self.block_tables.block_table[:num_reqs],
            slot_mapping=self.block_tables.slot_mappings[:num_tokens],
            positions=positions,
        )

    # --- legacy methods (bridge compat) ---

    def _prefill_one(
        self, request_id: str,
        input_ids: torch.Tensor, cache: KVCache
    ) -> int:
        logits = cache.prefill(self.model, input_ids)
        self._kv_caches[request_id] = cache
        token = greedy_sample(logits).item()
        if request_id not in self.request_state.req_id_to_index:
            self.request_state.add_request(request_id, [0])
        self.request_state.set_last_sampled_token(request_id, token)
        return token

    def _decode_one_legacy(
        self, request_id: str, cache: KVCache
    ) -> int:
        self._kv_caches[request_id] = cache
        token = self.request_state.get_last_token(request_id)
        if token == 0:
            return self.eos_token_id
        next_input = torch.tensor([token], device=self.device)
        logits = cache.decode(self.model, next_input)
        next_token = greedy_sample(logits).item()
        self.request_state.set_last_sampled_token(request_id, next_token)
        self.request_state.append_output_token(request_id, next_token)
        return next_token

    def get_step(self, request_id: str) -> int:
        return self.request_state.get_step(request_id)

    def cleanup(self, request_id: str):
        self._kv_caches.pop(request_id, None)
        idx = self.request_state.req_id_to_index.get(request_id)
        if idx is not None:
            self.block_tables.num_blocks[idx] = 0
        self.request_state.remove_request(request_id)
