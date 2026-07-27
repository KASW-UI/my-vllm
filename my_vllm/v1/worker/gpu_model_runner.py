"""GPUModelRunner — batched model execution.

Mirrors vLLM V1 v1/worker/gpu_model_runner.py core structure.

C017: replaces per-request prefill/decode with batched prefill
and centralized decode.  EngineCore calls execute_model() once
per step instead of looping over individual requests.

C019: adds BlockTables GPU tensor management.  New requests'
block IDs are written to the BlockTable; slot_mappings are
computed and attached to InputBatch.
"""

import torch
import torch.nn as nn

from my_vllm.cache.kv_cache import KVCache
from my_vllm.generation.sampler import greedy_sample
from my_vllm.v1.core.sched.output import SchedulerOutput
from my_vllm.v1.outputs import ModelRunnerOutput
from my_vllm.v1.worker.gpu.block_table import BlockTables
from my_vllm.v1.worker.gpu.input_batch import InputBatch


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
    ):
        self.model = model
        self.device = device
        self.eos_token_id = eos_token_id
        self.block_size = block_size

        self._kv_caches: dict[str, KVCache] = {}
        self._steps: dict[str, int] = {}
        self._next_token: dict[str, int] = {}

        # Per-request GPU slot index (reuses slot when request finishes)
        self._req_index: dict[str, int] = {}
        self._free_indices: list[int] = list(range(max_num_reqs))

        self.block_tables = BlockTables(
            block_size=block_size,
            max_num_reqs=max_num_reqs,
            max_num_blocks=max_num_blocks,
            max_num_batched_tokens=max_num_batched_tokens,
            device=device,
        )

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput:
        new_req_ids = {d.req_id for d in scheduler_output.scheduled_new_reqs}
        sampled_tokens: dict[str, list[int]] = {}

        # Assign GPU slots to new requests and populate block tables
        prefill_datas = []
        for d in scheduler_output.scheduled_new_reqs:
            if d.req_id not in new_req_ids:
                continue
            if d.req_id not in self._req_index:
                idx = self._acquire_index()
                self._req_index[d.req_id] = idx
            idx = self._req_index[d.req_id]
            self.block_tables.append_block_ids(idx, d.block_ids)
            prefill_datas.append(d)

        # 1. Batch prefill
        if prefill_datas:
            prefill_results = self._prefill_batch(prefill_datas)
            for rid, token in prefill_results.items():
                self._steps[rid] = 1
                self._next_token[rid] = token
                sampled_tokens.setdefault(rid, []).append(token)

        # 2. Decode — per-request with HF past_key_values
        for rid in scheduler_output.scheduled_cached_reqs.req_ids:
            if rid not in self._kv_caches:
                continue
            token = self._decode_one(rid)
            if token is not None:
                self._steps[rid] = self._steps.get(rid, 0) + 1
                sampled_tokens.setdefault(rid, []).append(token)

        return ModelRunnerOutput(
            req_ids=list(sampled_tokens.keys()),
            sampled_token_ids=sampled_tokens,
        )

    def _acquire_index(self) -> int:
        if self._free_indices:
            return self._free_indices.pop(0)
        raise RuntimeError("No free request slots — increase max_num_reqs")

    def _release_index(self, request_id: str):
        idx = self._req_index.pop(request_id, None)
        if idx is not None:
            self._free_indices.append(idx)
            self.block_tables.num_blocks[idx] = 0

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

        # Compute slot_mappings for prefill tokens
        req_indices = [
            self._req_index.get(rid, 0) for rid in req_ids
        ]
        query_start = torch.tensor(
            [0] + [len(d.prompt_token_ids) for d in new_datas],
            dtype=torch.long, device=self.device,
        ).cumsum(0)
        flat_positions = positions.flatten()[:max_len * batch_size]
        num_tokens = batch_size * max_len
        self.block_tables.compute_slot_mappings(
            query_start, flat_positions, num_tokens
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

        token = self._next_token.get(request_id)
        if token is None:
            return None

        next_input = torch.tensor([token], device=self.device)
        logits = cache.decode(self.model, next_input)
        next_token = greedy_sample(logits).item()
        self._next_token[request_id] = next_token

        if next_token == self.eos_token_id:
            return None

        return next_token

    def _prefill_one(
        self, request_id: str,
        input_ids: torch.Tensor, cache: KVCache
    ) -> int:
        logits = cache.prefill(self.model, input_ids)
        self._kv_caches[request_id] = cache
        token = greedy_sample(logits).item()
        self._steps[request_id] = 1
        self._next_token[request_id] = token
        return token

    def _decode_one_legacy(
        self, request_id: str, cache: KVCache
    ) -> int:
        self._kv_caches[request_id] = cache
        token = self._next_token.get(request_id)
        if token is None:
            return self.eos_token_id
        next_input = torch.tensor([token], device=self.device)
        logits = cache.decode(self.model, next_input)
        next_token = greedy_sample(logits).item()
        self._next_token[request_id] = next_token
        self._steps[request_id] = self._steps.get(request_id, 0) + 1
        return next_token

    def get_step(self, request_id: str) -> int:
        return self._steps.get(request_id, 0)

    def cleanup(self, request_id: str):
        self._kv_caches.pop(request_id, None)
        self._steps.pop(request_id, None)
        self._next_token.pop(request_id, None)
        self._release_index(request_id)
