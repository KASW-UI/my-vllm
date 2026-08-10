"""GPUModelRunner — batched model execution.

Mirrors vLLM V1 v1/worker/gpu_model_runner.py core structure.

C017: replaces per-request prefill/decode with batched prefill and centralized decode.
C019: adds BlockTables GPU tensor management.
C020: RequestState replaces Python dicts for per-request state.
C022: builds CommonAttentionMetadata from InputBatch + BlockTables.
C023: physical KV cache pool replaces HF past_key_values.
C024: Sampler (nn.Module) replaces greedy_sample() function.
C032: model loading moved inside load_model().
Unified batching: positions = num_computed_tokens + local_offset (aligned with vLLM L1985).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from my_vllm.v1.attention.metadata import CommonAttentionMetadata
from my_vllm.v1.attention.selector import get_attn_backend
from my_vllm.v1.attention.backends.ops.paged_attn import (
    gather_cached_kv_pytorch,
    reshape_and_cache_pytorch,
)
from my_vllm.v1.core.sched.output import SchedulerOutput
from my_vllm.model_executor.model_loader import load_model
from my_vllm.v1.outputs import ModelRunnerOutput
from my_vllm.v1.sample.metadata import SamplingMetadata
from my_vllm.v1.sample.sampler import Sampler
from my_vllm.v1.worker.gpu.block_table import BlockTables
from my_vllm.v1.worker.gpu.states import RequestState

if TYPE_CHECKING:
    from my_vllm.config.vllm import VllmConfig


def _str_to_torch_dtype(dtype_str: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype_str, torch.float32)


class GPUModelRunner:
    """Batched model executor — mirrors vLLM V1 GPUModelRunner."""

    _EOS_TOKEN_ID = 2

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        mc = vllm_config.model_config
        cc = vllm_config.cache_config
        sc = vllm_config.scheduler_config

        self.model_name = mc.model_name
        self.dtype_str = mc.dtype
        self.block_size = cc.block_size
        self.max_num_blocks = cc.num_kv_blocks
        self.device = device

        self.model: nn.Module | None = None
        self.num_layers: int = 0
        self.num_kv_heads: int = 0
        self.head_dim: int = 0
        self.dtype: torch.dtype = _str_to_torch_dtype(mc.dtype)

        self.sampler = Sampler()
        self.attn_backend_cls = get_attn_backend()
        self.attn_metadata: CommonAttentionMetadata | None = None

        self._init_kwargs = dict(
            max_num_reqs=sc.max_num_seqs,
            max_num_blocks=cc.num_kv_blocks,
            max_num_batched_tokens=sc.max_num_batched_tokens,
            max_model_len=mc.max_model_len,
        )

    def load_model(self) -> None:
        model = load_model(self.model_name, dtype=self.dtype_str)
        self.model = model
        self.device = next(model.parameters()).device

        config = model.config
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = getattr(
            config, "num_key_value_heads", config.num_attention_heads
        )
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.dtype = next(model.parameters()).dtype

        self.request_state = RequestState(
            max_num_reqs=self._init_kwargs["max_num_reqs"],
            max_model_len=self._init_kwargs["max_model_len"],
            device=self.device,
        )
        self.block_tables = BlockTables(
            block_size=self.block_size,
            max_num_reqs=self._init_kwargs["max_num_reqs"],
            max_num_blocks=self._init_kwargs["max_num_blocks"],
            max_num_batched_tokens=self._init_kwargs["max_num_batched_tokens"],
            device=self.device,
        )

        self.kv_cache_k = torch.zeros(
            self.num_layers, self.max_num_blocks, self.num_kv_heads,
            self.block_size, self.head_dim,
            dtype=self.dtype, device=self.device,
        )
        self.kv_cache_v = torch.zeros(
            self.num_layers, self.max_num_blocks, self.num_kv_heads,
            self.block_size, self.head_dim,
            dtype=self.dtype, device=self.device,
        )

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput:
        self._update_states(scheduler_output)

        sampled_tokens: dict[str, list[int]] = {}

        new_reqs = scheduler_output.scheduled_new_reqs
        cached = scheduler_output.scheduled_cached_reqs

        if new_reqs:
            prefill_results = self._execute_prefill(new_reqs)
            for rid, token in prefill_results.items():
                self.request_state.set_last_sampled_token(rid, token)
                sampled_tokens.setdefault(rid, []).append(token)

        decode_req_ids = cached.req_ids
        if decode_req_ids:
            decode_results = self._execute_decode(decode_req_ids)
            for rid, token in decode_results.items():
                if token is not None:
                    self.request_state.append_output_token(rid, token)
                    sampled_tokens.setdefault(rid, []).append(token)

        req_ids = list(sampled_tokens.keys())
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=[sampled_tokens[rid] for rid in req_ids],
        )

    def _update_states(self, scheduler_output: SchedulerOutput) -> None:
        for d in scheduler_output.scheduled_new_reqs:
            if d.req_id not in self.request_state.req_id_to_index:
                self.request_state.add_request(d.req_id, d.prompt_token_ids)
            idx = self.request_state.get_index(d.req_id)
            self.block_tables.append_block_ids(idx, d.block_ids)

        cached = scheduler_output.scheduled_cached_reqs
        for i, rid in enumerate(cached.req_ids):
            new_bids = cached.new_block_ids[i]
            if new_bids is not None:
                idx = self.request_state.get_index(rid)
                self.block_tables.append_block_ids(idx, new_bids)

    def _execute_prefill(self, new_datas: list) -> dict[str, int]:
        results: dict[str, int] = {}
        if not new_datas:
            return results

        effective_lens = []
        for d in new_datas:
            start = d.num_computed_tokens
            effective_lens.append(len(d.prompt_token_ids) - start)

        max_len = max(effective_lens)
        batch_size = len(new_datas)

        input_ids = torch.zeros(
            batch_size, max_len, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros(
            batch_size, max_len, dtype=torch.long, device=self.device)

        req_ids = []
        for i, d in enumerate(new_datas):
            start = d.num_computed_tokens
            chunk = d.prompt_token_ids[start:]
            clen = len(chunk)
            input_ids[i, :clen] = torch.tensor(
                chunk, dtype=torch.long, device=self.device)
            attention_mask[i, :clen] = 1
            req_ids.append(d.req_id)

        positions_tensor = torch.zeros(
            batch_size, max_len, dtype=torch.long, device=self.device)
        for i, d in enumerate(new_datas):
            start = d.num_computed_tokens
            clen = effective_lens[i]
            positions_tensor[i, :clen] = torch.arange(
                start, start + clen, dtype=torch.long, device=self.device)

        query_start = torch.tensor(
            [0] + effective_lens,
            dtype=torch.long, device=self.device,
        ).cumsum(0)
        num_tokens = batch_size * max_len
        flat_pos = positions_tensor.flatten()[:num_tokens]

        self.block_tables.compute_slot_mappings(query_start, flat_pos, num_tokens)

        self.attn_metadata = self._build_attn_metadata(
            num_reqs=batch_size,
            query_start_loc=query_start,
            positions=flat_pos,
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

        self._write_kv_to_paged_cache(past_kvs, new_datas, batch_size, effective_lens)

        token_logits = torch.stack(
            [logits[i, effective_lens[i] - 1, :] for i in range(batch_size)])

        sampled = self.sampler(token_logits, SamplingMetadata(all_greedy=True))

        for i, rid in enumerate(req_ids):
            results[rid] = int(sampled[i].item())

        return results

    def _execute_decode(self, req_ids: list[str]) -> dict[str, int | None]:
        results: dict[str, int | None] = {}
        if not req_ids:
            return results

        valid_reqs = []
        valid_tokens = []
        for rid in req_ids:
            idx = self.request_state.get_index(rid)
            seq_len = int(self.request_state.num_computed_tokens_cpu[idx])
            if seq_len == 0:
                continue
            last_token = int(self.request_state.get_last_token(rid))
            if last_token == 0 and seq_len > 0:
                continue
            valid_reqs.append(rid)
            valid_tokens.append(last_token)

        if not valid_reqs:
            return results

        num_reqs = len(valid_reqs)
        seq_lens = torch.tensor(
            [int(self.request_state.num_computed_tokens_cpu[
                self.request_state.get_index(r)]) for r in valid_reqs],
            dtype=torch.int32, device=self.device,
        )

        block_table = self.block_tables.block_table[:num_reqs]
        past_key_values = []
        for layer_idx in range(self.num_layers):
            keys, values = gather_cached_kv_pytorch(
                self.kv_cache_k[layer_idx],
                self.kv_cache_v[layer_idx],
                block_table, seq_lens, self.block_size,
            )
            past_key_values.append((keys, values))
        past_key_values = tuple(past_key_values)

        next_inputs = torch.tensor(
            [[t] for t in valid_tokens], dtype=torch.long, device=self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=next_inputs,
                past_key_values=past_key_values,
                use_cache=True,
            )

        logits_flat = outputs.logits[:, -1, :]
        new_past_kvs = outputs.past_key_values
        sampled = self.sampler(logits_flat, SamplingMetadata(all_greedy=True))

        decode_slots = []
        for i, rid in enumerate(valid_reqs):
            idx = self.request_state.get_index(rid)
            old_seq_len = int(self.request_state.num_computed_tokens_cpu[idx])
            block_index = old_seq_len // self.block_size
            block_offset = old_seq_len % self.block_size
            block_id = int(
                self.block_tables.block_table[idx, block_index].item())
            decode_slots.append(block_id * self.block_size + block_offset)

        slot_tensor = torch.tensor(decode_slots, dtype=torch.int64, device=self.device)
        for layer_idx, (layer_k, layer_v) in enumerate(new_past_kvs):
            k_last = layer_k[:, :, -1, :]
            v_last = layer_v[:, :, -1, :]
            reshape_and_cache_pytorch(
                k_last, v_last,
                self.kv_cache_k[layer_idx],
                self.kv_cache_v[layer_idx],
                slot_tensor,
            )

        for i, rid in enumerate(valid_reqs):
            next_token = int(sampled[i].item())
            self.request_state.set_last_sampled_token(rid, next_token)
            if next_token == self._EOS_TOKEN_ID:
                results[rid] = None
            else:
                results[rid] = next_token

        return results

    def _write_kv_to_paged_cache(
        self, past_key_values, new_datas: list, batch_size: int,
        effective_lens: list[int] = None,
    ) -> None:
        if effective_lens is None:
            effective_lens = [len(d.prompt_token_ids) for d in new_datas]
        padded_len = max(effective_lens)
        total_tokens = batch_size * padded_len
        slot_map = self.block_tables.slot_mappings[:total_tokens]

        valid = torch.zeros(total_tokens, dtype=torch.bool, device=self.device)
        offset = 0
        for plen in effective_lens:
            valid[offset:offset + plen] = True
            offset += padded_len

        for layer_idx, (layer_k, layer_v) in enumerate(past_key_values):
            bsz, nheads, _, hdim = layer_k.shape
            k_flat = layer_k.permute(0, 2, 1, 3).reshape(
                bsz * padded_len, nheads, hdim)
            v_flat = layer_v.permute(0, 2, 1, 3).reshape(
                bsz * padded_len, nheads, hdim)
            reshape_and_cache_pytorch(
                k_flat[valid], v_flat[valid],
                self.kv_cache_k[layer_idx],
                self.kv_cache_v[layer_idx],
                slot_map[valid],
            )

    def _build_attn_metadata(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        num_tokens: int,
        max_query_len: int,
    ) -> CommonAttentionMetadata:
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

    def get_step(self, request_id: str) -> int:
        return self.request_state.get_step(request_id)

    def cleanup(self, request_id: str):
        idx = self.request_state.req_id_to_index.get(request_id)
        if idx is not None:
            self.block_tables.num_blocks[idx] = 0
        self.request_state.remove_request(request_id)