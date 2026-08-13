"""GPUModelRunner — batched model execution with vLLM-native model.

Aligns with vLLM V1 v1/worker/gpu_model_runner.py core structure.
Uses vLLM-native OPTForCausalLM (model(input_ids, positions)) with
attention backend (paged KV cache read/write inside FlashAttentionImpl).

Key changes from C022-C024 era:
- model(input_ids, positions) replaces HF model(input_ids, attention_mask)
- paged KV cache is read/written inside Attention.forward() via FlashAttentionImpl
- ForwardContext carries attn_metadata to attention layers
- compute_logits() replaces output.logits
- No past_key_values / use_cache / attention_mask from HF
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from my_vllm.forward_context import set_forward_context
from my_vllm.v1.attention.metadata import CommonAttentionMetadata
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
    """Batched model executor — aligns with vLLM V1 GPUModelRunner."""

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
        self.attn_metadata: CommonAttentionMetadata | None = None

        self._step_ms: float = 0.0
        self._prefill_ms: float = 0.0
        self._decode_ms: float = 0.0

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
        self.dtype = next(model.parameters()).dtype

        config = model.config
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

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

        self._bind_kv_caches()

    def _bind_kv_caches(self) -> None:
        for i, layer in enumerate(self.model.model.decoder.layers):
            k = self.kv_cache_k[i]
            v = self.kv_cache_v[i]
            layer.self_attn.attn.bind_kv_cache((k, v))

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        self._update_states(scheduler_output)

        sampled_tokens: dict[str, list[int]] = {}

        new_reqs = scheduler_output.scheduled_new_reqs
        cached = scheduler_output.scheduled_cached_reqs

        if new_reqs:
            prefill_event = torch.cuda.Event(enable_timing=True)
            prefill_event.record()
            prefill_results = self._execute_prefill(new_reqs)
            for rid, token in prefill_results.items():
                self.request_state.set_last_sampled_token(rid, token)
                sampled_tokens.setdefault(rid, []).append(token)
            self._prefill_ms = start_event.elapsed_time(prefill_event)

        decode_req_ids = cached.req_ids
        if decode_req_ids:
            decode_event = torch.cuda.Event(enable_timing=True)
            decode_event.record()
            decode_results = self._execute_decode(decode_req_ids)
            for rid, token in decode_results.items():
                if token is not None:
                    self.request_state.append_output_token(rid, token)
                    sampled_tokens.setdefault(rid, []).append(token)
            self._decode_ms = (
                prefill_event.elapsed_time(decode_event) if new_reqs
                else start_event.elapsed_time(decode_event)
            )

        end_event.record()
        torch.cuda.synchronize()
        self._step_ms = start_event.elapsed_time(end_event)

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
            if i < len(cached.new_block_ids):
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
        positions_tensor = torch.zeros(
            batch_size, max_len, dtype=torch.long, device=self.device)

        req_ids = []
        for i, d in enumerate(new_datas):
            start = d.num_computed_tokens
            chunk = d.prompt_token_ids[start:]
            clen = len(chunk)
            input_ids[i, :clen] = torch.tensor(
                chunk, dtype=torch.long, device=self.device)
            positions_tensor[i, :clen] = torch.arange(
                start, start + clen, dtype=torch.long, device=self.device)
            req_ids.append(d.req_id)

        query_start = torch.tensor(
            [0] + effective_lens,
            dtype=torch.long, device=self.device,
        ).cumsum(0)
        num_tokens = batch_size * max_len
        flat_pos = positions_tensor.flatten()[:num_tokens]

        self.block_tables.compute_slot_mappings(query_start, flat_pos, num_tokens)

        seq_lens = torch.tensor(effective_lens, dtype=torch.int32, device=self.device)
        self.attn_metadata = CommonAttentionMetadata(
            query_start_loc=query_start,
            query_start_loc_cpu=query_start.cpu(),
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.cpu(),
            num_reqs=batch_size,
            num_actual_tokens=num_tokens,
            max_query_len=max_len,
            max_seq_len=max_len,
            block_table_tensor=self.block_tables.block_table[:batch_size],
            slot_mapping=self.block_tables.slot_mappings[:num_tokens],
            positions=flat_pos,
            causal=True,
        )

        attn_metadata_dict = self._build_attn_metadata_dict()

        with set_forward_context(attn_metadata_dict):
            with torch.no_grad():
                hidden_states = self.model(input_ids, positions_tensor)

        logits = self.model.compute_logits(hidden_states)

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

        next_inputs = torch.tensor(
            [[t] for t in valid_tokens], dtype=torch.long, device=self.device)
        next_positions = seq_lens.unsqueeze(1).to(torch.long)

        query_start = torch.tensor(
            [0, num_reqs], dtype=torch.long, device=self.device,
        )

        decode_slot_mapping = torch.full(
            (num_reqs,), -1, dtype=torch.int64, device=self.device,
        )
        for i, rid in enumerate(valid_reqs):
            idx = self.request_state.get_index(rid)
            slen = int(seq_lens[i].item())
            block_index = slen // self.block_size
            block_offset = slen % self.block_size
            block_id = int(self.block_tables.block_table[idx, block_index].item())
            decode_slot_mapping[i] = block_id * self.block_size + block_offset

        self.attn_metadata = CommonAttentionMetadata(
            query_start_loc=query_start,
            query_start_loc_cpu=query_start.cpu(),
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.cpu(),
            num_reqs=num_reqs,
            num_actual_tokens=num_reqs,
            max_query_len=1,
            max_seq_len=int(seq_lens.max().item()) if num_reqs > 0 else 1,
            block_table_tensor=self.block_tables.block_table[:num_reqs],
            slot_mapping=decode_slot_mapping,
            positions=None,
            causal=False,
        )

        attn_metadata_dict = self._build_attn_metadata_dict()

        with set_forward_context(attn_metadata_dict):
            with torch.no_grad():
                hidden_states = self.model(next_inputs, next_positions)

        logits = self.model.compute_logits(hidden_states)
        logits_flat = logits[:, -1, :] if logits.dim() == 3 else logits
        sampled = self.sampler(logits_flat, SamplingMetadata(all_greedy=True))

        for i, rid in enumerate(valid_reqs):
            next_token = int(sampled[i].item())
            self.request_state.set_last_sampled_token(rid, next_token)
            if next_token == self._EOS_TOKEN_ID:
                results[rid] = None
            else:
                results[rid] = next_token

        return results

    def _build_attn_metadata_dict(self) -> dict[str, CommonAttentionMetadata]:
        result: dict[str, CommonAttentionMetadata] = {}
        for layer in self.model.model.decoder.layers:
            name = layer.self_attn.attn.layer_name
            result[name] = self.attn_metadata
        return result

    def get_step(self, request_id: str) -> int:
        return self.request_state.get_step(request_id)

    def cleanup(self, request_id: str):
        idx = self.request_state.req_id_to_index.get(request_id)
        if idx is not None:
            self.block_tables.num_blocks[idx] = 0
        self.request_state.remove_request(request_id)
