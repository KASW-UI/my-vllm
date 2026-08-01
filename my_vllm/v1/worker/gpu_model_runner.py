"""GPUModelRunner — batched model execution.

Mirrors vLLM V1 v1/worker/gpu_model_runner.py core structure.

C017: replaces per-request prefill/decode with batched prefill
and centralized decode.
C019: adds BlockTables GPU tensor management.
C020: RequestState replaces Python dicts for per-request state.
C022: builds CommonAttentionMetadata from InputBatch + BlockTables.
C023: physical KV cache pool replaces HF past_key_values.
C024: Sampler (nn.Module) replaces greedy_sample() function.
"""

import torch
import torch.nn as nn

from my_vllm.v1.attention.metadata import CommonAttentionMetadata
from my_vllm.v1.attention.selector import get_attn_backend
from my_vllm.v1.attention.backends.ops.paged_attn import (
    gather_cached_kv_pytorch,
    reshape_and_cache_pytorch,
)
from my_vllm.v1.core.sched.output import SchedulerOutput
from my_vllm.v1.outputs import ModelRunnerOutput
from my_vllm.v1.sample.metadata import SamplingMetadata
from my_vllm.v1.sample.sampler import Sampler
from my_vllm.v1.worker.gpu.block_table import BlockTables
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
        self.max_num_blocks = max_num_blocks

        config = model.config
        num_layers = config.num_hidden_layers
        num_kv_heads = getattr(
            config, "num_key_value_heads", config.num_attention_heads
        )
        head_dim = config.hidden_size // config.num_attention_heads
        dtype = next(model.parameters()).dtype

        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # C024: Sampler (nn.Module) — replaces greedy_sample()
        self.sampler = Sampler()

        # C023: physical KV cache pool
        self.kv_cache_k = torch.zeros(
            num_layers, max_num_blocks, num_kv_heads, block_size, head_dim,
            dtype=dtype, device=device,
        )
        self.kv_cache_v = torch.zeros(
            num_layers, max_num_blocks, num_kv_heads, block_size, head_dim,
            dtype=dtype, device=device,
        )

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
        sampled_tokens: dict[str, list[int]] = {}

        prefill_datas = []
        for d in scheduler_output.scheduled_new_reqs:
            if d.req_id not in self.request_state.req_id_to_index:
                self.request_state.add_request(d.req_id, d.prompt_token_ids)
            idx = self.request_state.get_index(d.req_id)
            self.block_tables.append_block_ids(idx, d.block_ids)
            prefill_datas.append(d)

        for rid, new_bids in scheduler_output.scheduled_cached_reqs.new_block_ids.items():
            idx = self.request_state.get_index(rid)
            self.block_tables.append_block_ids(idx, new_bids)

        if prefill_datas:
            prefill_results = self._prefill_batch(prefill_datas)
            for rid, token in prefill_results.items():
                self.request_state.set_last_sampled_token(rid, token)
                sampled_tokens.setdefault(rid, []).append(token)

        decode_req_ids = scheduler_output.scheduled_cached_reqs.req_ids
        if decode_req_ids:
            decode_results = self._decode_batch(decode_req_ids)
            for rid, token in decode_results.items():
                if token is not None:
                    self.request_state.append_output_token(rid, token)
                    sampled_tokens.setdefault(rid, []).append(token)

        return ModelRunnerOutput(
            req_ids=list(sampled_tokens.keys()),
            sampled_token_ids=sampled_tokens,
        )

    def _prefill_batch(self, new_datas: list) -> dict[str, int]:
        results: dict[str, int] = {}
        if not new_datas:
            return results

        # C030: support chunked prefill — compute effective (uncached) prompt
        # length per request based on num_computed_tokens
        effective_lens = []
        for d in new_datas:
            start = d.num_computed_tokens
            effective_lens.append(len(d.prompt_token_ids) - start)

        max_len = max(effective_lens)
        batch_size = len(new_datas)

        input_ids = torch.zeros(
            batch_size, max_len, dtype=torch.long, device=self.device,
        )
        attention_mask = torch.zeros(
            batch_size, max_len, dtype=torch.long, device=self.device,
        )

        req_ids = []
        for i, d in enumerate(new_datas):
            start = d.num_computed_tokens
            chunk = d.prompt_token_ids[start:]
            clen = len(chunk)
            input_ids[i, :clen] = torch.tensor(
                chunk, dtype=torch.long, device=self.device,
            )
            attention_mask[i, :clen] = 1
            req_ids.append(d.req_id)

        positions_tensor = torch.zeros(
            batch_size, max_len, dtype=torch.long, device=self.device,
        )
        for i, d in enumerate(new_datas):
            start = d.num_computed_tokens
            clen = effective_lens[i]
            positions_tensor[i, :clen] = torch.arange(
                start, start + clen, dtype=torch.long, device=self.device,
            )
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

        self._write_kv_to_physical_cache(past_kvs, new_datas, batch_size, effective_lens)

        # C024: use Sampler — logits at the last effective token position
        token_logits = torch.stack(
            [logits[i, effective_lens[i] - 1, :]
             for i in range(batch_size)]
        )  # [batch_size, vocab_size]

        sampling_meta = SamplingMetadata(all_greedy=True)
        sampled = self.sampler(token_logits, sampling_meta)

        for i, rid in enumerate(req_ids):
            results[rid] = int(sampled[i].item())

        return results

    def _write_kv_to_physical_cache(
        self, past_key_values, new_datas: list, batch_size: int,
        effective_lens: list[int] = None,
    ) -> None:
        """Write HF past_key_values into physical GPU KV cache via slot_mapping.
        """
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
                bsz * padded_len, nheads, hdim,
            )
            v_flat = layer_v.permute(0, 2, 1, 3).reshape(
                bsz * padded_len, nheads, hdim,
            )
            reshape_and_cache_pytorch(
                k_flat[valid], v_flat[valid],
                self.kv_cache_k[layer_idx],
                self.kv_cache_v[layer_idx],
                slot_map[valid],
            )

    def _decode_batch(self, req_ids: list[str]) -> dict[str, int | None]:
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
                self.request_state.get_index(r)
            ]) for r in valid_reqs],
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
            [[t] for t in valid_tokens], dtype=torch.long, device=self.device,
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=next_inputs,
                past_key_values=past_key_values,
                use_cache=True,
            )

        logits_all = outputs.logits  # [batch, 1, vocab]
        new_past_kvs = outputs.past_key_values

        # C024: batch sample using Sampler
        logits_flat = logits_all[:, -1, :]  # [num_reqs, vocab_size]
        sampling_meta = SamplingMetadata(all_greedy=True)
        sampled = self.sampler(logits_flat, sampling_meta)

        # Write new decode K/V to physical cache via slot mapping
        decode_slots = []
        for i, rid in enumerate(valid_reqs):
            idx = self.request_state.get_index(rid)
            old_seq_len = int(self.request_state.num_computed_tokens_cpu[idx])
            block_index = old_seq_len // self.block_size
            block_offset = old_seq_len % self.block_size
            block_id = int(
                self.block_tables.block_table[idx, block_index].item()
            )
            decode_slots.append(block_id * self.block_size + block_offset)

        slot_tensor = torch.tensor(decode_slots, dtype=torch.int64, device=self.device)
        for layer_idx, (layer_k, layer_v) in enumerate(new_past_kvs):
            k_last = layer_k[:, :, -1, :]  # [batch, nheads, hdim]
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

            if next_token == self.eos_token_id:
                results[rid] = None
            else:
                results[rid] = next_token

        return results

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
