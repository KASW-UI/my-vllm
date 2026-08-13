"""GPUModelRunner — batched model execution following vLLM V1.

Aligns with vllm/vllm/v1/worker/gpu/model_runner.py execute_model() pipeline:

  execute_model(scheduler_output)
    ├─ update_states(scheduler_output)
    ├─ prepare_inputs(scheduler_output)        → InputBatch
    ├─ prepare_attn(input_batch)               → CommonAttentionMetadata
    ├─ set_forward_context(attn_metadata)      (my_vllm.forward_context)
    ├─ model(input_ids, positions)             → hidden_states
    ├─ sample(hidden_states, input_batch)      → SamplerOutput
    └─ postprocess(sampled, input_batch)

Mirrors the V1 shape exactly: a single execute_model() with a unified
prepare_inputs() that handles both prefill (new reqs, prompt chunks) and
decode (cached reqs, last-token). No separate _execute_prefill /
_execute_decode split — that's the "naive" structure this commit replaces.

Engineering details skipped (principled scope reduction):
- CP / DCP: no Context-Parallel split.
- Spec-decode: no draft tokens / rejection sampling.
- Structured output bitmask: not consumed here (sampler is greedy-only).
- Multimodal inputs: model_state.get_mm_embeddings() not wired.
- KV-connector (NIXL / P/D): no receive/prefetch hooks in the hot path.
- TP/PP communication: not exercised at TP=1, PP=1.
- CUDA graph capture/replay (C119): not yet wired — eager mode only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from my_vllm.forward_context import set_forward_context
from my_vllm.v1.attention.metadata import CommonAttentionMetadata
from my_vllm.v1.core.sched.output import SchedulerOutput
from my_vllm.v1.outputs import ModelRunnerOutput, SamplerOutput
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


@dataclass
class InputBatch:
    """Unified per-batch view produced by _prepare_inputs().

    Mirrors vllm v1/worker/gpu/model_runner.py:InputBatch (subset — no
    spec-decode, no LoRA multiplexing, no DP). Holds the per-request
    flat tensors and per-request index mappings consumed by
    _prepare_attn / _sample / _postprocess.

    num_reqs:           number of scheduled requests in this step.
    req_ids:            [num_reqs] request ids in execution order.
    input_ids:          [num_tokens] flat input token ids.
    positions:          [num_tokens] flat per-token positions.
    query_start_loc:    [num_reqs + 1] cumulative token offsets (int32).
    seq_lens:           [num_reqs]    total seq len after this step (int32).
    num_computed_tokens:[num_reqs]    num tokens computed *before* this step.
    logits_indices:     [num_reqs]    indices into hidden_states for sampling.
                         For prefill, the last token of each request's chunk.
                         For decode, the only token.
    block_table:        [num_reqs, max_blocks] int32 (slice of self.block_tables.block_table).
    slot_mapping:       [num_tokens] int64 (slice of self.block_tables.slot_mappings).
    """

    num_reqs: int
    req_ids: list[str]
    input_ids: torch.Tensor
    positions: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    num_computed_tokens: torch.Tensor
    logits_indices: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor


class GPUModelRunner:
    """Mirrors vLLM V1 v1/worker/gpu/model_runner.py:GPUModelRunner (subset).

    Constructor: (vllm_config, device). Mirrors the Worker.init_device() contract
    (Worker constructs the runner and calls load_model()).
    """

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
        self.enforce_eager: bool = mc.enforce_eager

        self.model = None
        self.num_layers: int = 0
        self.num_kv_heads: int = 0
        self.head_dim: int = 0
        self.hidden_size: int = 0
        self.dtype: torch.dtype = _str_to_torch_dtype(mc.dtype)
        self.sampler = Sampler()

        self._step_ms: float = 0.0
        self._prefill_ms: float = 0.0
        self._decode_ms: float = 0.0

        self.request_state: RequestState | None = None
        self.block_tables: BlockTables | None = None
        self.kv_cache_k: torch.Tensor | None = None
        self.kv_cache_v: torch.Tensor | None = None
        # HND-packed KV cache (5D) — see load_model() for layout.
        self.kv_cache: torch.Tensor | None = None
        self.attn_metadata: CommonAttentionMetadata | None = None

        # CUDA graph capture state — populated by capture_cudagraph().
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.graph_bs: list[int] = []
        self.graph_pool: torch.cuda.CUDAGraph | None = None
        self.graph_vars: dict[str, torch.Tensor] | None = None

        self._max_num_reqs = sc.max_num_seqs
        self._max_num_batched_tokens = sc.max_num_batched_tokens
        self._max_model_len = mc.max_model_len

    # ── Lifecycle (mirrors vLLM Worker.init_device / load_model) ────────

    def load_model(self) -> None:
        from my_vllm.model_executor.model_loader import load_model

        model = load_model(self.model_name, dtype=self.dtype_str)
        self.model = model
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

        config = model.config
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size

        self.request_state = RequestState(
            max_num_reqs=self._max_num_reqs,
            max_model_len=self._max_model_len,
            device=self.device,
        )
        self.block_tables = BlockTables(
            block_size=self.block_size,
            max_num_reqs=self._max_num_reqs,
            max_num_blocks=self.max_num_blocks,
            max_num_batched_tokens=self._max_num_batched_tokens,
            device=self.device,
        )

        self.kv_cache_k = None
        self.kv_cache_v = None
        # HND-packed layout: (num_layers, num_blocks, num_kv_heads,
        #                     block_size, 2 * head_dim). K = [..., :head_dim],
        # V = [..., head_dim:]. Matches vllm TritonAttentionBackend.
        # get_kv_cache_shape and is what TritonAttentionImpl expects via
        # bind_kv_cache (single 5D tensor per layer).
        self.kv_cache = torch.zeros(
            self.num_layers, self.max_num_blocks, self.num_kv_heads,
            self.block_size, 2 * self.head_dim,
            dtype=self.dtype, device=self.device,
        )

        self._bind_kv_caches()

        if not self.enforce_eager:
            self.capture_cudagraph()

    def _bind_kv_caches(self) -> None:
        for i, layer in enumerate(self.model.model.decoder.layers):
            layer.self_attn.attn.bind_kv_cache(self.kv_cache[i])

    # ── CUDA graph capture (nano-vllm style; mirrors vllm cudagraph_utils) ─

    @torch.inference_mode()
    def capture_cudagraph(self) -> None:
        """Capture CUDA graphs for decode batches of varying sizes.

        Mirrors nano-vllm model_runner.capture_cudagraph and vllm
        v1/worker/gpu/cudagraph_utils.py:cudagraph_capture. Captures one
        graph per batch size in [1, 2, 4, 8, 16, 32, ..., max_bs], sharing
        a CUDA graph pool across sizes. Decode steps whose bs ≤ max_bs
        can replay the closest captured graph and copy inputs into the
        pinned graph_vars buffers.
        """
        max_bs = min(self._max_num_reqs, 512)
        max_num_blocks = (
            (self._max_model_len + self.block_size - 1) // self.block_size
        )

        # Pre-allocate pinned input buffers — addresses stay constant across
        # replays so the captured graph's device pointers remain valid.
        graph_input_ids = torch.zeros(max_bs, dtype=torch.int64, device=self.device)
        graph_positions = torch.zeros(max_bs, dtype=torch.int64, device=self.device)
        graph_slot_mapping = torch.zeros(max_bs, dtype=torch.int64, device=self.device)
        graph_seq_lens = torch.zeros(max_bs, dtype=torch.int32, device=self.device)
        graph_block_tables = torch.zeros(
            max_bs, max_num_blocks, dtype=torch.int32, device=self.device,
        )
        graph_query_start_loc = torch.zeros(
            max_bs + 1, dtype=torch.int32, device=self.device,
        )
        graph_outputs = torch.zeros(
            max_bs, self.hidden_size, dtype=self.dtype, device=self.device,
        )

        # graph_bs ladder: nano-vllm pattern.
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # Build a CommonAttentionMetadata template for decode at each bs.
        # causal=False (no causal mask for single-token decode queries),
        # positions=[0] * bs (the model treats positions as the seq-len
        # offset rather than relying on it for prefill masking).
        causal_template = False
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            # Populate graph_vars so the captured forward sees realistic data.
            graph_input_ids[:bs].fill_(0)
            graph_positions[:bs].fill_(0)
            graph_slot_mapping.fill_(-1)
            graph_seq_lens.fill_(1)
            graph_block_tables[:bs].fill_(0)
            graph_query_start_loc[: bs + 1].fill_(0)
            graph_query_start_loc[1 : bs + 1] = torch.arange(
                1, bs + 1, dtype=torch.int32, device=self.device,
            )

            attn_metadata = CommonAttentionMetadata(
                query_start_loc=graph_query_start_loc[: bs + 1],
                query_start_loc_cpu=graph_query_start_loc[: bs + 1].cpu(),
                seq_lens=graph_seq_lens[:bs],
                seq_lens_cpu=graph_seq_lens[:bs].cpu(),
                num_reqs=bs,
                num_actual_tokens=bs,
                max_query_len=1,
                max_seq_len=1,
                block_table_tensor=graph_block_tables[:bs],
                slot_mapping=graph_slot_mapping[:bs],
                positions=graph_positions[:bs],
                causal=causal_template,
            )
            attn_dict = {
                layer.self_attn.attn.layer_name: attn_metadata
                for layer in self.model.model.decoder.layers
            }

            # Warmup before capture (CUDA graphs need kernels to be JIT-cached).
            with set_forward_context(attn_dict):
                graph_outputs[:bs] = self.model(
                    graph_input_ids[:bs], graph_positions[:bs],
                )
            torch.cuda.synchronize()

            with set_forward_context(attn_dict):
                with torch.cuda.graph(graph, self.graph_pool):
                    graph_outputs[:bs] = self.model(
                        graph_input_ids[:bs], graph_positions[:bs],
                    )
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()

        self.graph_vars = {
            "input_ids": graph_input_ids,
            "positions": graph_positions,
            "slot_mapping": graph_slot_mapping,
            "seq_lens": graph_seq_lens,
            "block_tables": graph_block_tables,
            "query_start_loc": graph_query_start_loc,
            "outputs": graph_outputs,
        }

    def _next_graph_bs(self, bs: int) -> int:
        """Smallest captured bs >= bs (nano-vllm pattern)."""
        for s in self.graph_bs:
            if s >= bs:
                return s
        return self.graph_bs[-1]

    # ── execute_model: the V1 hot path ──────────────────────────────────

    @torch.inference_mode()
    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput:
        """Run one scheduling + execution step. Mirrors vLLM execute_model().

        Pipeline:
            update_states → prepare_inputs → prepare_attn →
            set_forward_context → model.forward → sample → postprocess.

        Decode-only steps (no new reqs, all bs <= max captured) replay
        a captured CUDA graph when enforce_eager is False.
        """
        if scheduler_output.total_num_scheduled_tokens == 0:
            return ModelRunnerOutput(
                req_ids=[], req_id_to_index={}, sampled_token_ids=[],
            )

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        self._update_states(scheduler_output)

        input_batch = self._prepare_inputs(scheduler_output)

        can_use_graph = (
            not self.enforce_eager
            and self.graph_vars is not None
            and not scheduler_output.scheduled_new_reqs
            and input_batch.num_reqs <= self._max_num_reqs
            and (input_batch.query_start_loc[1:] - input_batch.query_start_loc[:-1]).max().item() == 1
        )

        if can_use_graph:
            hidden_states = self._run_with_cudagraph(input_batch)
            attn_metadata = self._prepare_attn(input_batch)
        else:
            attn_metadata = self._prepare_attn(input_batch)
            attn_dict = self._build_attn_metadata_dict(attn_metadata)
            with set_forward_context(attn_dict):
                hidden_states = self.model(
                    input_batch.input_ids, input_batch.positions,
                )

        sampled = self._sample(hidden_states, input_batch)
        self._postprocess(sampled, input_batch)

        end_event.record()
        torch.cuda.synchronize()
        self._step_ms = start_event.elapsed_time(end_event)

        return self._build_model_runner_output(sampled, input_batch)

    def _run_with_cudagraph(
        self, input_batch: InputBatch,
    ) -> torch.Tensor:
        """Replay the closest captured CUDA graph for a decode batch.

        Mirrors nano-vllm run_model() — copies live input_batch tensors
        into pinned graph_vars buffers, replays the captured graph, and
        returns the per-request hidden_states slice.
        """
        bs = input_batch.num_reqs
        graph_bs = self._next_graph_bs(bs)
        graph = self.graphs[graph_bs]
        gv = self.graph_vars

        gv["input_ids"][:bs] = input_batch.input_ids
        gv["positions"][:bs] = input_batch.positions
        gv["slot_mapping"].fill_(-1)
        gv["slot_mapping"][:bs] = input_batch.slot_mapping
        gv["seq_lens"].fill_(0)
        gv["seq_lens"][:bs] = input_batch.seq_lens
        bt_live = input_batch.block_table
        bt_width = bt_live.shape[1]
        gv["block_tables"][:bs, :bt_width] = bt_live
        # qsl: row i has start=i, end=i+1 for decode (single token per req).
        gv["query_start_loc"][: bs + 1].fill_(0)
        gv["query_start_loc"][1 : bs + 1] = torch.arange(
            1, bs + 1, dtype=torch.int32, device=self.device,
        )

        graph.replay()
        return gv["outputs"][:bs]

    # ── State update (mirrors vLLM add_requests / update_requests) ──────

    def _update_states(self, scheduler_output: SchedulerOutput) -> None:
        """Register new requests and append any new block ids for cached reqs.

        Mirrors vLLM's add_requests() (for new reqs) and update_requests()
        (for cached reqs' new_block_ids).
        """
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

    # ── prepare_inputs (mirrors vLLM prepare_inputs) ────────────────────

    def _prepare_inputs(self, scheduler_output: SchedulerOutput) -> InputBatch:
        """Build flat input tensors from new + cached reqs.

        Mirrors vllm prepare_inputs(). Both new (prefill, possibly chunked) and
        cached (decode, single last-token) reqs share the same flat buffers:
        - input_ids[i]: token for the i-th scheduled token (prompt chunk or
          last sampled token).
        - positions[i]:  absolute position of that token in the seq.
        - query_start_loc[r]: cumulative sum of num_scheduled_tokens.
        - seq_lens[r]:   total seq len *after* this step's tokens.
        - logits_indices[r]: index of the per-request token whose hidden state
          gets passed to the sampler (last token of prefill chunk, or the only
          decode token).
        """
        new_datas = scheduler_output.scheduled_new_reqs
        cached = scheduler_output.scheduled_cached_reqs
        cached_reqs = cached.req_ids
        nct_per_cached = cached.num_computed_tokens

        input_ids_list: list[torch.Tensor] = []
        positions_list: list[torch.Tensor] = []
        query_start_loc_list: list[int] = [0]
        seq_lens_list: list[int] = []
        nct_list: list[int] = []
        req_ids: list[str] = []
        logits_token_idx: list[int] = []

        # New reqs (prefill, possibly chunked).
        for d in new_datas:
            nct_before = d.num_computed_tokens
            scheduled = scheduler_output.num_scheduled_tokens.get(d.req_id)
            if scheduled is None:
                scheduled = len(d.prompt_token_ids) - nct_before
            chunk = d.prompt_token_ids[nct_before:nct_before + scheduled]
            nct_after = nct_before + len(chunk)
            positions = list(range(nct_before, nct_after))

            input_ids_list.append(
                torch.tensor(chunk, dtype=torch.long, device=self.device)
            )
            positions_list.append(
                torch.tensor(positions, dtype=torch.long, device=self.device)
            )
            query_start_loc_list.append(
                query_start_loc_list[-1] + len(chunk)
            )
            seq_lens_list.append(nct_after)
            nct_list.append(nct_before)
            req_ids.append(d.req_id)
            logits_token_idx.append(query_start_loc_list[-1] - 1)

        # Cached reqs (decode — single token).
        for i, rid in enumerate(cached_reqs):
            nct_before = nct_per_cached[i] if i < len(nct_per_cached) else 0
            idx = self.request_state.get_index(rid)
            last_token = self.request_state.get_last_token(rid)
            input_ids_list.append(
                torch.tensor([last_token], dtype=torch.long, device=self.device)
            )
            positions_list.append(
                torch.tensor([nct_before], dtype=torch.long, device=self.device)
            )
            query_start_loc_list.append(query_start_loc_list[-1] + 1)
            seq_lens_list.append(nct_before + 1)
            nct_list.append(nct_before)
            req_ids.append(rid)
            logits_token_idx.append(query_start_loc_list[-1] - 1)

        num_reqs = len(req_ids)
        num_tokens = query_start_loc_list[-1]

        input_ids = (
            torch.cat(input_ids_list)
            if input_ids_list
            else torch.empty(0, dtype=torch.long, device=self.device)
        )
        positions = (
            torch.cat(positions_list)
            if positions_list
            else torch.empty(0, dtype=torch.long, device=self.device)
        )
        query_start_loc = torch.tensor(
            query_start_loc_list, dtype=torch.int32, device=self.device,
        )
        seq_lens = torch.tensor(
            seq_lens_list, dtype=torch.int32, device=self.device,
        )
        num_computed_tokens = torch.tensor(
            nct_list, dtype=torch.int32, device=self.device,
        )
        logits_indices = torch.tensor(
            logits_token_idx, dtype=torch.int64, device=self.device,
        )

        # Block tables + slot mappings.
        self.block_tables.compute_slot_mappings(
            query_start_loc, positions, num_tokens,
        )
        block_table = self.block_tables.block_table[:num_reqs]
        slot_mapping = self.block_tables.slot_mappings[:num_tokens]

        return InputBatch(
            num_reqs=num_reqs,
            req_ids=req_ids,
            input_ids=input_ids,
            positions=positions,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            num_computed_tokens=num_computed_tokens,
            logits_indices=logits_indices,
            block_table=block_table,
            slot_mapping=slot_mapping,
        )

    # ── prepare_attn (mirrors vLLM prepare_attn) ────────────────────────

    def _prepare_attn(self, input_batch: InputBatch) -> CommonAttentionMetadata:
        """Build the CommonAttentionMetadata for the current step.

        Mirrors vllm model_state.prepare_attn(). Single KV cache group, no
        CP/DCP, no prefix-cache (block_table carries full path).
        """
        query_start_loc = input_batch.query_start_loc
        seq_lens = input_batch.seq_lens
        max_query_len = int((query_start_loc[1:] - query_start_loc[:-1]).max().item())
        max_seq_len = int(seq_lens.max().item()) if seq_lens.numel() > 0 else 1

        # Cached reqs are decode (1-token queries); new reqs may have chunked
        # prefill queries. causal=True only when there is any prefill chunk —
        # the decoder's causal mask is consumed via the impl; we pass causal
        # based on the uniform-token heuristic vllm uses.
        is_all_decode = all(
            int((query_start_loc[i + 1] - query_start_loc[i]).item()) == 1
            for i in range(input_batch.num_reqs)
        )
        causal = not is_all_decode

        metadata = CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc.cpu(),
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.cpu(),
            num_reqs=input_batch.num_reqs,
            num_actual_tokens=int(input_batch.input_ids.shape[0]),
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            block_table_tensor=input_batch.block_table,
            slot_mapping=input_batch.slot_mapping,
            positions=input_batch.positions,
            causal=causal,
        )
        self.attn_metadata = metadata
        return metadata

    def _build_attn_metadata_dict(
        self, attn_metadata: CommonAttentionMetadata
    ) -> dict[str, CommonAttentionMetadata]:
        """Map layer_name → attn_metadata for set_forward_context().

        Mirrors vllm: every Attention layer reads its metadata by name.
        """
        return {
            layer.self_attn.attn.layer_name: attn_metadata
            for layer in self.model.model.decoder.layers
        }

    # ── sample (mirrors vLLM sample) ────────────────────────────────────

    def _sample(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
    ) -> SamplerOutput:
        """Pick the next token for each request from its last-token hidden state.

        Mirrors vllm model_runner.sample(). hidden_states[input_batch.logits_indices]
        selects the per-request token whose hidden state feeds the LM head.
        """
        sample_hidden = hidden_states[input_batch.logits_indices]
        logits = self.model.compute_logits(sample_hidden)
        sampled = self.sampler(logits, SamplingMetadata(all_greedy=True))
        return SamplerOutput(
            sampled_token_ids=[int(t.item()) for t in sampled],
        )

    # ── postprocess (mirrors vLLM postprocess_sampled) ──────────────────

    def _postprocess(
        self,
        sampled: SamplerOutput,
        input_batch: InputBatch,
    ) -> None:
        """Update per-request state with sampled tokens.

        Mirrors vllm postprocess_sampled(). Increments num_computed_tokens,
        records last_sampled_tokens, marks EOS.
        """
        for rid, tok in zip(input_batch.req_ids, sampled.sampled_token_ids):
            if tok == self._EOS_TOKEN_ID:
                continue
            self.request_state.set_last_sampled_token(rid, tok)
            self.request_state.append_output_token(rid, tok)

    # ── output assembly ─────────────────────────────────────────────────

    def _build_model_runner_output(
        self,
        sampled: SamplerOutput,
        input_batch: InputBatch,
    ) -> ModelRunnerOutput:
        req_ids = list(input_batch.req_ids)
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=[[tok] for tok in sampled.sampled_token_ids],
        )

    # ── housekeeping ────────────────────────────────────────────────────

    def get_step(self, request_id: str) -> int:
        return self.request_state.get_step(request_id)

    def cleanup(self, request_id: str):
        idx = self.request_state.req_id_to_index.get(request_id)
        if idx is not None:
            self.block_tables.num_blocks[idx] = 0
        self.request_state.remove_request(request_id)

    @property
    def _empty_output(self) -> ModelRunnerOutput:
        return ModelRunnerOutput(req_ids=[], req_id_to_index={}, sampled_token_ids=[])