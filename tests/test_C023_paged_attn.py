"""C023: tests for PagedAttention ops and physical KV cache integration."""

import torch

from my_vllm.v1.attention.backends.ops.paged_attn import (
    gather_cached_kv_pytorch,
    reshape_and_cache_pytorch,
)
from my_vllm.v1.worker.gpu_model_runner import GPUModelRunner
from my_vllm.v1.core.sched.output import (
    CachedRequestData, NewRequestData, SchedulerOutput,
)


class TestReshapeAndCache:
    def test_write_single(self):
        k_cache = torch.zeros(8, 4, 16, 64, device="cpu")
        v_cache = torch.zeros(8, 4, 16, 64, device="cpu")
        k_new = torch.randn(1, 4, 64)
        v_new = torch.randn(1, 4, 64)
        slot = torch.tensor([3 * 16 + 5], dtype=torch.int64)
        reshape_and_cache_pytorch(k_new, v_new, k_cache, v_cache, slot)
        assert torch.allclose(k_cache[3, :, 5, :], k_new[0])
        assert torch.allclose(v_cache[3, :, 5, :], v_new[0])

    def test_write_multiple(self):
        k_cache = torch.zeros(8, 4, 16, 64, device="cpu")
        v_cache = torch.zeros(8, 4, 16, 64, device="cpu")
        k_new = torch.randn(3, 4, 64)
        v_new = torch.randn(3, 4, 64)
        slots = torch.tensor([0, 1, 2], dtype=torch.int64)
        reshape_and_cache_pytorch(k_new, v_new, k_cache, v_cache, slots)
        assert torch.allclose(k_cache[0, :, 0, :], k_new[0])
        assert torch.allclose(k_cache[0, :, 1, :], k_new[1])
        assert torch.allclose(k_cache[0, :, 2, :], k_new[2])

    def test_ignore_negative_slots(self):
        k_cache = torch.zeros(8, 4, 16, 64, device="cpu")
        v_cache = torch.zeros(8, 4, 16, 64, device="cpu")
        k_new = torch.randn(2, 4, 64)
        v_new = torch.randn(2, 4, 64)
        slots = torch.tensor([-1, 0], dtype=torch.int64)
        reshape_and_cache_pytorch(k_new, v_new, k_cache, v_cache, slots)
        assert torch.allclose(k_cache[0, :, 0, :], k_new[1])
        assert (k_cache[0, :, 1, :] == 0).all()


class TestGatherCachedKV:
    def test_gather_single_block(self):
        k_cache = torch.zeros(8, 4, 16, 64)
        v_cache = torch.zeros(8, 4, 16, 64)
        # write token at block 1, offset 3
        k_cache[1, :, 3, :] = 1.0
        v_cache[1, :, 3, :] = 2.0

        block_table = torch.tensor([[1, 0, 0, 0]], dtype=torch.int32)
        seq_lens = torch.tensor([4], dtype=torch.int32)  # 4 tokens
        keys, values = gather_cached_kv_pytorch(
            k_cache, v_cache, block_table, seq_lens, 16,
        )
        assert keys.shape == (1, 4, 4, 64)
        assert values.shape == (1, 4, 4, 64)
        assert (keys[0, :, 3, :] == 1.0).all()
        assert (values[0, :, 3, :] == 2.0).all()
        # padding positions should be zero
        assert (keys[0, :, 0, :] == 0.0).all()

    def test_gather_multi_block(self):
        k_cache = torch.zeros(8, 4, 16, 64)
        v_cache = torch.zeros(8, 4, 16, 64)
        k_cache[1, :, 10, :] = 5.0
        k_cache[3, :, 0, :] = 6.0

        block_table = torch.tensor([[1, 3, 0, 0]], dtype=torch.int32)
        seq_lens = torch.tensor([17], dtype=torch.int32)
        keys, _ = gather_cached_kv_pytorch(
            k_cache, v_cache, block_table, seq_lens, 16,
        )
        assert keys.shape == (1, 4, 17, 64)
        assert (keys[0, :, 10, :] == 5.0).all()  # block1, off 10
        assert (keys[0, :, 16, :] == 6.0).all()  # block3, off 0=pos16


class TestPagedOpsRoundTrip:
    def test_write_then_gather(self):
        k_cache = torch.zeros(8, 4, 16, 64)
        v_cache = torch.zeros(8, 4, 16, 64)
        k_tok = torch.randn(5, 4, 64)
        v_tok = torch.randn(5, 4, 64)
        slots = torch.tensor(
            [2 * 16 + i for i in range(5)], dtype=torch.int64,
        )
        reshape_and_cache_pytorch(k_tok, v_tok, k_cache, v_cache, slots)

        block_table = torch.tensor([[2, 0, 0, 0]], dtype=torch.int32)
        seq_lens = torch.tensor([5], dtype=torch.int32)
        keys, values = gather_cached_kv_pytorch(
            k_cache, v_cache, block_table, seq_lens, 16,
        )
        # keys: [1, num_heads=4, seq=5, dim=64], k_tok: [5, 4, 64]
        assert torch.allclose(keys[0, :, :5, :], k_tok.permute(1, 0, 2))
        assert torch.allclose(values[0, :, :5, :], v_tok.permute(1, 0, 2))


MODEL = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m"


def _get_input_ids(prompt_text):
    from my_vllm.entrypoints.llm import load_tokenizer
    tok = load_tokenizer(MODEL)
    result = tok.encode(prompt_text)
    return list(result) if not isinstance(result, list) else result


def _load_model():
    from my_vllm.model_executor.model_loader import load_model
    model = load_model(MODEL)
    device = next(model.parameters()).device
    return model, device


class TestPhysicalKVCachePrefill:
    def test_single_request_prefill(self):
        model, device = _load_model()
        runner = GPUModelRunner(
            model, device, block_size=16,
            max_num_blocks=64, max_model_len=128,
        )
        prompt_ids = _get_input_ids("Hello world")
        so = SchedulerOutput(
            scheduled_new_reqs=[
                NewRequestData(
                    req_id="r1",
                    prompt_token_ids=prompt_ids,
                    block_ids=list(range(8)),
                ),
            ],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            total_num_scheduled_tokens=len(prompt_ids),
        )
        result = runner.execute_model(so)
        assert "r1" in result.sampled_token_ids
        assert len(result.sampled_token_ids["r1"]) == 1
        assert isinstance(result.sampled_token_ids["r1"][0], int)
        # KV cache should be non-zero after prefill
        assert (runner.kv_cache_k[0, 0] != 0).any()

    def test_prefill_no_requests(self):
        model, device = _load_model()
        runner = GPUModelRunner(model, device)
        so = SchedulerOutput.make_empty()
        result = runner.execute_model(so)
        assert result.sampled_token_ids == {}


class TestPhysicalKVCacheDecode:
    def test_decode_after_prefill(self):
        model, device = _load_model()
        runner = GPUModelRunner(
            model, device, block_size=16,
            max_num_blocks=64, max_model_len=128,
        )
        prompt_ids = _get_input_ids("The")
        so = SchedulerOutput(
            scheduled_new_reqs=[
                NewRequestData(
                    req_id="d1", prompt_token_ids=prompt_ids,
                    block_ids=list(range(8)),
                ),
            ],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            total_num_scheduled_tokens=len(prompt_ids),
        )
        runner.execute_model(so)

        # Decode one step
        so2 = SchedulerOutput(
            scheduled_cached_reqs=CachedRequestData(req_ids=["d1"]),
            total_num_scheduled_tokens=1,
        )
        result = runner.execute_model(so2)
        assert "d1" in result.sampled_token_ids
        tok = result.sampled_token_ids["d1"][0]
        assert isinstance(tok, int)

    def test_multiple_decode_steps(self):
        model, device = _load_model()
        runner = GPUModelRunner(
            model, device, block_size=16,
            max_num_blocks=64, max_model_len=128,
        )
        prompt_ids = _get_input_ids("Hello")
        so = SchedulerOutput(
            scheduled_new_reqs=[
                NewRequestData(
                    req_id="d1", prompt_token_ids=prompt_ids,
                    block_ids=list(range(8)),
                ),
            ],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            total_num_scheduled_tokens=len(prompt_ids),
        )
        runner.execute_model(so)

        tokens = []
        for _ in range(3):
            so2 = SchedulerOutput(
                scheduled_cached_reqs=CachedRequestData(req_ids=["d1"]),
                total_num_scheduled_tokens=1,
            )
            result = runner.execute_model(so2)
            tok = result.sampled_token_ids.get("d1", [None])[0]
            if tok is None:
                break
            tokens.append(tok)
        assert len(tokens) >= 1

    def test_cleanup(self):
        model, device = _load_model()
        runner = GPUModelRunner(
            model, device, block_size=16,
            max_num_blocks=64, max_model_len=128,
        )
        prompt_ids = _get_input_ids("Hi")
        so = SchedulerOutput(
            scheduled_new_reqs=[
                NewRequestData(
                    req_id="c1", prompt_token_ids=prompt_ids,
                    block_ids=list(range(4)),
                ),
            ],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            total_num_scheduled_tokens=len(prompt_ids),
        )
        runner.execute_model(so)
        runner.cleanup("c1")
        assert "c1" not in runner.request_state.req_id_to_index
        assert runner.get_step("c1") == 0

    def test_new_block_ids_in_decode(self):
        model, device = _load_model()
        runner = GPUModelRunner(
            model, device, block_size=16,
            max_num_blocks=64, max_model_len=128,
        )
        prompt_ids = _get_input_ids("A")
        so = SchedulerOutput(
            scheduled_new_reqs=[
                NewRequestData(
                    req_id="b1", prompt_token_ids=prompt_ids,
                    block_ids=list(range(8)),
                ),
            ],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            total_num_scheduled_tokens=len(prompt_ids),
        )
        runner.execute_model(so)

        so2 = SchedulerOutput(
            scheduled_cached_reqs=CachedRequestData(
                req_ids=["b1"],
                new_block_ids={"b1": [10]},
            ),
            total_num_scheduled_tokens=1,
        )
        runner.execute_model(so2)
        idx = runner.request_state.get_index("b1")
        assert runner.block_tables.block_table[idx, 0].item() == 0
