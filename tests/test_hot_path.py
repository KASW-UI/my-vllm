"""C120: end-to-end integration test for the V1 hot path.

Exercises:
  - GPUModelRunner.execute_model with the unified pipeline
    (update_states → prepare_inputs → prepare_attn → set_forward_context
     → model.forward → sample → postprocess).
  - TritonAttentionBackend as the default backend (post-C117).
  - Decode-only loop via EngineCore.step().
  - Both enforce_eager=True and enforce_eager=False paths produce
    the same token sequence (CUDA graph replay is byte-identical to
    eager on a fixed seed).

Replaces the prior "naive for model.generate in N times" pattern with
a real vLLM V1 loop: EngineCore.step() drives Scheduler.schedule →
Executor.execute_model → Scheduler.update_from_output → next step.
"""

import os

import pytest
import torch

from my_vllm.config.vllm import VllmConfig
from my_vllm.config.cache import CacheConfig
from my_vllm.config.model import ModelConfig
from my_vllm.config.scheduler import SchedulerConfig
from my_vllm.entrypoints.llm import LLM, load_tokenizer
from my_vllm.v1.attention.selector import get_attn_backend
from my_vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

MODEL_PATH = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m"
PROMPTS = ["Hello, my name is", "The meaning of life is"]


def _expected_greedy_tokens(tokenizer, prompts, max_new_tokens):
    """Reference: run the LLMEngine in eager mode and collect sampled tokens.

    Greedy decoding is deterministic — same model, same weights, same KV
    cache layout, same Triton kernels — so eager and cudagraph modes
    produce the same tokens for a fixed prompt and max_new_tokens.
    """
    llm = LLM(model=MODEL_PATH)
    results = []
    for prompt in prompts:
        out = llm.generate(prompt, max_new_tokens=max_new_tokens)[0]
        ids = tokenizer.encode(out["output_text"], add_special_tokens=False)
        results.append((out["output_text"], out["output_tokens"], ids))
    return results


def _patch_enforce_eager(value: bool):
    """Override ModelConfig.enforce_eager default."""
    import my_vllm.config.model as mm
    import my_vllm.config.vllm as vv

    orig = mm.ModelConfig
    class Patched(orig):
        enforce_eager: bool = value
    mm.ModelConfig = Patched
    vv.ModelConfig = Patched
    return orig


class TestHotPath:
    """End-to-end V1 pipeline tests."""

    def test_default_backend_is_triton(self):
        """C117: the selector returns TritonAttentionBackend by default.

        FlashAttentionBackend is preserved as a fallback only.
        """
        backend = get_attn_backend()
        assert backend is TritonAttentionBackend
        assert backend.get_name() == "TRITON_ATTN"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="requires CUDA"
    )
    def test_eager_and_cudagraph_produce_same_tokens(self):
        """enforce_eager=True and =False must yield identical token streams.

        Greedy decoding on a deterministic model + fixed seed should be
        reproducible; CUDA-graph replay is byte-identical to the eager
        forward under the same Triton kernel.
        """
        tokenizer = load_tokenizer(MODEL_PATH)
        max_new_tokens = 12

        orig = _patch_enforce_eager(True)
        try:
            eager_results = _expected_greedy_tokens(
                tokenizer, PROMPTS, max_new_tokens,
            )
        finally:
            import my_vllm.config.model as mm
            import my_vllm.config.vllm as vv
            mm.ModelConfig = orig
            vv.ModelConfig = orig

        orig = _patch_enforce_eager(False)
        try:
            graph_results = _expected_greedy_tokens(
                tokenizer, PROMPTS, max_new_tokens,
            )
        finally:
            import my_vllm.config.model as mm
            import my_vllm.config.vllm as vv
            mm.ModelConfig = orig
            vv.ModelConfig = orig

        # Same text, same token count, same token ids.
        assert len(eager_results) == len(graph_results) == len(PROMPTS)
        for (eager_text, eager_n, eager_ids), (graph_text, graph_n, graph_ids) in zip(
            eager_results, graph_results
        ):
            assert eager_text == graph_text, (
                f"eager vs cudagraph text mismatch:\n"
                f"  eager:  {eager_text!r}\n"
                f"  graph:  {graph_text!r}"
            )
            assert eager_n == graph_n
            assert eager_ids == graph_ids

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="requires CUDA"
    )
    def test_engine_core_step_pipeline(self):
        """EngineCore.step drives the V1 loop end-to-end.

        Mirrors vllm EngineCore.step():
            scheduler.schedule() → executor.execute_model() →
            scheduler.update_from_output() → EngineCoreOutputs.
        """
        from my_vllm.v1.engine.core import EngineCore
        from my_vllm.v1.executor.uniproc_executor import UniProcExecutor

        vllm_config = VllmConfig(
            model_config=ModelConfig(model_name=MODEL_PATH),
            cache_config=CacheConfig(block_size=16, num_kv_blocks=512),
            scheduler_config=SchedulerConfig(max_num_seqs=8),
        )
        core = EngineCore(
            vllm_config=vllm_config,
            executor_class=UniProcExecutor,
            log_stats=False,
        )

        tokenizer = load_tokenizer(MODEL_PATH)
        prompt = tokenizer.encode(PROMPTS[0])
        from my_vllm.v1.engine.input_processor import InputProcessor
        ip = InputProcessor(tokenizer, default_max_tokens=8)
        req = ip.process(PROMPTS[0], max_new_tokens=8)
        core.add_request(req)

        step = 0
        outputs_seen: list[str] = []
        while core.has_work() and step < 20:
            step += 1
            eco = core.step()
            for eo in eco.outputs:
                if eo.new_token_ids:
                    outputs_seen.append(str(eo.new_token_ids))

        # EngineCore processed at least one step and produced tokens.
        assert step >= 2, f"expected ≥2 steps, got {step}"
        assert len(outputs_seen) >= 1