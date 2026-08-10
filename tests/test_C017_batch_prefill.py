"""C017: tests for GPUModelRunner batch prefill."""

import torch

from my_vllm.runtime.tokenizer import load_tokenizer
from my_vllm.model_executor.model_loader import load_model
from my_vllm.v1.worker.gpu_model_runner import GPUModelRunner
from my_vllm.v1.core.sched.output import (
    SchedulerOutput, NewRequestData, CachedRequestData,
)
from my_vllm.v1.outputs import ModelRunnerOutput

MODEL = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m"


class TestGPUModelRunner:
    def test_execute_model_single_prefill(self):
        tokenizer = load_tokenizer(MODEL)
        model = load_model(MODEL)
        runner = GPUModelRunner(model, model.device)
        prompt = tokenizer.encode("Hello world")

        o = SchedulerOutput(
            scheduled_new_reqs=[
                NewRequestData(req_id="r1", prompt_token_ids=prompt),
            ],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            total_num_scheduled_tokens=len(prompt),
        )
        result = runner.execute_model(o)
        assert isinstance(result, ModelRunnerOutput)
        assert "r1" in result.sampled_token_ids
        assert len(result.sampled_token_ids["r1"]) == 1
        assert isinstance(result.sampled_token_ids["r1"][0], int)

    def test_execute_model_batch_prefill(self):
        tokenizer = load_tokenizer(MODEL)
        model = load_model(MODEL)
        runner = GPUModelRunner(model, model.device)
        prompts = [
            tokenizer.encode("The meaning of"),
            tokenizer.encode("Artificial intelligence"),
            tokenizer.encode("Deep learning"),
        ]

        new_reqs = [
            NewRequestData(req_id=f"r{i}", prompt_token_ids=p)
            for i, p in enumerate(prompts)
        ]
        o = SchedulerOutput(
            scheduled_new_reqs=new_reqs,
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            total_num_scheduled_tokens=sum(len(p) for p in prompts),
        )
        result = runner.execute_model(o)
        assert isinstance(result, ModelRunnerOutput)
        assert len(result.sampled_token_ids) == 3
        for i in range(3):
            rid = f"r{i}"
            assert rid in result.sampled_token_ids
            assert len(result.sampled_token_ids[rid]) == 1

    def test_execute_model_empty(self):
        model = load_model(MODEL)
        runner = GPUModelRunner(model, model.device)
        o = SchedulerOutput.make_empty()
        result = runner.execute_model(o)
        assert result.sampled_token_ids == {}

    def test_get_step(self):
        model = load_model(MODEL)
        runner = GPUModelRunner(model, model.device)
        assert runner.get_step("nonexistent") == 0

    def test_cleanup(self):
        model = load_model(MODEL)
        runner = GPUModelRunner(model, model.device)
        # cleanup should not raise
        runner.cleanup("nonexistent")
        runner.cleanup("nonexistent")
