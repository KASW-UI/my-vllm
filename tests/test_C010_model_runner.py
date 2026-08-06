"""C010 / C032: tests for GPUModelRunner via V1 execute_model() API.

C032: legacy _prefill_one / _decode_one_legacy removed.
Rewritten to test the V1 batch execution path using SchedulerOutput.
"""

import torch

from my_vllm.runtime.tokenizer import load_tokenizer
from my_vllm.v1.model_executor.model_loader import load_model
from my_vllm.v1.worker.gpu_model_runner import GPUModelRunner
from my_vllm.v1.core.sched.output import (
    SchedulerOutput, NewRequestData, CachedRequestData,
)
from my_vllm.v1.outputs import ModelRunnerOutput

MODEL = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m"


def _get_input_ids(prompt_text):
    tok = load_tokenizer(MODEL)
    result = tok.encode(prompt_text)
    return list(result) if not isinstance(result, list) else result


def test_model_runner_prefill_decode():
    model = load_model(MODEL)
    device = next(model.parameters()).device
    runner = GPUModelRunner(
        model, device, eos_token_id=2, block_size=16,
        max_num_blocks=64, max_model_len=128,
    )

    prompt_ids = _get_input_ids("Hello world")
    prefill_out = SchedulerOutput(
        scheduled_new_reqs=[
            NewRequestData(
                req_id="test",
                prompt_token_ids=prompt_ids,
                block_ids=list(range(8)),
            ),
        ],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        total_num_scheduled_tokens=len(prompt_ids),
    )
    result = runner.execute_model(prefill_out)
    assert "test" in result.sampled_token_ids
    first = result.sampled_token_ids["test"][0]
    assert first != 2
    assert runner.get_step("test") == len(prompt_ids) + 1

    # Decode a few steps
    for _ in range(5):
        decode_out = SchedulerOutput(
            scheduled_cached_reqs=CachedRequestData(req_ids=["test"]),
            total_num_scheduled_tokens=1,
        )
        result = runner.execute_model(decode_out)
        if "test" in result.sampled_token_ids:
            assert result.sampled_token_ids["test"][0] >= 0

    assert runner.get_step("test") >= 7

    runner.cleanup("test")
    assert runner.get_step("test") == 0
