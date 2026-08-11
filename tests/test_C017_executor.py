"""C017: tests for Executor / UniProcExecutor (v1/executor/)."""

import torch

from my_vllm.entrypoints.llm import load_tokenizer
from my_vllm.model_executor.model_loader import load_model
from my_vllm.v1.worker.gpu_worker import Worker
from my_vllm.v1.executor.uniproc_executor import UniProcExecutor
from my_vllm.v1.core.sched.output import (
    SchedulerOutput, NewRequestData, CachedRequestData,
)
from my_vllm.v1.outputs import ModelRunnerOutput

MODEL = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m"


class TestUniProcExecutor:
    def test_execute_model(self):
        model = load_model(MODEL)
        worker = Worker(model, model.device)
        worker.init_device()
        executor = UniProcExecutor(worker)
        tokenizer = load_tokenizer(MODEL)

        o = SchedulerOutput(
            scheduled_new_reqs=[
                NewRequestData(
                    req_id="ex1",
                    prompt_token_ids=tokenizer.encode("Hello"),
                ),
            ],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            total_num_scheduled_tokens=1,
        )
        result = executor.execute_model(o)
        assert isinstance(result, ModelRunnerOutput)
        assert len(result.sampled_token_ids) == 1

    def test_get_worker(self):
        model = load_model(MODEL)
        worker = Worker(model, model.device)
        executor = UniProcExecutor(worker)
        assert executor.get_worker() is worker
