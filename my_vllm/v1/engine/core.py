import torch

from my_vllm.v1.core.block_pool import BlockPool
from my_vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from my_vllm.v1.core.sched.scheduler import Scheduler
from my_vllm.v1.engine.output_processor import OutputProcessor, RequestOutput
from my_vllm.v1.executor.abstract import Executor
from my_vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from my_vllm.v1.request import Request, RequestStatus


class EngineCore:
    """Core execution loop — mirrors vLLM V1 v1/engine/core.py:EngineCore.

    C017: batch prefill via executor.execute_model().
    C025: output processing delegated to OutputProcessor; LLMEngine
    receives structured RequestOutput instead of raw Request objects.
    """

    def __init__(
        self,
        executor: Executor,
        device: torch.device,
        max_new_tokens: int = 50,
        eos_token_id: int = 2,
        max_running: int = 8,
        num_kv_blocks: int = 512,
        block_size: int = 16,
        tokenizer=None,
        num_kv_heads: int = 12,
        head_size: int = 64,
    ):
        self.max_new_tokens = max_new_tokens
        self.eos_token_id = eos_token_id
        self.device = device

        self.executor = executor
        self.worker = executor.get_worker()

        block_pool = BlockPool(num_kv_blocks, block_size, enable_caching=True)

        # C034: build KVCacheConfig → KVCacheCoordinator
        spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=num_kv_blocks,
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=["all"],
                    kv_cache_spec=spec,
                )
            ],
        )
        self.kv_cache_coordinator = KVCacheCoordinator(
            kv_cache_config=kv_cache_config,
            block_pool=block_pool,
        )
        self.scheduler = Scheduler(
            max_running=max_running,
            kv_cache_coordinator=self.kv_cache_coordinator,
        )
        self.finished: list[Request] = []

        # C025: OutputProcessor with detokenizer
        self.output_processor = OutputProcessor(tokenizer)

    def add_request(self, request: Request):
        self.scheduler.add_request(request)

    def has_work(self) -> bool:
        return self.scheduler.has_work()

    def step(self) -> list[RequestOutput]:
        """Run one scheduling + execution step, return finished outputs.

        C032: scheduler.update_from_output() handles finish detection;
        EngineCore frees resources for finished requests.
        """
        scheduler_output = self.scheduler.schedule()
        if scheduler_output.total_num_scheduled_tokens == 0:
            return []

        model_output = self.executor.execute_model(scheduler_output)
        if model_output is None:
            return []

        engine_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output,
            eos_token_id=self.eos_token_id,
            max_tokens=self.max_new_tokens,
        )

        req_map = {r.request_id: r for r in self.scheduler.running}

        # C025: delegate to OutputProcessor
        outputs = self.output_processor.process_outputs(
            model_output, req_map,
            eos_token_id=self.eos_token_id,
            max_tokens=self.max_new_tokens,
        )

        for req in engine_outputs.finished_requests:
            if req not in self.finished:
                self._finish_request(req)

        return [o for o in outputs if o.finished]

    def _finish_request(self, req: Request):
        rid = req.request_id
        self.kv_cache_coordinator.free(rid)
        self.worker.cleanup(rid)
        self.scheduler.finish(req)
        self.output_processor.cleanup(rid)
        self.finished.append(req)
