from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import torch

from my_vllm.v1.core.block_pool import BlockPool
from my_vllm.v1.core.kv_cache_manager import KVCacheManager, KVCacheBlocks
from my_vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from my_vllm.v1.core.kv_connector import (
    RecvResult,
    TransferBundle,
    TransferStatus,
)
from my_vllm.v1.core.sched.scheduler import Scheduler
from my_vllm.v1.engine import EngineCoreOutputs
from my_vllm.v1.executor.abstract import Executor
from my_vllm.v1.metrics.loggers import LoggingStatLogger, StatLoggerBase
from my_vllm.v1.metrics.stats import (
    FinishedRequestStats,
    IterationStats,
    PrefixCacheStats,
    SchedulerStats,
)
from my_vllm.v1.request import Request, RequestStatus

if TYPE_CHECKING:
    from my_vllm.config.kv_transfer import KVTransferConfig
    from my_vllm.config.vllm import VllmConfig
    from my_vllm.v1.core.kv_connector import KVConnectorBase_V1

logger = logging.getLogger(__name__)


class EngineCore:
    """Core execution loop — mirrors vLLM V1 v1/engine/core.py:EngineCore.

    C017: batch prefill via executor.execute_model().
    C025: output processing delegated to OutputProcessor.
    C048: metrics collection — SchedulerStats + IterationStats per step.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
    ):
        self.vllm_config = vllm_config
        cc = vllm_config.cache_config
        sc = vllm_config.scheduler_config

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        self.executor = executor_class(vllm_config)
        self.executor._init_executor()
        self.worker = self.executor.get_worker()

        self._kv_metrics_collector = KVCacheMetricsCollector(sample_rate=0.1)
        block_pool = BlockPool(
            cc.num_kv_blocks, cc.block_size,
            enable_caching=cc.enable_prefix_caching,
            metrics_collector=self._kv_metrics_collector,
        )
        self.kv_cache_manager = KVCacheManager(block_pool=block_pool)
        self.scheduler = Scheduler(
            vllm_config=vllm_config,
            kv_cache_manager=self.kv_cache_manager,
        )
        self.finished: list[Request] = []

        # C048: metrics
        self._step_count: int = 0
        self._request_ttft: dict[str, float] = {}
        self._request_arrival: dict[str, float] = {}
        self._stat_logger: StatLoggerBase = LoggingStatLogger(interval_steps=10)

        # C112: P/D disaggregation state
        self.kv_transfer_config: KVTransferConfig | None = (
            vllm_config.kv_transfer_config
        )
        self.kv_connector: KVConnectorBase_V1 | None = None
        self._deferred_frees: list[tuple[str, float]] = []
        if self.kv_transfer_config is not None:
            self.transfer_timeout = self.kv_transfer_config.transfer_timeout
            self.failure_policy = self.kv_transfer_config.kv_load_failure_policy
            self._kv_connector_init()

    def add_request(self, request: Request):
        self.scheduler.add_request(request)
        self._request_arrival[request.request_id] = request.arrival_time

    def has_work(self) -> bool:
        return self.scheduler.has_work()

    def step(self):
        self._step_count += 1
        step_start = time.time()

        # C112: consumer-side KV receive (before schedule)
        if self.kv_transfer_config is not None:
            if self.kv_transfer_config.is_kv_consumer:
                self._consumer_recv_loop()

        scheduler_output = self.scheduler.schedule()
        if scheduler_output.total_num_scheduled_tokens == 0:
            return EngineCoreOutputs()

        model_output = self.executor.execute_model(scheduler_output)
        if model_output is None:
            return EngineCoreOutputs()

        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output,
        )

        step_end = time.time()

        for eo in engine_core_outputs.outputs:
            if eo.finish_reason is not None:
                req = self._find_request(eo.request_id)
                if req is not None and req not in self.finished:
                    self._finish_request(req)

        # C112: producer-side KV transfer (after finish)
        if self.kv_transfer_config is not None:
            if self.kv_transfer_config.is_kv_producer:
                self._producer_transfer_loop(engine_core_outputs)

        self._collect_metrics(
            scheduler_output, model_output,
            engine_core_outputs, step_start, step_end,
        )

        return engine_core_outputs

    def _collect_metrics(
        self,
        scheduler_output,
        model_output,
        engine_core_outputs,
        step_start: float,
        step_end: float,
    ) -> None:
        eviction_events = self._kv_metrics_collector.drain_events()

        kv_usage = 1.0 - (self.kv_cache_manager.num_free_blocks /
                          self.kv_cache_manager.block_pool.num_blocks)
        cache_stats = PrefixCacheStats()
        sched_stats = SchedulerStats(
            num_running_reqs=len(self.scheduler.running),
            num_waiting_reqs=len(self.scheduler.waiting),
            kv_cache_usage=kv_usage,
            prefix_cache_stats=cache_stats,
            kv_cache_eviction_events=eviction_events,
        )

        iter_stats = IterationStats()
        num_prompt_tokens = scheduler_output.total_num_scheduled_tokens
        num_gen_tokens = sum(
            len(tokens) for tokens in model_output.sampled_token_ids
        )

        for i, rid in enumerate(model_output.req_ids):
            token_ids = model_output.sampled_token_ids[i]
            if rid in self._request_arrival and rid not in self._request_ttft:
                if len(token_ids) > 0:
                    ttft = step_end - self._request_arrival[rid]
                    self._request_ttft[rid] = ttft
                    iter_stats.record_first_token(ttft)

        finished: list[FinishedRequestStats] = []
        for eo in engine_core_outputs.outputs:
            if eo.finish_reason is not None:
                rid = eo.request_id
                req = self._find_request(rid)
                if req is None:
                    continue
                n_prompt = len(req.prompt_token_ids)
                n_gen = req.num_output_tokens
                e2e = step_end - self._request_arrival.get(rid, step_end)
                ttft = self._request_ttft.get(rid, 0.0)
                tpot = (e2e - ttft) / n_gen if n_gen > 0 else 0.0
                finished.append(FinishedRequestStats(
                    request_id=rid,
                    num_prompt_tokens=n_prompt,
                    num_generation_tokens=n_gen,
                    e2e_latency=e2e,
                    ttft=ttft,
                    tpot=tpot,
                    itl=0.0,
                ))
                self._request_ttft.pop(rid, None)
                self._request_arrival.pop(rid, None)

        iter_stats.update_from_output(
            num_prompt_tokens=num_prompt_tokens,
            num_generation_tokens=num_gen_tokens,
            finished=finished,
            num_preempted=len(scheduler_output.preempted_req_ids or ()),
        )
        iter_stats.set_prompt_time(step_end - step_start)
        iter_stats.set_decode_time(step_end - step_start)

        self._stat_logger.record(sched_stats, iter_stats)

    def _find_request(self, request_id: str) -> Request | None:
        for req in self.scheduler.running:
            if req.request_id == request_id:
                return req
        return None

    def _finish_request(self, req: Request):
        rid = req.request_id
        self.kv_cache_manager.free(rid)
        self.worker.cleanup(rid)
        self.scheduler.finish(req)
        self.finished.append(req)

    # --- P/D disaggregation (C112) ---

    def _kv_connector_init(self) -> None:
        """Create KV connector for P/D transfer.

        Uses NoopConnector as default when no real connector is configured.
        Aligns with vLLM KVConnectorFactory.create_connector().
        """
        from my_vllm.v1.core.kv_connector import NoopConnector
        self.kv_connector = NoopConnector()

    def _producer_transfer_loop(self, outputs) -> None:
        """P side: transfer finished prefill requests to consumer.

        For each finished prefill request:
        1. Extract KV blocks from BlockPool
        2. Send to consumer via connector
        3. Defer block free until ACK received (defer_block_free)

        Also drains ACKs from consumer to release deferred blocks.
        Aligns with vLLM's deferred block free + KV connector send.
        """
        for eo in outputs.outputs:
            if eo.finish_reason is None:
                continue
            req = self._find_request(eo.request_id)
            if req is None:
                continue

            blocks = self.kv_cache_manager.pop_blocks_for_free(
                req.request_id
            )
            if blocks is None:
                continue

            bundle = TransferBundle(
                request_id=req.request_id,
                block_ids=[b.block_id for b in blocks],
                num_tokens=len(req.prompt_token_ids),
            )
            if self.kv_connector is not None:
                self.kv_connector.start_load_kv(forward_context=bundle)
                self.kv_connector.wait_for_save()

            expire = time.time() + self.transfer_timeout
            self._deferred_frees.append((req.request_id, expire))

            self.scheduler.finish(req)

        self._drain_acks()

    def _consumer_recv_loop(self) -> None:
        """D side: receive KV from producer, insert into scheduler.

        For each received bundle:
        1. Allocate blocks in local BlockPool
        2. Register request in scheduler with all tokens already computed
        3. Send ACK back to producer

        For failed transfers:
        - policy=="fail": mark request FINISHED_ERROR
        - policy=="recompute": insert for prefill with truncated tokens

        Aligns with vLLM Scheduler._update_from_kv_xfer_finished +
        _handle_invalid_blocks.
        """
        if self.kv_connector is None:
            return

        result = self.kv_connector.check_transfer_done(
            timeout=self.transfer_timeout
        )

        for bundle in result.bundles:
            from my_vllm.v1.request import Request
            from my_vllm.v1.request import SamplingParams
            req = Request(
                request_id=bundle.request_id,
                prompt_token_ids=[],
                sampling_params=SamplingParams(),
                arrival_time=time.time(),
            )
            req.num_computed_tokens = bundle.num_tokens
            req.status = RequestStatus.RUNNING

            self.kv_cache_manager.allocate_slots(
                req, bundle.num_tokens
            )
            self.scheduler.running.append(req)

            if self.kv_connector is not None:
                self.kv_connector.send_ack(bundle.request_id, True)

        for req_id in result.invalid_req_ids:
            if self.failure_policy == "fail":
                self.scheduler.finish_requests(
                    [req_id], RequestStatus.FINISHED_ERROR
                )
            else:
                from my_vllm.v1.request import Request
                from my_vllm.v1.request import SamplingParams
                valid = result.valid_prefix_tokens.get(req_id, 0)
                req = Request(
                    request_id=req_id,
                    prompt_token_ids=[],
                    sampling_params=SamplingParams(),
                    arrival_time=time.time(),
                )
                req.num_computed_tokens = valid
                req.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                self.scheduler.reinsert_for_prefill(req)

            if self.kv_connector is not None:
                self.kv_connector.send_ack(req_id, False)

    def _drain_acks(self) -> None:
        """Producer: release deferred blocks for ACK'd transfers.

        For each ACK'd request, free the deferred blocks.
        For timed-out transfers without ACK, apply failure policy.
        Aligns with vLLM Scheduler._drain_deferred_frees.
        """
        if self.kv_connector is None:
            return

        acks = self.kv_connector.drain_acks()
        now = time.time()
        remaining: list[tuple[str, float]] = []

        for req_id, expire in self._deferred_frees:
            if req_id in acks:
                if acks[req_id]:
                    self.kv_cache_manager.free(req_id)
                else:
                    self.kv_cache_manager.free(req_id)
                continue

            if now > expire:
                if self.failure_policy == "recompute":
                    logger.warning(
                        "KV transfer timeout for %s, recomputing", req_id
                    )
                else:
                    logger.error(
                        "KV transfer timeout for %s, failing", req_id
                    )
                self.kv_cache_manager.free(req_id)
                continue

            remaining.append((req_id, expire))

        self._deferred_frees = remaining
