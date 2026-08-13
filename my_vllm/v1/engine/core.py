"""EngineCore — core execution loop. Mirrors vllm V1 v1/engine/core.py.

The hot path (step) is a 12-line pipeline:

    schedule → execute → update_from_output →
    finish_completed → kv_transfer → collect_metrics

C017: batch prefill via executor.execute_model().
C025: output processing delegated to OutputProcessor.
C048: metrics collection — SchedulerStats + IterationStats per step.
C112: P/D disaggregation (NIXL), LoRA multiplexing, multi-tenant quotas,
      CUDA Event profiling — all reachable as side-module helpers,
      not inlined into step().
C121: tighten step() body to a top-to-bottom pipeline. Every concern
      beyond schedule/execute/update_from_output is delegated to a
      helper; the main path reads top-to-bottom like nano-vllm.
C124: EngineCoreRequestType + EngineShutdownState + input/output queues
      + run_busy_loop primitives (no DP / ZMQ / batch_queue). The loop
      still defaults to caller-driven step(); BusyLoopClient arrives at
      C125.
"""

from __future__ import annotations

import logging
import queue
import time
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import torch

from my_vllm.v1.core.block_pool import BlockPool
from my_vllm.v1.core.kv_cache_manager import KVCacheManager
from my_vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from my_vllm.v1.core.kv_connector import TransferBundle
from my_vllm.v1.core.sched.scheduler import Scheduler
from my_vllm.v1.engine import EngineCoreOutputs, EngineCoreRequestType
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


class EngineShutdownState(IntEnum):
    """Lifecycle state for EngineCore shutdown.

    Mirrors vLLM V1 v1/engine/core.py:EngineShutdownState (L989).
    """

    RUNNING = 0
    REQUESTED = 1
    SHUTTING_DOWN = 2


class EngineCore:
    """Core execution loop — mirrors vLLM V1 v1/engine/core.py:EngineCore.

    step() body is intentionally short: every per-step concern beyond the
    core V1 loop (schedule → execute → update_from_output) lives in a
    dedicated helper. The 12-line top-to-bottom shape matches nano-vllm's
    LLMEngine.step() while still wiring C112 features (LoRA, multi-tenant,
    NIXL P/D) via the side-module helpers.
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

        # C124: busy-loop primitives. input_queue carries
        # (EngineCoreRequestType, request) tuples from any thread; the
        # loop owns output_queue and pushes EngineCoreOutputs.
        self.input_queue: queue.Queue[tuple[EngineCoreRequestType, Any]] = (
            queue.Queue()
        )
        self.output_queue: queue.Queue[EngineCoreOutputs] = queue.Queue()
        self.shutdown_state = EngineShutdownState.RUNNING

    # ── Public API ──────────────────────────────────────────────────────

    def add_request(self, request: Request):
        self.scheduler.add_request(request)
        self._request_arrival[request.request_id] = request.arrival_time

    def has_work(self) -> bool:
        return self.scheduler.has_work()

    # ── C124: busy-loop primitives ──────────────────────────────────────
    #
    # The methods below add `run_busy_loop` + input/output queues to
    # EngineCore. Caller-driven `step()` is preserved unchanged — Inproc
    # clients (C125 / C126) keep the current synchronous path; only
    # BusyLoopClient (C125) actually spins the loop.

    def is_running(self) -> bool:
        """Returns true if shutdown has not been requested.

        Mirrors vLLM V1 v1/engine/core.py:EngineCore.is_running (L1354).
        """
        return self.shutdown_state == EngineShutdownState.RUNNING

    def run_busy_loop(self) -> None:
        """Core busy loop of the EngineCore.

        Mirrors vLLM V1 v1/engine/core.py:EngineCore.run_busy_loop
        (L1358). Simplified: no ``SystemExit`` (callers decide exit
        policy), no DP ``engines_running`` / ``batch_queue`` / DP wave
        / EEP / UTILITY RPC.
        """
        while self._handle_shutdown():
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # 2) Step the engine core and forward outputs.
            self._process_engine_step()

    def _process_input_queue(self) -> None:
        """Block on the input queue until work shows up.

        Mirrors vLLM V1 v1/engine/core.py:_process_input_queue (L1368).
        Drops the aborts_queue drain (my-vllm merges aborts into
        input_queue as ADD/ABORT pairs).
        """
        while not self.has_work() and self.is_running():
            try:
                req = self.input_queue.get()
            except queue.Empty:
                break
            self._handle_client_request(*req)

        # Drain anything that arrived after the first request landed.
        while not self.input_queue.empty():
            req = self.input_queue.get_nowait()
            self._handle_client_request(*req)

    def _process_engine_step(self) -> bool:
        """Run one step; forward non-empty outputs to the output queue.

        Mirrors vLLM V1 v1/engine/core.py:_process_engine_step (L1399).
        """
        outputs = self.step()
        if outputs.outputs:
            self.output_queue.put_nowait(outputs)
        return bool(outputs.outputs)

    def _handle_shutdown(self) -> bool:
        """Drive the shutdown state machine. Return False to stop the loop.

        Mirrors vLLM V1 v1/engine/core.py:_handle_shutdown (L1423).
        Simplified: no shutdown_timeout-driven abort/drain (the
        Config layer does not yet expose ``shutdown_timeout``); the
        loop simply stops accepting new work and drains whatever is in
        flight until the scheduler reports no unfinished requests.
        """
        if self.shutdown_state == EngineShutdownState.RUNNING:
            return True
        if self.shutdown_state == EngineShutdownState.REQUESTED:
            self.shutdown_state = EngineShutdownState.SHUTTING_DOWN
        return self.has_work()

    def _handle_client_request(
        self,
        request_type: EngineCoreRequestType,
        request: Any,
    ) -> None:
        """Dispatch a request pulled from the input queue.

        Mirrors vLLM V1 v1/engine/core.py:_handle_client_request (L1471).
        Only ADD / ABORT / WAKEUP are honored; UTILITY / EXECUTOR_FAILED
        are out of scope (LOOP-PLAN §2 C124).
        """
        if request_type == EngineCoreRequestType.ADD:
            self.add_request(request)
        elif request_type == EngineCoreRequestType.ABORT:
            self.scheduler.finish_requests(
                request, RequestStatus.FINISHED_ABORTED,
            )
        elif request_type == EngineCoreRequestType.WAKEUP:
            return
        else:
            logger.error(
                "Unrecognized input request type encountered: %s",
                request_type,
            )

    def shutdown(self) -> None:
        """Request graceful shutdown; wake the busy loop via WAKEUP.

        Mirrors vLLM V1 v1/engine/core.py:EngineCore.shutdown (L743)
        for the queue-handshake part (the resource teardown at the
        end of vLLM's shutdown lives in EngineCoreProc and is not
        relevant to the in-process EngineCore class).
        """
        self.shutdown_state = EngineShutdownState.REQUESTED
        # Sentinel to unblock _process_input_queue.get() if it is idle.
        self.input_queue.put((EngineCoreRequestType.WAKEUP, None))

    def step(self) -> EngineCoreOutputs:
        """One scheduling + execution step. Top-to-bottom pipeline."""
        self._step_count += 1
        step_start = time.time()

        # 1. P/D consumer recv (no-op unless kv_transfer_config + is_consumer).
        self._maybe_consumer_recv()

        # 2. Schedule: pick what to run this step.
        scheduler_output = self.scheduler.schedule()
        if scheduler_output.total_num_scheduled_tokens == 0:
            return EngineCoreOutputs()

        # 3. Execute: model forward.
        model_output = self.executor.execute_model(scheduler_output)
        if model_output is None:
            return EngineCoreOutputs()

        # 4. Update: scheduler advances req state from model output.
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output,
        )

        # 5. Side concerns (each no-ops if disabled).
        self._finish_completed_requests(engine_core_outputs)
        self._maybe_producer_transfer(engine_core_outputs)
        self._collect_metrics(
            scheduler_output, model_output,
            engine_core_outputs, step_start, time.time(),
        )

        return engine_core_outputs

    # ── Side concerns (each is a self-contained helper) ────────────────

    def _finish_completed_requests(
        self, engine_core_outputs: EngineCoreOutputs,
    ) -> None:
        """Free KV blocks + cleanup worker state for finished reqs.

        Mirrors vLLM's request-finish handling inside step().
        """
        for eo in engine_core_outputs.outputs:
            if eo.finish_reason is None:
                continue
            req = self._find_request(eo.request_id)
            if req is None or req in self.finished:
                continue
            self._finish_request(req)

    def _maybe_consumer_recv(self) -> None:
        """P/D consumer: receive KV bundles from producer before schedule.

        No-op unless kv_transfer_config is set and the engine is a consumer.
        """
        if self.kv_transfer_config is None:
            return
        if not self.kv_transfer_config.is_kv_consumer:
            return
        self._consumer_recv_loop()

    def _maybe_producer_transfer(
        self, engine_core_outputs: EngineCoreOutputs,
    ) -> None:
        """P/D producer: ship finished prefill blocks to consumer.

        No-op unless kv_transfer_config is set and the engine is a producer.
        """
        if self.kv_transfer_config is None:
            return
        if not self.kv_transfer_config.is_kv_producer:
            return
        self._producer_transfer_loop(engine_core_outputs)

    def _collect_metrics(
        self,
        scheduler_output,
        model_output,
        engine_core_outputs: EngineCoreOutputs,
        step_start: float,
        step_end: float,
    ) -> None:
        """Build SchedulerStats + IterationStats for this step and log them.

        Mirrors vLLM Stats.update_from_output + StatLogger.record. Captures
        TTFT, TPOT, ITL, KV cache usage, prefix-cache hit events.
        """
        eviction_events = self._kv_metrics_collector.drain_events()

        kv_usage = 1.0 - (
            self.kv_cache_manager.num_free_blocks
            / self.kv_cache_manager.block_pool.num_blocks
        )
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
            if eo.finish_reason is None:
                continue
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

    # ── helpers ─────────────────────────────────────────────────────────

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

    # ── P/D disaggregation helpers (C112) ───────────────────────────────

    def _kv_connector_init(self) -> None:
        """Create KV connector for P/D transfer.

        Uses NoopConnector as default when no real connector is configured.
        Aligns with vLLM KVConnectorFactory.create_connector().
        """
        from my_vllm.v1.core.kv_connector import NoopConnector
        self.kv_connector = NoopConnector()

    def _producer_transfer_loop(self, outputs: EngineCoreOutputs) -> None:
        """P side: transfer finished prefill requests to consumer.

        For each finished prefill request:
        1. Extract KV blocks from BlockPool.
        2. Send to consumer via connector.
        3. Defer block free until ACK received.

        Also drains ACKs from consumer to release deferred blocks.
        Aligns with vLLM's deferred block free + KV connector send.
        """
        for eo in outputs.outputs:
            if eo.finish_reason is None:
                continue
            req = self._find_request(eo.request_id)
            if req is None:
                continue

            blocks = self.kv_cache_manager.pop_blocks_for_free(req.request_id)
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
        1. Allocate blocks in local BlockPool.
        2. Register request in scheduler with all tokens already computed.
        3. Send ACK back to producer.

        For failed transfers:
        - policy=="fail": mark request FINISHED_ERROR.
        - policy=="recompute": insert for prefill with truncated tokens.

        Aligns with vLLM Scheduler._update_from_kv_xfer_finished +
        _handle_invalid_blocks.
        """
        if self.kv_connector is None:
            return

        result = self.kv_connector.check_transfer_done(
            timeout=self.transfer_timeout,
        )

        for bundle in result.bundles:
            req = Request(
                request_id=bundle.request_id,
                prompt_token_ids=[],
                sampling_params=None,
                arrival_time=time.time(),
            )
            req.num_computed_tokens = bundle.num_tokens
            req.status = RequestStatus.RUNNING

            self.kv_cache_manager.allocate_slots(req, bundle.num_tokens)
            self.scheduler.running.append(req)

            if self.kv_connector is not None:
                self.kv_connector.send_ack(bundle.request_id, True)

        for req_id in result.invalid_req_ids:
            if self.failure_policy == "fail":
                self.scheduler.finish_requests(
                    [req_id], RequestStatus.FINISHED_ERROR,
                )
            else:
                valid = result.valid_prefix_tokens.get(req_id, 0)
                req = Request(
                    request_id=req_id,
                    prompt_token_ids=[],
                    sampling_params=None,
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
                # ACK (success or failure) → release blocks either way.
                self.kv_cache_manager.free(req_id)
                continue

            if now > expire:
                if self.failure_policy == "recompute":
                    logger.warning(
                        "KV transfer timeout for %s, recomputing", req_id,
                    )
                else:
                    logger.error(
                        "KV transfer timeout for %s, failing", req_id,
                    )
                self.kv_cache_manager.free(req_id)
                continue

            remaining.append((req_id, expire))

        self._deferred_frees = remaining