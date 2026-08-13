"""Public API — the main entry point for external users.

C015: migrated from engine/llm_engine.py.
C025: uses OutputProcessor for token→text; receives structured
RequestOutput from EngineCore.step().
C026: InputProcessor handles prompt → tokenize → Request.
C126: LLMEngine holds an EngineCoreClient (InprocClient / BusyLoopClient)
instead of an EngineCore directly. Default ``busy_loop=False`` keeps the
caller-driven semantics of run.py / LLMEngine.generate(). AsyncLLM /
api_server rewire to BusyLoopClient at C127.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from my_vllm.v1.engine.core_client import EngineCoreClient
from my_vllm.v1.engine.input_processor import InputProcessor
from my_vllm.v1.engine.output_processor import OutputProcessor
from my_vllm.v1.request import Request

if TYPE_CHECKING:
    from my_vllm.config.vllm import VllmConfig
    from my_vllm.v1.engine.core_client import EngineCoreClient
    from my_vllm.v1.executor.abstract import Executor


class LLMEngine:
    """High-level interface for LLM inference.

    Mirrors vLLM V1 v1/engine/llm_engine.py:48 (simplified).

    Receives VllmConfig + executor_class + tokenizer.
    Owns an ``EngineCoreClient`` (InprocClient or BusyLoopClient, chosen
    by ``busy_loop``) + OutputProcessor internally.
    step() flow: ``engine_core.get_output()`` → EngineCoreOutputs →
    OutputProcessor.process_outputs() → list[RequestOutput].
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        tokenizer,
        busy_loop: bool = False,
    ):
        self._tokenizer = tokenizer
        self.engine_core: EngineCoreClient = EngineCoreClient.make_client(
            busy_loop=busy_loop,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=False,
        )
        self._output_processor = OutputProcessor(tokenizer)
        self._pending: dict[str, Request] = {}

        self._input_processor = InputProcessor(
            tokenizer,
            default_max_tokens=256,
        )

    def add_request(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
    ) -> str:
        req = self._input_processor.process(prompt, max_new_tokens)
        self._pending[req.request_id] = req
        self.engine_core.add_request(req)
        return req.request_id

    def step(self) -> list[dict]:
        """Run one scheduling + execution step and return finished outputs."""
        results: list[dict] = []
        engine_core_outputs = self.engine_core.get_output()

        req_map = {
            r.request_id: r
            for r in self.engine_core.running_requests()
        }

        outputs = self._output_processor.process_outputs(
            engine_core_outputs, req_map,
        )

        for eo in engine_core_outputs.outputs:
            if eo.finish_reason is not None:
                self._output_processor.cleanup(eo.request_id)

        for out in outputs:
            prompt_text = out.prompt_text
            if not prompt_text and out.request_id in self._pending:
                req = self._pending[out.request_id]
                prompt_text = self._tokenizer.decode(req.prompt_token_ids)
            results.append({
                "request_id": out.request_id,
                "text": prompt_text + out.output_text,
                "prompt_text": prompt_text,
                "output_text": out.output_text,
                "delta_text": out.delta_text,
                "finish_reason": out.finish_reason,
                "output_tokens": out.output_tokens,
                "prompt_tokens": out.prompt_tokens,
            })
            if out.finished:
                self._pending.pop(out.request_id, None)
        return results

    def has_work(self) -> bool:
        return self.engine_core.has_work()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
    ) -> dict:
        """Blocking single-request generation.

        Runs steps until the request finishes (EOS, length cap, etc.) or
        has_work() drains. Each step may produce zero or more outputs for the
        request; we accumulate the latest output_text and return once
        finish_reason is set.
        """
        req_id = self.add_request(prompt, max_new_tokens=max_new_tokens)
        last_result: dict | None = None
        while self.has_work():
            results = self.step()
            for r in results:
                if r["request_id"] == req_id:
                    last_result = r
                    if r.get("finish_reason"):
                        return r
        if last_result is not None:
            return last_result
        return {"request_id": req_id, "text": "", "finish_reason": "UNKNOWN"}