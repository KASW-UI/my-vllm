"""OutputProcessor — converts engine outputs into structured RequestOutputs.

Mirrors vLLM V1 v1/engine/output_processor.py (simplified).

Consumes EngineCoreOutputs (already processed by Scheduler for EOS/length),
only performs detokenization. EOS/length handling moved to
Scheduler.update_from_output().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from my_vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs

if TYPE_CHECKING:
    from my_vllm.v1.engine.detokenizer import IncrementalDetokenizer
    from my_vllm.v1.request import Request


@dataclass
class RequestOutput:
    """Per-request output after processing a model step.

    Mirrors vLLM V1's RequestOutput (simplified — no logprobs, pooling).
    """

    request_id: str
    prompt_text: str
    output_text: str = ""
    delta_text: str = ""
    output_tokens: int = 0
    prompt_tokens: int = 0
    finish_reason: str = ""
    finished: bool = False
    token_ids: list[int] = field(default_factory=list)

    def is_finished(self) -> bool:
        return self.finished

    @property
    def output_token_ids(self) -> list[int]:
        return self.token_ids


class OutputProcessor:
    """Processes EngineCoreOutputs into structured RequestOutput.

    Responsibilities:
    - Map EngineCoreOutput to Request objects
    - Incremental detokenization (no EOS/length — handled by Scheduler)
    """

    def __init__(self, tokenizer=None):
        self._tokenizer = tokenizer
        self._detokenizers: dict[str, IncrementalDetokenizer] = {}

    def _decode(self, token_ids: list[int]) -> str:
        if self._tokenizer is None:
            return ""
        return self._tokenizer.decode(token_ids)

    def process_outputs(
        self,
        engine_core_outputs: EngineCoreOutputs,
        req_map: dict[str, Request],
    ) -> list[RequestOutput]:
        """Process engine outputs and return per-request outputs.

        Args:
            engine_core_outputs: processed outputs from Scheduler
            req_map: mapping request_id → Request

        Returns:
            List of RequestOutput, one per request that produced tokens
            this step.  Finished requests have finished=True.
        """
        outputs: list[RequestOutput] = []

        for eo in engine_core_outputs.outputs:
            req = req_map.get(eo.request_id)
            if req is None:
                continue

            detok = self._get_detokenizer(eo.request_id)
            delta = detok.update(eo.new_token_ids)

            is_finished = eo.finish_reason is not None

            outputs.append(RequestOutput(
                request_id=eo.request_id,
                prompt_text=self._decode(req.prompt_token_ids),
                output_text=detok.full_text(),
                delta_text=delta,
                output_tokens=req.num_output_tokens,
                prompt_tokens=len(req.prompt_token_ids),
                finish_reason=eo.finish_reason.name if eo.finish_reason is not None else "",
                finished=is_finished,
                token_ids=detok.token_ids,
            ))

        return outputs

    def _get_detokenizer(self, request_id: str) -> IncrementalDetokenizer:
        from my_vllm.v1.engine.detokenizer import IncrementalDetokenizer

        if request_id not in self._detokenizers:
            self._detokenizers[request_id] = IncrementalDetokenizer(
                self._tokenizer,
            )
        return self._detokenizers[request_id]

    def cleanup(self, request_id: str):
        self._detokenizers.pop(request_id, None)