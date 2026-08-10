"""OutputProcessor — converts model outputs into structured RequestOutputs.

Mirrors vLLM V1 v1/engine/output_processor.py (simplified).

C025: extracts output processing from EngineCore.step() into
a dedicated component.  Takes SchedulerOutput + ModelRunnerOutput
and produces a list of finished Request objects with processed
token IDs and status transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from my_vllm.v1.engine.detokenizer import IncrementalDetokenizer
    from my_vllm.v1.outputs import ModelRunnerOutput
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
    """Processes model outputs into structured RequestOutput.

    Responsibilities:
    - Map sampled token IDs back to Request objects
    - Detect EOS / max_tokens → mark finished
    - Append tokens to requests
    - Incremental detokenization via IncrementalDetokenizer
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
        model_output: ModelRunnerOutput,
        req_map: dict[str, Request],
        eos_token_id: int,
        max_tokens: int,
    ) -> list[RequestOutput]:
        """Process model output and return per-request outputs.

        Args:
            model_output: tokens sampled by the model runner
            req_map: mapping request_id → Request
            eos_token_id: EOS token ID
            max_tokens: default max tokens per request

        Returns:
            List of RequestOutput, one per request that produced tokens
            this step.  Finished requests have finished=True.
        """
        outputs: list[RequestOutput] = []

        for i, rid in enumerate(model_output.req_ids):
            token_ids = model_output.sampled_token_ids[i]
            req = req_map.get(rid)
            if req is None:
                continue

            detok = self._get_detokenizer(rid)
            appended_count = 0

            for token in token_ids:
                if token == eos_token_id:
                    req.status = req.status.__class__.FINISHED_STOPPED
                    break
                req.append_output_token_ids(token)
                appended_count += 1

            new_out = req._output_token_ids[-appended_count:] if appended_count else []
            delta = detok.update(new_out)
            step_val = req.num_output_tokens
            limit = req.max_tokens or max_tokens

            if step_val >= limit and not req.is_finished():
                req.status = req.status.__class__.FINISHED_LENGTH_CAPPED

            is_finished = req.is_finished()

            outputs.append(RequestOutput(
                request_id=rid,
                prompt_text=self._decode(req.prompt_token_ids),
                output_text=detok.full_text(),
                delta_text=delta,
                output_tokens=req.num_output_tokens,
                prompt_tokens=len(req.prompt_token_ids),
                finish_reason=req.status.name if is_finished else "",
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
