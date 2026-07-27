"""Public API — the main entry point for external users."""

from my_vllm.engine.request import Request, RequestStatus
from my_vllm.engine.state_manager import StateManager


class LLMEngine:
    """High-level interface for LLM inference."""

    def __init__(
        self,
        state_manager: StateManager,
        tokenizer,
    ):
        self._sm = state_manager
        self._tokenizer = tokenizer
        self._pending: dict[str, Request] = {}

    def add_request(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
    ) -> str:
        max_tok = max_new_tokens or self._sm.max_new_tokens
        req = Request(request_id=f"req-{len(self._pending)}")
        req.prompt_token_ids = self._tokenizer.encode(prompt)
        req.max_tokens = max_tok
        self._pending[req.request_id] = req
        self._sm.add_request(req)
        return req.request_id

    def step(self) -> list[dict]:
        """Run one scheduling + execution step and return finished outputs."""
        results: list[dict] = []
        finished = self._sm.step()
        for req in finished:
            text = self._tokenizer.decode(req.output_token_ids)
            prompt = self._tokenizer.decode(req.prompt_token_ids)
            results.append({
                "request_id": req.request_id,
                "text": prompt + text,
                "finish_reason": req.status.name,
                "output_tokens": len(req.output_token_ids),
                "prompt_tokens": len(req.prompt_token_ids),
            })
            self._pending.pop(req.request_id, None)
        return results

    def has_work(self) -> bool:
        return self._sm.has_work()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
    ) -> dict:
        """Blocking single-request generation."""
        req_id = self.add_request(prompt, max_new_tokens=max_new_tokens)
        while self.has_work():
            results = self.step()
            for r in results:
                if r["request_id"] == req_id:
                    return r
        return {"request_id": req_id, "text": "", "finish_reason": "UNKNOWN"}
