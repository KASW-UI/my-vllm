"""InputProcessor — prompt → tokenize → Request.

Mirrors vLLM V1 v1/engine/input_processor.py (simplified).

C026: extracts prompt tokenization and Request creation from
LLMEngine.add_request() into a dedicated component.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from my_vllm.v1.request import Request

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class InputProcessor:
    """Converts raw prompt strings to Request objects.

    Responsibilities:
    - Generate unique request IDs
    - Tokenize prompt text
    - Create Request objects with tokenized prompt + parameters
    """

    def __init__(
        self,
        tokenizer,
        default_max_tokens: int = 50,
    ):
        self._tokenizer = tokenizer
        self._default_max_tokens = default_max_tokens
        self._counter: int = 0

    def process(
        self,
        prompt: str | list[int],
        max_new_tokens: int | None = None,
        request_id: str | None = None,
    ) -> Request:
        """Convert a prompt into a Request object.

        Args:
            prompt: raw text string or pre-tokenized list of ints
            max_new_tokens: max tokens to generate (uses default if None)
            request_id: custom request_id (auto-generated if None)

        Returns:
            Request object ready for EngineCore.add_request()
        """
        if request_id is None:
            request_id = f"req-{self._counter}"
            self._counter += 1

        max_tok = max_new_tokens if max_new_tokens is not None else self._default_max_tokens

        if isinstance(prompt, list):
            token_ids = list(prompt)
        else:
            encode_result = self._tokenizer.encode(prompt)
            if isinstance(encode_result, list):
                token_ids = encode_result
            elif hasattr(encode_result, 'tolist'):
                token_ids = encode_result.tolist()
            else:
                token_ids = list(encode_result)

        req = Request(request_id=request_id)
        req.prompt_token_ids = token_ids
        req.max_tokens = max_tok
        return req
