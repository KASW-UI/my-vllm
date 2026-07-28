"""IncrementalDetokenizer — incremental text decoding.

Mirrors vLLM V1 v1/engine/detokenizer.py (simplified).

C025: decouples token→text conversion from LLMEngine, supports
incremental decoding so each step only decodes new tokens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class IncrementalDetokenizer:
    """Token → text converter with incremental state.

    Tracks all output token IDs so far and can produce:
      - Full text: decode(all tokens)
      - Delta text: decode(new tokens since last call)
    """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._token_ids: list[int] = []
        self._cursor: int = 0  # how many tokens have been consumed

    def update(self, new_token_ids: list[int]) -> str:
        """Add new tokens and return the delta text since last call."""
        self._token_ids.extend(new_token_ids)
        delta = self._decode_delta()
        self._cursor = len(self._token_ids)
        return delta

    def full_text(self) -> str:
        """Return the full decoded text of all tokens."""
        if not self._token_ids or self._tokenizer is None:
            return ""
        return self._tokenizer.decode(self._token_ids)

    @property
    def token_ids(self) -> list[int]:
        return list(self._token_ids)

    def num_output_tokens(self) -> int:
        return len(self._token_ids)

    def _decode_delta(self) -> str:
        """Decode only the new tokens since last cursor position."""
        if self._cursor >= len(self._token_ids):
            return ""
        if self._tokenizer is None:
            return ""
        new_ids = self._token_ids[self._cursor:]
        return self._tokenizer.decode(new_ids)
