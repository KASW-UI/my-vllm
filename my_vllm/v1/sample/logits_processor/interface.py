"""LogitsProcessor interface — ABC for pluggable logits processors.

Mirrors vLLM V1 v1/sample/logits_processor/interface.py:24.

C132: stub file matching the vLLM file-tree layout. The full
`LogitsProcessor` ABC, `BatchUpdate` dataclass, and `MoveDirectionality`
enum (which my-vLLM doesn't currently use — penalties are free
functions in `builtin.py`) live here so that adding a custom processor
later only requires subclassing, not file-tree refactoring.
"""


class LogitsProcessor:
    """Abstract base for logits processors.

    vLLM mirrors this as `vllm.v1.sample.logits_processor.interface.LogitsProcessor`.
    In my-vLLM penalties are applied as free functions
    (`builtin.apply_repetition_penalty` etc.); this class exists only
    so the package layout matches vLLM. A concrete subclass can be
    added if/when guided-decoding or min-tokens processors are wired in.
    """

    def is_argmax_invariant(self) -> bool:
        """Returns True if the processor does not change argmax (e.g. bias=0)."""
        return False