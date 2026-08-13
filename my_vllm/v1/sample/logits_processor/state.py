"""LogitsProcessor state — batch-update bookkeeping for processor ordering.

Mirrors vLLM V1 v1/sample/logits_processor/state.py.

C132: stub file matching the vLLM file-tree layout. vLLM's
`BatchUpdateBuilder` tracks per-step batch-membership changes so that
logits processors can update their per-request state. my-vLLM's
penalties are stateless (free functions in `builtin.py`), so the state
machinery is not needed today — this file exists only for the
1:1 layout alignment.
"""