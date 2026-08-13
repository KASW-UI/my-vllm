"""Attention layer package — aligns with vLLM vllm/model_executor/layers/attention/.

Mirrors the vLLM layout where the Attention nn.Module lives in
`attention/attention.py`. Importers use
`from my_vllm.model_executor.layers.attention import Attention`
which re-exports from this package.

Engineering details skipped (principled scope reduction):
- cross_attention / encoder_only_attention / mla_attention / mm_encoder_attention
  variants — single Attention path only (DECODER).
- chunked_local_attention / pcp / prefill_prefix_lm_attention / rswa_attention /
  sparse_mla_attention / static_sink_attention — specialized attention modes.
- kv_transfer_utils — no KV transfer connector at this layer.
"""

from my_vllm.model_executor.layers.attention.attention import Attention

__all__ = ["Attention"]