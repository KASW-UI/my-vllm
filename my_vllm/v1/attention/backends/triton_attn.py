"""TritonAttentionBackend — Triton-based paged attention backend.

Mirrors vllm/vllm/v1/attention/backends/triton_attn.py. Uses real vLLM
Triton kernels extracted from the reference codebase:
  - triton_reshape_and_cache_flash  (KV cache write)
  - unified_attention                (paged attention compute)

KV cache layout follows real vLLM's HND-packed convention:
    (num_blocks, num_kv_heads, block_size, 2 * head_size)
K is the first head_size slice of the last dim, V is the second half.
At call time the impl splits the packed cache into separate
(num_blocks, block_size, num_kv_heads, head_size) flash-layout tensors
for the two kernels (matches vllm triton_attn.py:821).
"""

import torch

from my_vllm.v1.attention.backend import AttentionBackend, AttentionImpl
from my_vllm.v1.attention.backend import CommonAttentionMetadata
from my_vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from my_vllm.v1.attention.ops.triton_unified_attention import unified_attention


class TritonAttentionBackend(AttentionBackend):
    """Triton-based paged attention backend."""

    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN"

    @staticmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        return TritonAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        # HND packed layout: K and V share the last dim (2 * head_size).
        # Mirrors vllm TritonAttentionBackend.get_kv_cache_shape (non-quant path).
        return (num_blocks, num_kv_heads, block_size, 2 * head_size)


class TritonAttentionImpl(AttentionImpl):
    """Triton paged attention forward pass.

    Mirrors vllm/v1/attention/backends/triton_attn.py:TritonAttentionImpl.forward
    (non-quantized, single KV cache group, no CP, no DCP).
    C149: KV cache dtype + per-tensor K/V scales are now
    plumbed in via `set_kv_cache_dtype(...)` /
    `set_kv_scales(k_scale, v_scale)` so the
    `triton_reshape_and_cache_flash` call takes the FP8 path
    when the engine is configured for FP8 storage.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
    ):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads
        self.alibi_slopes = None
        self.use_alibi_sqrt = False
        self.sliding_window = None
        self.logits_soft_cap = None
        # C149: KV cache dtype + per-tensor scales. Defaults
        # preserve the pre-C149 behaviour (BF16 store, scale=1).
        self.kv_cache_dtype: str = "auto"
        self._k_scale: torch.Tensor | None = None
        self._v_scale: torch.Tensor | None = None

    def set_kv_cache_dtype(self, kv_cache_dtype: str) -> None:
        """Configure the KV cache storage dtype for this impl.

        One of:
          - "auto" (default) — store KV in the model's dtype
            (BF16/FP16). Same as the pre-C149 behaviour.
          - "fp8_e4m3" / "fp8_e5m2" / "fp8" — quantize per
            tensor on write via the existing
            `reshape_and_cache_kernel_flash` FP8 path with
            `k_scale` / `v_scale`.

        Called from `Attention.__init__` after the impl is
        constructed so the dtype follows `cache_config`.
        """
        self.kv_cache_dtype = kv_cache_dtype

    def set_kv_scales(
        self,
        k_scale: torch.Tensor | None,
        v_scale: torch.Tensor | None,
    ) -> None:
        """Set the per-tensor K / V scales for FP8 quantization.

        Each scale is a 1-element float32 tensor (typically on
        the same device as the KV cache). When unset
        (defaults to `None`) the impl uses scale=1.0, which
        preserves the pre-C149 behaviour regardless of
        `kv_cache_dtype` (i.e. the FP8 cast becomes a no-op
        truncation).
        """
        self._k_scale = k_scale
        self._v_scale = v_scale

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: CommonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Compute paged attention with real vLLM Triton kernels.

        Args:
            query: [num_tokens, num_heads, head_size]
            key:   [num_tokens, num_kv_heads, head_size]
            value: [num_tokens, num_kv_heads, head_size]
            kv_cache: [num_blocks, num_kv_heads, block_size, 2 * head_size]
                      HND packed (K = [..., :head_size], V = [..., head_size:]).
            attn_metadata: per-batch CommonAttentionMetadata.
            output: [num_tokens, num_heads, head_size] pre-allocated output.

        Steps (mirrors vllm triton_attn.py:808-834):
          1. Split kv_cache (HND packed) → key_cache, value_cache
             (both flash layout [num_blocks, block_size, num_kv_heads, head_size]).
          2. Write new K/V into paged cache via triton_reshape_and_cache_flash.
          3. Compute paged attention via unified_attention kernel.
        """
        num_tokens = query.shape[0]
        if num_tokens == 0:
            return output

        slot_mapping = attn_metadata.slot_mapping
        block_table = attn_metadata.block_table_tensor
        seq_lens = attn_metadata.seq_lens
        query_start_loc = attn_metadata.query_start_loc
        max_query_len = attn_metadata.max_query_len
        max_seq_len = attn_metadata.max_seq_len

        kv_cache_dtype = self.kv_cache_dtype
        # C149: honour the per-layer scales when present,
        # otherwise fall back to scale=1.0 (the pre-C149
        # behaviour — FP8 cast becomes a no-op truncation).
        if self._k_scale is not None:
            k_scale = self._k_scale
        else:
            k_scale = torch.tensor(
                [1.0], device=query.device, dtype=torch.float32,
            )
        if self._v_scale is not None:
            v_scale = self._v_scale
        else:
            v_scale = torch.tensor(
                [1.0], device=query.device, dtype=torch.float32,
            )

        # Step 1: Split HND packed kv_cache into separate flash-layout K/V.
        # (B, H, N, 2*hs).transpose(1, 2) → (B, N, H, 2*hs) → split on last dim.
        key_cache, value_cache = kv_cache.transpose(1, 2).split(
            self.head_size, dim=-1,
        )

        # Step 2: Write K/V into paged cache.
        triton_reshape_and_cache_flash(
            key=key.contiguous(),
            value=value.contiguous(),
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=slot_mapping,
            kv_cache_dtype=kv_cache_dtype,
            k_scale=k_scale,
            v_scale=v_scale,
        )

        # Step 3: Unified paged attention.
        num_actual_tokens = attn_metadata.num_actual_tokens
        cu_seqlens_q = query_start_loc[:num_actual_tokens + 1]
        max_seqlen_q = max_query_len
        seqused_k = seq_lens
        max_seqlen_k = max_seq_len

        unified_attention(
            q=query[:num_actual_tokens].contiguous(),
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            # Triton kernel expects a tuple-like (-1, -1) when sliding window
            # is disabled; None is not subscriptable. self.sliding_window is
            # None for OPT (no sliding window).
            window_size=(self.sliding_window, self.sliding_window)
            if self.sliding_window is not None
            else (-1, -1),
            block_table=block_table,
            # Triton kernel computes `softcap > 0`; pass 0.0 when disabled.
            softcap=self.logits_soft_cap if self.logits_soft_cap is not None else 0.0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            alibi_slopes=self.alibi_slopes,
            use_alibi_sqrt=self.use_alibi_sqrt,
        )

        return output