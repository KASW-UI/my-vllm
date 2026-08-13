"""AttentionConfig — attention backend / kernel selection.

Mirrors vLLM V1 config/attention.py:17.

C137: minimal subset — backend name + a few mode flags. The full
vLLM config (MLA prefill backend, FlexAttention tile sizes,
TurboQuant splits, etc.) is out of scope for my-vllm's Dense OPT
path. The config exists for 1:1 file-tree alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# my-vllm only knows two attention backends (TRITON_ATTN / FLASH_ATTN);
# these strings are the public-facing names. vLLM's full enum covers
# ~20 backends — out of scope per docs/alignment-plan.md.
AttentionBackendName = Literal["auto", "TRITON_ATTN", "FLASH_ATTN"]


@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms.

    Mirrors vLLM V1 config/attention.py:AttentionConfig.
    """

    backend: AttentionBackendName | None = "auto"
    """Attention backend name; "auto" / None triggers selector-based
    automatic selection."""

    flash_attn_version: Literal[2, 3, 4] | None = None
    """Force a specific flash-attention version. Only valid when
    `backend == "FLASH_ATTN"`."""

    use_prefill_decode_attention: bool = False
    """Use separate prefill and decode kernels instead of the unified
    triton kernel."""

    use_non_causal: bool = False
    """Use non-causal (bidirectional) attention."""

    def compute_hash(self) -> str:
        """Hash affected by backend / kernel-mode selection."""
        return (
            f"AttentionConfig(backend={self.backend},"
            f" fa_ver={self.flash_attn_version},"
            f" prefill_decode={self.use_prefill_decode_attention})"
        )