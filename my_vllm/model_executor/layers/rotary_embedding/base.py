"""Rotary Positional Embeddings Base Class — aligns with vLLM.

Architecture preserved: RotaryEmbeddingBase (CustomOp), RotaryEmbedding,
cos/sin cache computation, forward_static.

Engineering details skipped: CUDA kernel dispatch (always forward_native),
flashinfer, aiter, HIP.
"""

import torch

from my_vllm.model_executor.custom_op import CustomOp

from .common import ApplyRotaryEmb


class RotaryEmbeddingBase(CustomOp):
    """Original rotary positional embedding — aligns with vLLM."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        init_cache: bool = True,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        self.use_flashinfer = False
        self.use_aiter = False

        if init_cache:
            cache = self._compute_cos_sin_cache()
            cache = cache.to(dtype)
            self.cos_sin_cache: torch.Tensor
            self.register_buffer("cos_sin_cache", cache, persistent=False)

        self.apply_rotary_emb = ApplyRotaryEmb(
            is_neox_style=self.is_neox_style,
        )

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float)
                / self.rotary_dim
            )
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def _match_cos_sin_cache_dtype(self, query: torch.Tensor) -> torch.Tensor:
        cos_sin_cache = self.cos_sin_cache
        if (
            cos_sin_cache.device == query.device
            and cos_sin_cache.dtype == query.dtype
        ):
            return cos_sin_cache
        cos_sin_cache = cos_sin_cache.to(query.device, dtype=query.dtype)
        self.cos_sin_cache = cos_sin_cache
        return cos_sin_cache

    def get_cos_sin(self, seqlen: int) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[:seqlen]
        cos, sin = cos_sin.chunk(2, dim=-1)
        return cos, sin


class RotaryEmbedding(RotaryEmbeddingBase):
    """Rotary positional embedding — aligns with vLLM RotaryEmbedding."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        init_cache: bool = True,
    ) -> None:
        super().__init__(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype,
            init_cache=init_cache,
        )

    @staticmethod
    def forward_static(
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None,
        head_size: int,
        rotary_dim: int,
        cos_sin_cache: torch.Tensor,
        is_neox_style: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """A PyTorch-native implementation of forward()."""
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, head_size)
        query_rot = query[..., :rotary_dim]
        query_pass = query[..., rotary_dim:]
        query_rot = ApplyRotaryEmb.forward_static(
            query_rot, cos, sin, is_neox_style
        )
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        if key is not None:
            key_shape = key.shape
            key = key.view(num_tokens, -1, head_size)
            key_rot = key[..., :rotary_dim]
            key_pass = key[..., rotary_dim:]
            key_rot = ApplyRotaryEmb.forward_static(
                key_rot, cos, sin, is_neox_style
            )
            key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        cos_sin_cache = self._match_cos_sin_cache_dtype(query)
        return self.forward_static(
            positions,
            query,
            key,
            self.head_size,
            self.rotary_dim,
            cos_sin_cache,
            self.is_neox_style,
        )

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """CUDA dispatch — in-place fused Q/K rotation via Triton.

        Mirrors ``vllm/model_executor/layers/rotary_embedding/base.py:
        RotaryEmbeddingBase.forward_cuda``.  We dispatch through
        ``my_vllm._custom_ops.rotary_embedding`` which routes to the
        Triton kernel on CUDA and falls back to the PyTorch path on
        CPU.  In-place: query and key are modified directly.
        """
        from my_vllm._custom_ops import rotary_embedding

        cos_sin_cache = self._match_cos_sin_cache_dtype(query)
        rotary_embedding(
            positions,
            query,
            key,
            self.head_size,
            cos_sin_cache,
            self.is_neox_style,
        )
        return query, key

    def extra_repr(self) -> str:
        s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"
        s += f", max_position_embeddings={self.max_position_embeddings}"
        s += f", base={self.base}, is_neox_style={self.is_neox_style}"
        return s
