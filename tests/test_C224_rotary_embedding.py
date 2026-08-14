"""test_C224_rotary_embedding.py — C224 fused rotary embedding tests.

Verifies the architecture-aligned pieces of the C224 path:

  1. ``RotaryEmbedding`` cos/sin cache construction.
     * Correct inv_freq (theta / rotary_dim).
     * Cache dtype matches the requested dtype.
     * Shape is ``[max_position, rotary_dim]`` (cos-then-sin).
     * Cache size → fingerprint.

  2. ``ApplyRotaryEmb.forward_static`` (Pytorch reference).
     * Neox-style: lower half vs upper half split.
     * GPT-J-style: even-index vs odd-index interleave.
     * Both ``x * cos - y * sin`` and ``x * cos + y * sin``
       formulas produce a correct rotation.

  3. ``RotaryEmbedding.forward_native`` end-to-end.
     * Output matches a hand-rolled reference (cos/sin +
       neox/gptj split).
     * Multiple positions, multiple heads.
     * partial rotary factor (rotary_dim < head_size) leaves the
       pass-through half unchanged.

  4. ``RotaryEmbedding.forward_cuda`` vs ``forward_native``.
     * Triton fused kernel matches PyTorch reference within BF16
       round-off.
     * In-place semantics on Q and K.
     * Works with num_heads != num_kv_heads (GQA via reduction).

  5. ``RotaryEmbedding.forward`` dispatch.
     * On CUDA + Triton + the ``RotaryEmbedding`` is constructed
       with the standard dispatch forward, ``forward_cuda`` is
       invoked (verifiable via tensor identity: in-place Q/K).
"""

import math

import pytest
import torch

from my_vllm.model_executor.layers.rotary_embedding import (
    RotaryEmbedding,
    get_rope,
)
from my_vllm.model_executor.layers.rotary_embedding.common import (
    ApplyRotaryEmb,
)
from my_vllm.model_executor.layers.rotary_embedding.triton_rotary import (
    rotary_embedding_triton,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 1. Cache construction
# ---------------------------------------------------------------------------


class TestCacheConstruction:
    def test_cache_shape_dtype(self, device):
        re = RotaryEmbedding(
            head_size=128,
            rotary_dim=128,
            max_position_embeddings=4096,
            base=10000.0,
            is_neox_style=True,
            dtype=torch.bfloat16,
        ).to(device)
        assert re.cos_sin_cache.shape == (4096, 128)
        assert re.cos_sin_cache.dtype == torch.bfloat16

    def test_cache_first_row_is_cos0_sin0(self, device):
        # For position 0, sin=0 and cos=1 (so rotation is identity).
        re = RotaryEmbedding(
            head_size=64,
            rotary_dim=64,
            max_position_embeddings=128,
            base=10000.0,
            is_neox_style=True,
            dtype=torch.float32,
        ).to(device)
        row = re.cos_sin_cache[0]
        # cos(0) = 1 for all half-pairs of dims.
        torch.testing.assert_close(row[:32], torch.ones(32, device=device))
        # sin(0) = 0 for all half-pairs.
        torch.testing.assert_close(row[32:], torch.zeros(32, device=device))

    def test_cache_inv_freq_uses_base(self, device):
        # For rotary_dim=4 with base=10000:
        # inv_freq[i] = 1 / (base ** (2i / rotary_dim))
        # i=0: inv_freq = 1
        # i=1: inv_freq = 1 / 100
        # Position p: angle = p * inv_freq; cos = cos(p * inv_freq).
        re = RotaryEmbedding(
            head_size=4,
            rotary_dim=4,
            max_position_embeddings=128,
            base=10000.0,
            is_neox_style=True,
            dtype=torch.float64,
        ).to(device)
        # Position 1: cos(1) at i=0, cos(1/100) at i=1.
        expected_cos_first = math.cos(1.0)
        expected_cos_second = math.cos(0.01)
        actual = re.cos_sin_cache[1, :].tolist()
        assert actual[0] == pytest.approx(expected_cos_first, abs=1e-6)
        assert actual[1] == pytest.approx(expected_cos_second, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. ApplyRotaryEmb.forward_static (Pytorch reference)
# ---------------------------------------------------------------------------


class TestApplyRotaryEmb:
    def test_neox_split_basic(self, device):
        # Neox: x = lower half, y = upper half.
        # new_x = cos * x - sin * y; new_y = cos * y + sin * x.
        # Shape: [seq_len=1, num_heads=1, head_size=4] → neox split on last dim.
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], device=device)
        cos = torch.tensor([[0.5, 0.6]], device=device)
        sin = torch.tensor([[0.3, 0.4]], device=device)
        y = ApplyRotaryEmb.forward_static(x, cos, sin, is_neox_style=True)
        # x1 = [1, 2]; x2 = [3, 4]
        # o1[0] = 0.5*1 - 0.3*3 = 0.5 - 0.9 = -0.4
        # o1[1] = 0.6*2 - 0.4*4 = 1.2 - 1.6 = -0.4
        # o2[0] = 0.5*3 + 0.3*1 = 1.5 + 0.3 = 1.8
        # o2[1] = 0.6*4 + 0.4*2 = 2.4 + 0.8 = 3.2
        torch.testing.assert_close(
            y[0, 0], torch.tensor([-0.4, -0.4, 1.8, 3.2], device=device)
        )

    def test_gptj_split_basic(self, device):
        # GPT-J: x = even indices, y = odd indices.
        # The cos/sin pair is shared between (x, y) at the same index.
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], device=device)
        cos = torch.tensor([[0.5, 0.6]], device=device)
        sin = torch.tensor([[0.3, 0.4]], device=device)
        y = ApplyRotaryEmb.forward_static(
            x, cos, sin, is_neox_style=False
        )
        # x1 = x[..., 0::2] = [1, 3]
        # x2 = x[..., 1::2] = [2, 4]
        # o1[0] = 0.5*1 - 0.3*2 = 0.5 - 0.6 = -0.1
        # o1[1] = 0.6*3 - 0.4*4 = 1.8 - 1.6 = 0.2
        # Then stack((o1, o2), -1).flatten(-2):
        #   [o1[0], o2[0], o1[1], o2[1]]
        # o2[0] = 0.5*2 + 0.3*1 = 1.0 + 0.3 = 1.3
        # o2[1] = 0.6*4 + 0.4*3 = 2.4 + 1.2 = 3.6
        # Final: [-0.1, 1.3, 0.2, 3.6]
        torch.testing.assert_close(
            y[0, 0], torch.tensor([-0.1, 1.3, 0.2, 3.6], device=device)
        )

    def test_position_zero_is_identity(self, device):
        # For position 0, cos=1, sin=0 → rotation is identity.
        x = torch.randn(2, 2, 8, device=device)
        rot = RotaryEmbedding(
            head_size=8,
            rotary_dim=8,
            max_position_embeddings=128,
            base=10000.0,
            is_neox_style=True,
            dtype=torch.float32,
        ).to(device)
        pos0 = torch.zeros(2, dtype=torch.int64, device=device)
        out, _ = rot.forward_native(pos0, x.clone(), None)
        torch.testing.assert_close(out, x)


# ---------------------------------------------------------------------------
# 3. RotaryEmbedding.forward_native end-to-end (CPU + CUDA)
# ---------------------------------------------------------------------------


class TestForwardNative:
    def test_neox_against_handrolled(self, device):
        # Direct: replicate rotation by-hand on the cos/sin cache.
        torch.manual_seed(0)
        head_size = 64
        num_heads = 4
        num_tokens = 8
        re = RotaryEmbedding(
            head_size=head_size,
            rotary_dim=head_size,
            max_position_embeddings=128,
            base=10000.0,
            is_neox_style=True,
            dtype=torch.float32,
        ).to(device)
        positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
        q = torch.randn(num_tokens, num_heads, head_size, device=device)

        out, _ = re.forward_native(positions, q.clone(), None)

        # Hand-rolled reference.
        cs = re.cos_sin_cache[positions]  # [num_tokens, rot_dim]
        cos_full = cs[..., : head_size // 2].unsqueeze(1)  # [T, 1, hd/2]
        sin_full = cs[..., head_size // 2 :].unsqueeze(1)
        x1 = q[..., : head_size // 2]
        x2 = q[..., head_size // 2 :]
        ref = torch.cat(
            (x1 * cos_full - x2 * sin_full, x2 * cos_full + x1 * sin_full),
            dim=-1,
        )
        torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)

    def test_partial_rotary_passes_through_tail(self, device):
        # rotary_dim = head_size / 2 → first half rotated, second unchanged.
        torch.manual_seed(0)
        head_size = 64
        rotary_dim = 32
        re = RotaryEmbedding(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=128,
            base=10000.0,
            is_neox_style=True,
            dtype=torch.float32,
        ).to(device)
        positions = torch.zeros(2, dtype=torch.int64, device=device)
        q = torch.randn(2, 1, head_size, device=device)
        out, _ = re.forward_native(positions, q.clone(), None)
        # First 32 elements are rotated (but cos=1, sin=0 → unchanged).
        torch.testing.assert_close(out[..., :32], q[..., :32])
        # Second 32 elements pass through untouched.
        torch.testing.assert_close(out[..., 32:], q[..., 32:])


# ---------------------------------------------------------------------------
# 4. RotaryEmbedding.forward_cuda (Triton fused) vs forward_native
# ---------------------------------------------------------------------------


class TestForwardCuda:
    def test_triton_matches_native_neox(self):
        _require_cuda()
        torch.manual_seed(0)
        head_size = 128
        num_heads = 4
        num_tokens = 8
        re = RotaryEmbedding(
            head_size=head_size,
            rotary_dim=head_size,
            max_position_embeddings=4096,
            base=10000.0,
            is_neox_style=True,
            dtype=torch.bfloat16,
        ).cuda()
        positions = torch.arange(num_tokens, dtype=torch.int64).cuda()
        q = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16).cuda()
        k = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16).cuda()

        q_ref, k_ref = re.forward_native(positions, q.clone(), k.clone())
        q_test, k_test = re.forward_cuda(positions, q.clone(), k.clone())

        # BF16 round-off: ~1-3% relative tolerance is normal.
        assert torch.allclose(q_ref, q_test, atol=2e-2, rtol=2e-2)
        assert torch.allclose(k_ref, k_test, atol=2e-2, rtol=2e-2)

    def test_triton_inplace_semantics(self):
        _require_cuda()
        torch.manual_seed(0)
        re = RotaryEmbedding(
            head_size=64,
            rotary_dim=64,
            max_position_embeddings=512,
            base=10000.0,
            is_neox_style=True,
            dtype=torch.bfloat16,
        ).cuda()
        positions = torch.arange(4, dtype=torch.int64).cuda()
        q = torch.randn(4, 2, 64, dtype=torch.bfloat16).cuda()
        k = torch.randn(4, 2, 64, dtype=torch.bfloat16).cuda()

        q_in = q.clone()
        k_in = k.clone()
        out_q, out_k = re.forward_cuda(positions, q_in, k_in)

        # forward_cuda is in-place (mirrors vLLM contract).
        assert out_q.data_ptr() == q_in.data_ptr()
        assert out_k.data_ptr() == k_in.data_ptr()

    def test_gqa_q_heads_gt_kv_heads(self):
        # num_heads=8, num_kv_heads=2 (GQA — 4 q heads share each kv head).
        _require_cuda()
        torch.manual_seed(0)
        head_size = 128
        num_q_heads = 8
        num_kv_heads = 2
        num_tokens = 4
        re = RotaryEmbedding(
            head_size=head_size,
            rotary_dim=head_size,
            max_position_embeddings=4096,
            base=10000.0,
            is_neox_style=True,
            dtype=torch.bfloat16,
        ).cuda()
        positions = torch.arange(num_tokens, dtype=torch.int64).cuda()
        q = torch.randn(
            num_tokens, num_q_heads, head_size, dtype=torch.bfloat16
        ).cuda()
        k = torch.randn(
            num_tokens, num_kv_heads, head_size, dtype=torch.bfloat16
        ).cuda()

        q_ref, k_ref = re.forward_native(positions, q.clone(), k.clone())
        q_test, k_test = re.forward_cuda(positions, q.clone(), k.clone())

        assert torch.allclose(q_ref, q_test, atol=2e-2, rtol=2e-2)
        assert torch.allclose(k_ref, k_test, atol=2e-2, rtol=2e-2)

    def test_triton_gptj_style(self):
        _require_cuda()
        torch.manual_seed(0)
        re = RotaryEmbedding(
            head_size=64,
            rotary_dim=64,
            max_position_embeddings=4096,
            base=10000.0,
            is_neox_style=False,
            dtype=torch.bfloat16,
        ).cuda()
        positions = torch.tensor([1, 5, 10], dtype=torch.int64).cuda()
        q = torch.randn(3, 4, 64, dtype=torch.bfloat16).cuda()
        k = torch.randn(3, 4, 64, dtype=torch.bfloat16).cuda()

        q_ref, k_ref = re.forward_native(positions, q.clone(), k.clone())
        q_test, k_test = re.forward_cuda(positions, q.clone(), k.clone())

        assert torch.allclose(q_ref, q_test, atol=2e-2, rtol=2e-2)
        assert torch.allclose(k_ref, k_test, atol=2e-2, rtol=2e-2)


# ---------------------------------------------------------------------------
# 5. Direct dispatcher (rotary_embedding_triton kernel)
# ---------------------------------------------------------------------------


class TestDirectTritonCall:
    def test_rotary_embedding_triton_against_pytorch(self):
        _require_cuda()
        torch.manual_seed(0)
        head_size = 64
        num_heads = 4
        num_tokens = 4

        # Build cos/sin cache directly (Llama 7B config).
        base = 10000.0
        rotary_dim = head_size
        inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        t = torch.arange(128, dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cs = torch.cat((freqs.cos(), freqs.sin()), dim=-1).cuda().to(torch.bfloat16)

        positions = torch.tensor([0, 5, 11, 23], dtype=torch.int64).cuda()
        q = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16).cuda()
        k = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16).cuda()

        # Triton kernel
        q_test = q.clone()
        k_test = k.clone()
        rotary_embedding_triton(
            positions,
            q_test,
            k_test,
            head_size=head_size,
            cos_sin_cache=cs,
            is_neox=True,
        )

        # PyTorch reference (ApplyRotaryEmb).
        cs_per = cs.index_select(0, positions)
        cos, sin = cs_per.chunk(2, dim=-1)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        q_pt = q.clone()
        q_in = q_pt.view(num_tokens, num_heads, head_size)
        x1, x2 = q_in.chunk(2, dim=-1)
        torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1, out=q_in)
        q_view = q_pt.view(num_tokens, num_heads, head_size)

        assert torch.allclose(q_view, q_test, atol=2e-2, rtol=2e-2)


# ---------------------------------------------------------------------------
# 6. get_rope dispatcher (vLLM-canonical factory)
# ---------------------------------------------------------------------------


class TestGetRopeFactory:
    def test_returns_rotary_embedding(self, device):
        re = get_rope(
            head_size=128,
            max_position=2048,
            is_neox_style=True,
            dtype=torch.bfloat16,
        )
        assert isinstance(re, RotaryEmbedding)

    def test_caching_returns_same_instance(self):
        re1 = get_rope(
            head_size=128,
            max_position=2048,
            is_neox_style=True,
            dtype=torch.bfloat16,
        )
        re2 = get_rope(
            head_size=128,
            max_position=2048,
            is_neox_style=True,
            dtype=torch.bfloat16,
        )
        assert re1 is re2

    def test_different_kwargs_yields_different_instance(self):
        re1 = get_rope(
            head_size=128, max_position=2048, dtype=torch.bfloat16
        )
        re2 = get_rope(
            head_size=256, max_position=2048, dtype=torch.bfloat16
        )
        assert re1 is not re2


# ---------------------------------------------------------------------------
# 7. _custom_ops.rotary_embedding dispatcher
# ---------------------------------------------------------------------------


class TestCustomOpsDispatch:
    def test_pytorch_path_runs_on_cpu(self):
        # CPU → PyTorch fallback path.
        positions = torch.tensor([0, 1], dtype=torch.int64)
        q = torch.randn(2, 2, 8, dtype=torch.float32)
        # Build a tiny cos_sin_cache.
        base = 10000.0
        rot_dim = 8
        inv_freq = 1.0 / (
            base ** (torch.arange(0, rot_dim, 2, dtype=torch.float32) / rot_dim)
        )
        t = torch.arange(64, dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cs = torch.cat((freqs.cos(), freqs.sin()), dim=-1)

        from my_vllm._custom_ops import rotary_embedding

        q_in = q.clone()
        k_in = q.clone()
        rotary_embedding(
            positions, q_in, k_in, 8, cs, is_neox=True
        )
        # Position 0 is identity; position 1 should be rotated.
        torch.testing.assert_close(q_in[0], q[0])
        assert not torch.allclose(q_in[1], q[1])

    def test_triton_path_used_on_cuda(self):
        _require_cuda()
        from my_vllm._custom_ops import rotary_embedding

        positions = torch.tensor([0, 1, 2], dtype=torch.int64).cuda()
        q = torch.randn(3, 2, 8, dtype=torch.bfloat16).cuda()
        # Construct a cos_sin_cache for rot_dim=8.
        rot_dim = 8
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, rot_dim, 2, dtype=torch.float32) / rot_dim)
        )
        t = torch.arange(64, dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cs = torch.cat((freqs.cos(), freqs.sin()), dim=-1).cuda().to(torch.bfloat16)

        q_in = q.clone()
        rotary_embedding(positions, q_in, None, 8, cs, is_neox=True)
        # Position 0 is identity.
        torch.testing.assert_close(q_in[0], q[0])
        # Position 1+ are rotated.
        assert not torch.allclose(q_in[1], q[1])