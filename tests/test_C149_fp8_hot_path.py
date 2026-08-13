"""C149: FP8 hot path — kv_cache_dtype plumbing is end-to-end.

Verifies the FP8 KV cache wiring is no longer a stub:

  1. `CacheConfig.kv_cache_dtype` exists and defaults to "auto".
  2. `is_quantized_kv_cache(...)` returns True for FP8 strings.
  3. `Attention.kv_cache_dtype` follows `cache_config.kv_cache_dtype`.
  4. `TritonAttentionImpl.set_kv_cache_dtype(...)` /
     `set_kv_scales(...)` propagate to the impl and are used
     by `forward()` (we verify via attribute inspection — the
     full reshape_and_cache call is exercised by
     `tests/test_C023_paged_attn.py`).
  5. End-to-end mock: with `kv_cache_dtype="fp8_e4m3"`,
     `_install_fp8_kv_scales` is called and each attention
     layer's impl carries the right dtype + scales.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from my_vllm.config.cache import CacheConfig
from my_vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    is_quantized_kv_cache,
)
from my_vllm.v1.attention.backends.triton_attn import TritonAttentionImpl


def test_cache_config_has_kv_cache_dtype_default_auto() -> None:
    """`CacheConfig.kv_cache_dtype` defaults to "auto"."""
    cfg = CacheConfig()
    assert cfg.kv_cache_dtype == "auto"

    cfg_fp8 = CacheConfig(kv_cache_dtype="fp8_e4m3")
    assert cfg_fp8.kv_cache_dtype == "fp8_e4m3"


def test_is_quantized_kv_cache_detects_fp8_strings() -> None:
    """`is_quantized_kv_cache` returns True for FP8 dtype strings.

    Previously a stub that always returned False — the FP8
    branch in `reshape_and_cache_kernel_flash` could never
    fire.
    """
    assert is_quantized_kv_cache("auto") is False
    assert is_quantized_kv_cache("bfloat16") is False
    assert is_quantized_kv_cache("float16") is False
    assert is_quantized_kv_cache("fp8") is True
    assert is_quantized_kv_cache("fp8_e4m3") is True
    assert is_quantized_kv_cache("fp8_e5m2") is True
    assert is_quantized_kv_cache("int8") is True


def test_triton_impl_default_dtype_is_auto() -> None:
    """Impl defaults preserve pre-C149 behaviour (dtype=auto)."""
    impl = TritonAttentionImpl(num_heads=2, head_size=4, scale=0.5)
    assert impl.kv_cache_dtype == "auto"
    assert impl._k_scale is None
    assert impl._v_scale is None


def test_triton_impl_set_kv_cache_dtype() -> None:
    """`set_kv_cache_dtype(...)` updates the impl's dtype."""
    impl = TritonAttentionImpl(num_heads=2, head_size=4, scale=0.5)
    impl.set_kv_cache_dtype("fp8_e4m3")
    assert impl.kv_cache_dtype == "fp8_e4m3"


def test_triton_impl_set_kv_scales() -> None:
    """`set_kv_scales(...)` stores K / V scales on the impl."""
    impl = TritonAttentionImpl(num_heads=2, head_size=4, scale=0.5)
    k = torch.tensor([0.1], dtype=torch.float32)
    v = torch.tensor([0.2], dtype=torch.float32)
    impl.set_kv_scales(k, v)
    assert impl._k_scale is k
    assert impl._v_scale is v


def test_attention_resolves_kv_cache_dtype_from_cache_config() -> None:
    """`Attention` reads `kv_cache_dtype` from `cache_config`.

    Uses a mock Attention impl (returned via
    `get_attn_backend().get_impl_cls()`) so the test doesn't
    need a CUDA device — the attention layer itself does the
    dtype plumbing before any forward call.
    """
    from my_vllm.model_executor.layers.attention.attention import Attention

    mock_impl_cls = MagicMock()
    mock_impl = MagicMock()
    mock_impl_cls.return_value = mock_impl

    # Patch the backend resolution so it returns our mock.
    import my_vllm.model_executor.layers.attention.attention as attn_mod

    original_get_attn_backend = attn_mod.Attention.get_attn_backend
    attn_mod.Attention.get_attn_backend = lambda self: MagicMock(
        get_impl_cls=lambda: mock_impl_cls,
    )
    try:
        layer = Attention(
            num_heads=2,
            head_size=4,
            scale=0.5,
            cache_config=CacheConfig(kv_cache_dtype="fp8_e4m3"),
        )
    finally:
        attn_mod.Attention.get_attn_backend = original_get_attn_backend

    assert layer.kv_cache_dtype == "fp8_e4m3"
    # Impl's dtype was set via set_kv_cache_dtype.
    mock_impl.set_kv_cache_dtype.assert_called_once_with("fp8_e4m3")


def test_attention_resolve_kv_cache_torch_dtype() -> None:
    """`_resolve_kv_cache_torch_dtype` maps FP8 strings to FP8 dtype."""
    from my_vllm.model_executor.layers.attention.attention import Attention
    from my_vllm.platforms import current_platform

    # FP8 strings → platform's FP8 dtype.
    assert Attention._resolve_kv_cache_torch_dtype("fp8") == current_platform.fp8_dtype()
    assert Attention._resolve_kv_cache_torch_dtype("fp8_e4m3") == current_platform.fp8_dtype()
    assert Attention._resolve_kv_cache_torch_dtype("fp8_e5m2") == current_platform.fp8_dtype()
    # Non-FP8 → default dtype (FP32 for our env).
    assert Attention._resolve_kv_cache_torch_dtype("auto") == torch.get_default_dtype()
    assert Attention._resolve_kv_cache_torch_dtype("bfloat16") == torch.get_default_dtype()


def test_attention_set_kv_scales_propagates_to_impl() -> None:
    """`Attention.set_kv_scales` forwards to the impl when supported."""
    from my_vllm.model_executor.layers.attention.attention import Attention

    mock_impl_cls = MagicMock()
    mock_impl = MagicMock()
    mock_impl_cls.return_value = mock_impl

    import my_vllm.model_executor.layers.attention.attention as attn_mod

    original_get_attn_backend = attn_mod.Attention.get_attn_backend
    attn_mod.Attention.get_attn_backend = lambda self: MagicMock(
        get_impl_cls=lambda: mock_impl_cls,
    )
    try:
        layer = Attention(
            num_heads=2,
            head_size=4,
            scale=0.5,
            cache_config=CacheConfig(kv_cache_dtype="fp8_e4m3"),
        )
    finally:
        attn_mod.Attention.get_attn_backend = original_get_attn_backend

    k_scale = torch.tensor([0.1], dtype=torch.float32)
    v_scale = torch.tensor([0.2], dtype=torch.float32)
    layer.set_kv_scales(k_scale, v_scale)
    assert layer._k_scale is k_scale
    assert layer._v_scale is v_scale
    mock_impl.set_kv_scales.assert_called_once_with(k_scale, v_scale)


def test_load_model_with_fp8_kv_calls_install_scales(monkeypatch) -> None:
    """`load_model` calls `_install_fp8_kv_scales` when FP8 is configured.

    Uses a fully-mocked GPUModelRunner to avoid loading the
    real OPT-125m checkpoint.
    """
    from my_vllm.v1.worker.gpu_model_runner import GPUModelRunner

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.vllm_config = MagicMock()
    runner.vllm_config.cache_config = CacheConfig(kv_cache_dtype="fp8_e4m3")
    runner.vllm_config.lora_config = None
    runner.device = torch.device("cpu")
    runner.dtype = torch.bfloat16
    runner.dtype_str = "bfloat16"
    runner.model_name = "facebook/opt-125m"
    runner.num_layers = 1
    runner.max_num_blocks = 1
    runner.num_kv_heads = 1
    runner.block_size = 4
    runner.head_dim = 4
    runner.enforce_eager = True
    runner.lora_config = None
    runner._max_num_reqs = 1
    runner._max_num_batched_tokens = 1
    runner._max_model_len = 4
    runner._active_lora_int_id = None

    # Mock the model + KV binding + LoRA path so load_model
    # doesn't try to touch a real OPT.
    monkeypatch.setattr(
        "my_vllm.model_executor.model_loader.load_model",
        lambda *a, **kw: MagicMock(
            parameters=lambda: iter([torch.zeros(1)]),
            config=MagicMock(num_hidden_layers=1, num_attention_heads=1, hidden_size=4),
            model=MagicMock(decoder=MagicMock(layers=[])),
        ),
    )

    install_called = MagicMock()
    monkeypatch.setattr(runner, "_install_fp8_kv_scales", install_called)

    runner._bind_kv_caches = MagicMock()  # avoid touching .model
    runner.capture_cudagraph = MagicMock()  # enforce_eager=True so not called anyway
    runner.model = MagicMock()  # init_lora path skips when lora_config=None

    runner.load_model()

    install_called.assert_called_once()


def test_install_fp8_kv_scales_calls_set_kv_scales_per_layer() -> None:
    """`_install_fp8_kv_scales` invokes `attn.set_kv_scales` per layer."""
    from my_vllm.v1.worker.gpu_model_runner import GPUModelRunner

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.device = torch.device("cpu")

    # Fake model with two decoder layers, each with a mock attn.
    layers = []
    for _ in range(2):
        attn = MagicMock()
        attn.set_kv_scales = MagicMock()
        layers.append(MagicMock(self_attn=MagicMock(attn=attn)))
    runner.model = MagicMock(model=MagicMock(decoder=MagicMock(layers=layers)))

    runner._install_fp8_kv_scales(default_scale=0.1)

    for layer in layers:
        attn = layer.self_attn.attn
        attn.set_kv_scales.assert_called_once()
        k, v = attn.set_kv_scales.call_args.args
        assert torch.equal(k.cpu(), torch.tensor([0.1]))
        assert torch.equal(v.cpu(), torch.tensor([0.1]))