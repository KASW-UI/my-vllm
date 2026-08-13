"""C147: update_lora_modules actually wraps Linear targets on OPTForCausalLM.

Mirrors vllm/lora/model_manager.py:LoRAModelManager._create_lora_modules
(minimal subset) — the wrap-from-base-class step that turns each
`ColumnParallelLinear` / `RowParallelLinear` declared in
`lora_target_modules` into the matching `*LoRA` wrapper, then
registers it in `LoRAModel.lora_modules`.

Test scope:
  1. OPTForCausalLM declares SupportsLoRA-style classvars.
  2. After `init_lora` (which calls `update_lora_modules`),
     every `qkv_proj` / `out_proj` / `fc1` / `fc2` on the host
     model is replaced with the matching LoRA wrapper.
  3. The wrappers are registered in `LoRAModel.lora_modules`
     under their fully-qualified module names.
  4. `set_lora(id)` propagates to every wrapped module.
"""

from __future__ import annotations

import torch
from transformers import OPTConfig

from my_vllm.lora.layers import (
    ColumnParallelLinearLoRA,
    RowParallelLinearLoRA,
)
from my_vllm.lora.layers.base_linear import BaseLinearLayer
from my_vllm.lora.request import LoRARequest
from my_vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from my_vllm.model_executor.models.opt import OPTForCausalLM
from my_vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin


def _tiny_opt_config() -> OPTConfig:
    """Construct a tiny OPTConfig — small dims keep the test fast."""
    return OPTConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        # OPT uses `ffn_dim`; map intermediate_size → ffn_dim.
        ffn_dim=64,
        max_position_embeddings=32,
        # Keep embedding-proj identity so project_in/out are skipped.
        word_embed_proj_dim=32,
        # Tie weights so `lm_head` is shared (avoids extra allocation).
        tie_word_embeddings=True,
        # No dropout.
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        layerdrop=0.0,
        # Disable bias to keep parameter count low.
        enable_bias=False,
        layer_norm_elementwise_affine=True,
        do_layer_norm_before=True,
        # OPT specifics.
        num_labels=1,
        init_std=0.02,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
    )


class _FakeRunner(LoRAModelRunnerMixin):
    """Minimal host for LoRAModelRunnerMixin — only needs the mixin's
    attribute contract (init_lora populates self.lora_model / manager /
    worker_manager)."""


def _build_wrapped() -> tuple[OPTForCausalLM, _FakeRunner]:
    cfg = _tiny_opt_config()
    model = OPTForCausalLM(cfg)
    runner = _FakeRunner()
    runner.init_lora(
        model=model,
        max_loras=2,
        max_lora_rank=4,
        device=torch.device("cpu"),
    )
    return model, runner


def test_supports_lora_metadata_declared() -> None:
    """OPTForCausalLM declares supports_lora + lora_target_modules."""
    cfg = _tiny_opt_config()
    model = OPTForCausalLM(cfg)
    assert model.supports_lora is True
    assert "embed_tokens" in model.embedding_modules
    assert {"qkv_proj", "out_proj", "fc1", "fc2"} <= model.lora_target_modules


def test_update_lora_modules_wraps_all_targets() -> None:
    """After init_lora, every target Linear is replaced with a *LoRA wrapper."""
    model, runner = _build_wrapped()

    # Walk the wrapped tree and count wrappers.
    n_col = 0
    n_row = 0
    for name, mod in model.named_modules():
        last = name.rsplit(".", 1)[-1]
        if last in {"qkv_proj", "fc1"}:
            assert isinstance(mod, ColumnParallelLinearLoRA), (
                f"{name} expected ColumnParallelLinearLoRA, got {type(mod).__name__}"
            )
            n_col += 1
        elif last in {"out_proj", "fc2"}:
            assert isinstance(mod, RowParallelLinearLoRA), (
                f"{name} expected RowParallelLinearLoRA, got {type(mod).__name__}"
            )
            n_row += 1

    # 2 layers × {qkv_proj, fc1} → 4 ColumnParallelLinearLoRA
    # 2 layers × {out_proj, fc2} → 4 RowParallelLinearLoRA
    assert n_col == 4
    assert n_row == 4


def test_lora_modules_registry_populated() -> None:
    """LoRAModel.lora_modules has one entry per wrapped layer."""
    model, runner = _build_wrapped()
    lora_modules = runner.lora_model.lora_modules
    # 2 layers × 4 target suffixes = 8 wrapped layers.
    assert len(lora_modules) == 8

    # All keys must be fully-qualified module names on the host.
    for name, wrapper in lora_modules.items():
        assert name.startswith("model.decoder.layers."), name
        assert isinstance(wrapper, BaseLinearLayer), name
        # The wrapper's `base_layer` must be the original Linear.
        assert isinstance(
            wrapper.base_layer,
            (QKVParallelLinear, ColumnParallelLinear, RowParallelLinear),
        ), name
        # And it must be reachable from the host model via the same path.
        resolved = model.get_submodule(name)
        assert resolved is wrapper, name


def test_set_lora_propagates_to_all_wrappers() -> None:
    """set_lora(id) marks every wrapper as active."""
    model, runner = _build_wrapped()
    # Register + activate an adapter.
    runner.add_lora(LoRARequest(lora_name="t", lora_int_id=1, lora_path=""))
    runner.set_lora(1)

    active_ids = {
        w.active_lora_int_id
        for w in runner.lora_model.lora_modules.values()
    }
    assert active_ids == {1}, active_ids


def test_forward_unchanged_when_no_lora_active() -> None:
    """Wrapping must not change the model's Linear parameters.

    The wrappers hold the base Linear as a child submodule, so
    `state_dict()` now nests under `.base_layer.`. Verify that,
    after stripping `.base_layer.`, the wrapped state's values
    for the wrapped Linear layers (`qkv_proj`/`out_proj`/`fc1`/
    `fc2`) match the original Linear's values exactly.

    Construction:
      1. Seed torch so the random init is deterministic.
      2. Build the un-wrapped `ref` first and snapshot its
         state_dict.
      3. Build `model` (gets different random init) and load
         ref's state_dict into it (using the un-wrapped key
         layout).
      4. Wrap `model` via `init_lora` + `update_lora_modules`.
      5. Verify that, after stripping `.base_layer.`, the
         wrapped state's Linear parameters equal the snapshot.

    Full forward equivalence is out of scope here: my-vllm's
    Attention layer requires `set_forward_context` to be active
    (kv-cache slot binding), which needs the full GPUModelRunner
    setup (covered by `tests/test_C010_model_runner.py`).
    """
    torch.manual_seed(1234)
    cfg_ref = _tiny_opt_config()
    ref = OPTForCausalLM(cfg_ref)
    ref_state_snapshot = {k: v.detach().clone() for k, v in ref.state_dict().items()}

    # New seed → different random init, then load_state_dict
    # overwrites with ref's values.
    torch.manual_seed(5678)
    model = OPTForCausalLM(_tiny_opt_config())
    model.load_state_dict(ref_state_snapshot)
    runner = _FakeRunner()
    runner.init_lora(
        model=model,
        max_loras=2,
        max_lora_rank=4,
        device=torch.device("cpu"),
    )

    wrapped_state = _copyable_state_dict(model)
    # Look at the raw state_dict (with `.base_layer.` keys) so we
    # can confirm the wrapped Linear layers actually exist.
    raw_wrapped_state = dict(model.state_dict())
    wrapped_keys = {
        k for k in raw_wrapped_state
        if ".base_layer." in k
    }
    assert wrapped_keys, "expected wrapped Linear keys under .base_layer."
    for k in wrapped_keys:
        unwrapped_key = k.replace(".base_layer.", ".")
        assert unwrapped_key in ref_state_snapshot, k
        # `equal_nan=True` handles NaN-init values from random
        # init that bit-compare unequal under `torch.equal`.
        assert torch.allclose(
            wrapped_state[unwrapped_key],
            ref_state_snapshot[unwrapped_key],
            equal_nan=True,
        ), f"Parameter {k} differs after wrapping"


def _copyable_state_dict(src: OPTForCausalLM) -> dict[str, torch.Tensor]:
    """Snapshot a wrapped model's state_dict under the un-wrapped keys.

    After wrapping, `state_dict()` returns nested keys like
    `model.decoder.layers.0.self_attn.qkv_proj.base_layer.weight`.
    To compare against the un-wrapped reference, strip the
    `.base_layer.` segment so the keys match the unwrapped
    module's parameter paths.
    """
    out: dict[str, torch.Tensor] = {}
    for k, v in src.state_dict().items():
        # `.base_layer.` appears at most once (it's the wrapper's
        # only child module path). Strip it to recover the
        # pre-wrap key.
        unwrapped_key = k.replace(".base_layer.", ".")
        out[unwrapped_key] = v.detach().clone()
    return out