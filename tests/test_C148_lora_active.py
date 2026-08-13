"""C148: runner mixin → execute_model — per-step LoRA activation.

Verifies the wiring that turns the C146 LoRA skeleton into a
real per-step active-adapter swap:

  1. `GPUModelRunner` inherits from `LoRAModelRunnerMixin`
     (mixin is in the MRO).
  2. `NewRequestData.lora_int_id` flows through
     `SchedulerOutput` (set by the scheduler from
     `Request.lora_request.lora_int_id`).
  3. `RequestState.req_id_to_lora_id` records the
     per-request mapping populated by `_update_states`.
  4. `_set_active_lora(scheduler_output)` selects the first
     non-None `lora_int_id` across new-then-cached reqs
     (single-active-adapter path) and calls
     `self.set_lora(id)`.
  5. The active-adapter cache (`_active_lora_int_id`)
     suppresses redundant `set_lora` calls step-to-step.
  6. LoRA-disabled runner is a no-op (no `lora_worker_manager`
     attribute).
"""

from __future__ import annotations

import torch
from transformers import OPTConfig

from my_vllm.lora.layers import (
    ColumnParallelLinearLoRA,
    RowParallelLinearLoRA,
)
from my_vllm.lora.request import LoRARequest
from my_vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from my_vllm.v1.worker.gpu_model_runner import GPUModelRunner
from my_vllm.v1.worker.gpu.states import RequestState
from my_vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin


def _tiny_opt_config() -> OPTConfig:
    return OPTConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        ffn_dim=64,
        max_position_embeddings=32,
        word_embed_proj_dim=32,
        tie_word_embeddings=True,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        layerdrop=0.0,
        enable_bias=False,
        layer_norm_elementwise_affine=True,
        do_layer_norm_before=True,
        num_labels=1,
        init_std=0.02,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
    )


class _LoraEnabledRunner(GPUModelRunner):
    """Subclass that bypasses full GPUModelRunner.__init__.

    The full __init__ needs a real VllmConfig + device + CUDA
    graph capture. For unit testing the LoRA flow we only need
    the attributes that `_set_active_lora` and
    `_update_states` touch. We populate them by hand after
    calling a lightweight init that just stores the
    minimum-viable config.
    """

    class _StubBlockTables:
        def append_block_ids(self, idx: int, block_ids: list[int]) -> None:
            pass

    def __init__(self) -> None:  # noqa: D401 — test helper
        # Skip the full __init__: we only need the LoRA wiring.
        self.request_state = RequestState(
            max_num_reqs=8,
            max_model_len=32,
            device=torch.device("cpu"),
        )
        self.block_tables = self._StubBlockTables()
        self._active_lora_int_id = None
        # Capture set_lora calls for assertion in tests.
        self.set_lora_calls: list[int] = []


def test_gpu_model_runner_inherits_lora_mixin() -> None:
    """GPUModelRunner is a LoRAModelRunnerMixin subclass."""
    assert issubclass(GPUModelRunner, LoRAModelRunnerMixin)
    assert LoRAModelRunnerMixin in GPUModelRunner.__mro__


def test_new_request_data_carries_lora_int_id() -> None:
    """NewRequestData has the new `lora_int_id` field with default None."""
    d = NewRequestData(
        req_id="r1",
        prompt_token_ids=[1, 2, 3],
        block_ids=[],
        num_computed_tokens=0,
    )
    assert d.lora_int_id is None

    d2 = NewRequestData(
        req_id="r2",
        prompt_token_ids=[1, 2, 3],
        block_ids=[],
        num_computed_tokens=0,
        lora_int_id=7,
    )
    assert d2.lora_int_id == 7


def test_request_state_tracks_lora_id() -> None:
    """RequestState stores req_id → lora_int_id and clears on remove."""
    rs = RequestState(max_num_reqs=4, max_model_len=16, device=torch.device("cpu"))
    rs.add_request("r1", [1, 2, 3])
    rs.req_id_to_lora_id["r1"] = 5
    assert rs.req_id_to_lora_id == {"r1": 5}

    rs.remove_request("r1")
    assert rs.req_id_to_lora_id == {}


def _register_adapter(runner: _LoraEnabledRunner, lora_id: int) -> None:
    """Register a LoRA adapter on the runner's lora manager (no init_lora).

    For unit tests we don't actually want to wrap a full OPT
    model — that's covered by `test_C147_lora_wrap.py`. Here
    we just need the manager's `set_lora` flow to work, so we
    stub out a minimal `lora_worker_manager` / `lora_model`
    with a recorder.

    Skips `add_lora()` (which iterates `lora_modules.values()`
    to allocate A/B tensors and needs a real `base_layer`) and
    injects the request directly into `_adapters`.
    """
    from my_vllm.lora.lora_model import LoRAModel
    from my_vllm.lora.model_manager import LoRAModelManager
    from my_vllm.lora.worker_manager import LoRAWorkerManager

    class _RecorderLayer:
        def __init__(self) -> None:
            self.active_lora_int_id = None
            self.base_layer = None
            self._adapters: dict[int, object] = {}

        def set_lora(self, lora_id: int) -> None:
            self.active_lora_int_id = lora_id

    layers = {"dummy": _RecorderLayer()}
    lora_model = LoRAModel(model=None, max_loras=4, max_lora_rank=8)
    lora_model.lora_modules = layers

    # Inject adapter directly (bypass add_lora's per-wrapper
    # A/B allocation — those need a real base_layer).
    req = LoRARequest(lora_name=f"t{lora_id}", lora_int_id=lora_id, lora_path="")
    lora_model._adapters[lora_id] = req
    for layer in layers.values():
        layer._adapters[lora_id] = object()

    model_manager = LoRAModelManager(model=lora_model)
    runner.lora_model = lora_model
    runner.lora_model_manager = model_manager
    runner.lora_worker_manager = LoRAWorkerManager(
        model_manager=model_manager, device=torch.device("cpu"),
    )


def _wrap_with_recorder(runner: _LoraEnabledRunner, lora_id: int) -> None:
    """Replace `lora_worker_manager.set_lora` to record calls."""
    original = runner.lora_worker_manager.set_lora

    def recorder(_id: int) -> None:
        runner.set_lora_calls.append(_id)
        # Manually mark the stub layer as active.
        for layer in runner.lora_model.lora_modules.values():
            layer.active_lora_int_id = _id
        # Don't call original — it would call into the real
        # `LoRAModelManager.activate_adapter` which iterates
        # wrappers and checks `_adapters`. We've already set
        # the recorder state manually.

    runner.lora_worker_manager.set_lora = recorder  # type: ignore[assignment]


def test_set_active_lora_picks_first_non_none_in_new_reqs() -> None:
    """`_set_active_lora` selects the first non-None lora from new reqs."""
    runner = _LoraEnabledRunner()
    _register_adapter(runner, 7)
    _wrap_with_recorder(runner, 7)

    so = SchedulerOutput(
        scheduled_new_reqs=[
            NewRequestData(req_id="r0", prompt_token_ids=[1], lora_int_id=None),
            NewRequestData(req_id="r1", prompt_token_ids=[1], lora_int_id=7),
            NewRequestData(req_id="r2", prompt_token_ids=[1], lora_int_id=8),
        ],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"r0": 1, "r1": 1, "r2": 1},
        total_num_scheduled_tokens=3,
    )
    # Call _update_states first to populate req_id_to_lora_id,
    # then _set_active_lora.
    runner._update_states(so)
    runner._set_active_lora(so)

    assert runner.set_lora_calls == [7]
    assert runner._active_lora_int_id == 7
    assert runner.request_state.req_id_to_lora_id == {
        "r0": None, "r1": 7, "r2": 8,
    }


def test_set_active_lora_falls_back_to_cached_reqs() -> None:
    """No new req → use the cached req's lora_id."""
    runner = _LoraEnabledRunner()
    _register_adapter(runner, 9)
    _wrap_with_recorder(runner, 9)

    so = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=["c0", "c1"],
            num_computed_tokens=[1, 1],
            num_output_tokens=[1, 1],
        ),
        num_scheduled_tokens={"c0": 1, "c1": 1},
        total_num_scheduled_tokens=2,
    )
    # Seed the cached-req lora ids (in real flow, this happens
    # in `_update_states` when these reqs were first added).
    runner.request_state.req_id_to_lora_id["c0"] = None
    runner.request_state.req_id_to_lora_id["c1"] = 9
    runner._set_active_lora(so)

    assert runner.set_lora_calls == [9]
    assert runner._active_lora_int_id == 9


def test_set_active_lora_no_lora_active_is_noop() -> None:
    """All scheduled reqs have no lora → no set_lora call."""
    runner = _LoraEnabledRunner()
    _register_adapter(runner, 1)
    _wrap_with_recorder(runner, 1)

    so = SchedulerOutput(
        scheduled_new_reqs=[
            NewRequestData(req_id="r0", prompt_token_ids=[1], lora_int_id=None),
        ],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"r0": 1},
        total_num_scheduled_tokens=1,
    )
    runner._update_states(so)
    runner._set_active_lora(so)

    assert runner.set_lora_calls == []
    assert runner._active_lora_int_id is None


def test_set_active_lora_skips_redundant_calls() -> None:
    """Same active adapter as last step → no set_lora call (fast path)."""
    runner = _LoraEnabledRunner()
    _register_adapter(runner, 7)
    _wrap_with_recorder(runner, 7)
    runner._active_lora_int_id = 7  # pretend we already activated 7

    so = SchedulerOutput(
        scheduled_new_reqs=[
            NewRequestData(req_id="r1", prompt_token_ids=[1], lora_int_id=7),
        ],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"r1": 1},
        total_num_scheduled_tokens=1,
    )
    runner._update_states(so)
    runner._set_active_lora(so)

    assert runner.set_lora_calls == []  # fast-path: skipped
    assert runner._active_lora_int_id == 7


def test_set_active_lora_switches_on_change() -> None:
    """Active adapter changes → set_lora called with new id."""
    runner = _LoraEnabledRunner()
    _register_adapter(runner, 7)
    _register_adapter(runner, 8)
    _wrap_with_recorder(runner, 7)
    # Wrap again — the second _wrap_with_recorder replaces
    # set_lora with a fresh recorder that appends to
    # set_lora_calls. Good.
    _wrap_with_recorder(runner, 8)
    runner._active_lora_int_id = 7

    so = SchedulerOutput(
        scheduled_new_reqs=[
            NewRequestData(req_id="r2", prompt_token_ids=[1], lora_int_id=8),
        ],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"r2": 1},
        total_num_scheduled_tokens=1,
    )
    runner._update_states(so)
    runner._set_active_lora(so)

    assert runner.set_lora_calls == [8]
    assert runner._active_lora_int_id == 8


def test_set_active_lora_noop_when_lora_disabled() -> None:
    """No `lora_worker_manager` attribute → no-op (default config)."""
    runner = _LoraEnabledRunner()  # no _register_adapter call
    assert not hasattr(runner, "lora_worker_manager")

    so = SchedulerOutput(
        scheduled_new_reqs=[
            NewRequestData(req_id="r1", prompt_token_ids=[1], lora_int_id=99),
        ],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"r1": 1},
        total_num_scheduled_tokens=1,
    )
    # Should not raise.
    runner._set_active_lora(so)
    assert runner._active_lora_int_id is None