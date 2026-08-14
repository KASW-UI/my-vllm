"""tests/test_C247_elastic_rebalance.py — EP elastic rebalance closed loop.

C247: closes the A-level "EP 弹性 rebalance 闭环" gap from
``phase3-distributed-runtime-plan.md``.

The C230 skeleton stubbed `ElasticEPScalingState` as a no-op state
machine.  C247 closes the loop:

  1. **Real progress() wiring** — each ``progress()`` call fires the
     barrier / weight-transfer hooks for the corresponding state
     transition (event semantics: barrier → state change →
     weight-transfer).
  2. **TCP barrier callback** — ``barrier_fn(name, participants)`` is
     invoked before each transition; default is a no-op logger,
     production wires ``torch.distributed.TCPStore``.
  3. **NCCL weight transfer callback** — ``weight_transfer_fn(src,
     dst, kind)`` is invoked for TRANSFER_EXPERT_MAPPING and
     TRANSFER_WEIGHTS transitions; production wires
     ``NCCLWeightTransferEngine.trainer_send_weights`` + ``receive_weights``
     (C243).
  4. **run_to_completion() helper** — drives the state machine
     end-to-end with a max-steps guard.

Tests:

  * ``TestStateMachineInitialStates`` — scale_up new / existing /
    scale_down remaining / removing all start at the correct entry
    state.
  * ``TestScaleUpExistingFullSequence`` — 9-state protocol, barriers
    + weight transfers fire at the right transitions, run_to_completion
    lands at COMPLETE in exactly 8 steps.
  * ``TestScaleUpNewFullSequence`` — 4-state short protocol,
    expert_mapping transfer fires on PREPARE.
  * ``TestScaleDownRemainingFullSequence`` — 4-state protocol.
  * ``TestScaleDownRemovingFullSequence`` — 3-state short protocol.
  * ``TestProgressIdempotent`` — calling progress() after COMPLETE
    is a no-op (always returns True).
  * ``TestCallbacksReceiveArguments`` — ``barrier_fn`` receives
    correct (name, participants); ``weight_transfer_fn`` receives
    (src, dst, kind).
  * ``TestNCCLWeightTransferIntegration`` — scale-up existing's
    ``TRANSFER_WEIGHTS`` transition invokes the weight_transfer
    callback with kind="weights" — production wires this to
    ``NCCLWeightTransferEngine`` (C243); test verifies the calling
    convention.
  * ``TestEndToEndPolicyToStateMachine`` —
    ``decide_and_create`` → ``run_to_completion`` drives a full
    policy-driven scale-up sequence.
  * ``TestRunToCompletionGuard`` — max_steps cap returns the
    overflow indicator without infinite-looping.

Total: 10 test classes, ~25 cases.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    worker_type: str,
    scale_type: str,
    *,
    barrier_fn=None,
    weight_transfer_fn=None,
    local_worker_id: int = 0,
    participants=None,
):
    """Build an ``ElasticEPScalingState`` with the given callbacks."""
    from my_vllm.config.vllm import VllmConfig
    from my_vllm.distributed.elastic_ep import ElasticEPScalingState

    vllm_cfg = VllmConfig()
    return ElasticEPScalingState(
        vllm_config=vllm_cfg,
        new_parallel_config=vllm_cfg.parallel_config,
        worker_type=worker_type,
        scale_type=scale_type,
        barrier_fn=barrier_fn,
        weight_transfer_fn=weight_transfer_fn,
        local_worker_id=local_worker_id,
        participants=participants,
    )


# ---------------------------------------------------------------------------
# 1. Initial states
# ---------------------------------------------------------------------------


class TestStateMachineInitialStates:
    """Each (scale_type, worker_type) starts at the correct entry state."""

    def test_scale_up_new_starts_at_pre_kv_init(self):
        from my_vllm.distributed.elastic_ep import ScaleUpNewEngineState

        state = _make_state(worker_type="new", scale_type="scale_up")
        assert state.state == ScaleUpNewEngineState.PRE_KV_INIT

    def test_scale_up_existing_starts_at_wait_new_core(self):
        from my_vllm.distributed.elastic_ep import ScaleUpExistingEngineState

        state = _make_state(worker_type="existing", scale_type="scale_up")
        assert state.state == ScaleUpExistingEngineState.WAIT_NEW_CORE_ENGINES_INIT

    def test_scale_down_remaining_starts_at_prepare(self):
        from my_vllm.distributed.elastic_ep import ScaleDownRemainingEngineState

        state = _make_state(worker_type="remaining", scale_type="scale_down")
        assert state.state == ScaleDownRemainingEngineState.PREPARE

    def test_scale_down_removing_starts_at_prepare(self):
        from my_vllm.distributed.elastic_ep import ScaleDownRemovingEngineState

        state = _make_state(worker_type="removing", scale_type="scale_down")
        assert state.state == ScaleDownRemovingEngineState.PREPARE


# ---------------------------------------------------------------------------
# 2. Scale-up existing — full 9-state protocol
# ---------------------------------------------------------------------------


class TestScaleUpExistingFullSequence:
    """Scale-up existing hits all 9 states with correct callbacks."""

    def _run(self, **cb_kwargs):
        barriers = []
        transfers = []
        state = _make_state(
            worker_type="existing",
            scale_type="scale_up",
            barrier_fn=lambda name, parts: barriers.append((name, list(parts))),
            weight_transfer_fn=lambda src, dst, kind: transfers.append((src, dst, kind)),
            local_worker_id=0,
            participants=["w0", "w1", "w2"],
        )
        steps = state.run_to_completion()
        return state, steps, barriers, transfers

    def test_completes_in_8_steps(self):
        # 9 states → 8 transitions.
        _, steps, _, _ = self._run()
        assert steps == 8

    def test_final_state_is_complete(self):
        from my_vllm.distributed.elastic_ep import ScaleUpExistingEngineState

        state, _, _, _ = self._run()
        assert state.is_complete
        assert state.state == ScaleUpExistingEngineState.COMPLETE

    def test_barriers_fire_on_every_transition(self):
        _, _, barriers, _ = self._run()
        assert len(barriers) == 8
        # Verify the expected sequence (one barrier per transition).
        names = [n for n, _ in barriers]
        assert names == [
            "WAIT_NEW_CORE_INIT",
            "CREATE_STANDBY",
            "TRANSFER_EXPERT_MAPPING",
            "WAIT_NEW_CORE_WEIGHTS_INIT",
            "TRANSFER_WEIGHTS",
            "SYNC_KV_CACHE_MEMORY_SIZE",
            "SWITCH_AND_PREPARE",
            "EPLB_RESHUFFLE",
        ]

    def test_barrier_callback_receives_participants(self):
        _, _, barriers, _ = self._run()
        # All barriers fire with the same participants.
        for _, participants in barriers:
            assert participants == ["w0", "w1", "w2"]

    def test_weight_transfers_fire_on_transfer_states(self):
        # Two weight-transfer transitions:
        # TRANSFER_EXPERT_MAPPING (transition 2 → 3) + TRANSFER_WEIGHTS (4 → 5)
        _, _, _, transfers = self._run()
        assert len(transfers) == 2
        # Both transfers are self-targeted (src=local=0, dst=0)
        # because in single-host tests there's only one sender.
        assert all(src == 0 and dst == 0 for src, dst, _ in transfers)
        kinds = [k for _, _, k in transfers]
        assert kinds == ["expert_mapping", "weights"]


# ---------------------------------------------------------------------------
# 3. Scale-up new — short 4-state protocol
# ---------------------------------------------------------------------------


class TestScaleUpNewFullSequence:
    """Scale-up new only walks 4 states (no need for full existing protocol)."""

    def test_completes_in_3_steps(self):
        from my_vllm.distributed.elastic_ep import ScaleUpNewEngineState

        state = _make_state(worker_type="new", scale_type="scale_up")
        steps = state.run_to_completion()
        assert steps == 3
        assert state.state == ScaleUpNewEngineState.COMPLETE

    def test_expert_mapping_transfer_fires_on_prepare(self):
        transfers = []
        state = _make_state(
            worker_type="new",
            scale_type="scale_up",
            weight_transfer_fn=lambda src, dst, kind: transfers.append(kind),
        )
        state.run_to_completion()
        assert transfers == ["expert_mapping"]

    def test_no_weights_transfer_for_new_engine_path(self):
        # The new-engine path transfers only expert_mapping, never weights.
        transfers = []
        state = _make_state(
            worker_type="new",
            scale_type="scale_up",
            weight_transfer_fn=lambda src, dst, kind: transfers.append(kind),
        )
        state.run_to_completion()
        assert "weights" not in transfers


# ---------------------------------------------------------------------------
# 4. Scale-down — remaining + removing
# ---------------------------------------------------------------------------


class TestScaleDownRemainingFullSequence:
    """Scale-down remaining: PREPARE → EPLB_RESHUFFLE → SWITCH_AND_PREPARE → COMPLETE."""

    def test_completes_in_3_steps(self):
        from my_vllm.distributed.elastic_ep import ScaleDownRemainingEngineState

        state = _make_state(worker_type="remaining", scale_type="scale_down")
        steps = state.run_to_completion()
        assert steps == 3
        assert state.state == ScaleDownRemainingEngineState.COMPLETE

    def test_barrier_sequence(self):
        barriers = []
        state = _make_state(
            worker_type="remaining",
            scale_type="scale_down",
            barrier_fn=lambda name, parts: barriers.append(name),
        )
        state.run_to_completion()
        assert barriers == [
            "PREPARE_REMAINING",
            "EPLB_RESHUFFLE",
            "SWITCH_AND_PREPARE",
        ]


class TestScaleDownRemovingFullSequence:
    """Scale-down removing: PREPARE → EPLB_RESHUFFLE → COMPLETE (no SWITCH)."""

    def test_completes_in_2_steps(self):
        from my_vllm.distributed.elastic_ep import ScaleDownRemovingEngineState

        state = _make_state(worker_type="removing", scale_type="scale_down")
        steps = state.run_to_completion()
        assert steps == 2
        assert state.state == ScaleDownRemovingEngineState.COMPLETE


# ---------------------------------------------------------------------------
# 5. Idempotency after COMPLETE
# ---------------------------------------------------------------------------


class TestProgressIdempotent:
    """``progress()`` after COMPLETE is a no-op."""

    def test_progress_after_complete_is_safe(self):
        state = _make_state(worker_type="existing", scale_type="scale_up")
        state.run_to_completion()
        assert state.is_complete
        # Multiple progress() calls after complete — no exception,
        # no extra barriers / transfers.
        for _ in range(3):
            assert state.progress() is True
        # The transition log is unchanged.
        assert len(state.transitions) == 8  # 9 states → 8 transitions

    def test_run_to_completion_after_complete_is_zero_steps(self):
        state = _make_state(worker_type="existing", scale_type="scale_up")
        state.run_to_completion()
        steps = state.run_to_completion()
        assert steps == 0


# ---------------------------------------------------------------------------
# 6. Callback argument correctness
# ---------------------------------------------------------------------------


class TestCallbacksReceiveArguments:
    """``barrier_fn`` / ``weight_transfer_fn`` receive correct arguments."""

    def test_barrier_receives_name_and_participants(self):
        captured = []
        state = _make_state(
            worker_type="existing",
            scale_type="scale_up",
            barrier_fn=lambda name, parts: captured.append((name, tuple(parts))),
            participants=["a", "b", "c"],
        )
        state.run_to_completion()
        # First barrier is WAIT_NEW_CORE_INIT; check its participants.
        first = captured[0]
        assert first[0] == "WAIT_NEW_CORE_INIT"
        assert first[1] == ("a", "b", "c")

    def test_weight_transfer_src_is_local(self):
        captured = []
        state = _make_state(
            worker_type="existing",
            scale_type="scale_up",
            weight_transfer_fn=lambda src, dst, kind: captured.append((src, dst, kind)),
            local_worker_id=3,
        )
        state.run_to_completion()
        # Both transfers are self-targeted (single-host test).
        for src, dst, kind in captured:
            assert src == 3
            assert dst == 3

    def test_transitions_log_records_from_to_scale(self):
        state = _make_state(worker_type="existing", scale_type="scale_up")
        state.run_to_completion()
        # 8 transitions; each is (from, to, "scale_up").
        assert len(state.transitions) == 8
        for from_state, to_state, scale_type in state.transitions:
            assert scale_type == "scale_up"
            # Sibling transitions should be adjacent in the enum.
            assert to_state.value == from_state.value + 1


# ---------------------------------------------------------------------------
# 7. NCCL weight transfer integration (production wiring)
# ---------------------------------------------------------------------------


class TestNCCLWeightTransferIntegration:
    """``weight_transfer_fn(0, 0, "weights")`` fires on TRANSFER_WEIGHTS.

    Production wires this callback to ``NCCLWeightTransferEngine``
    (C243) which broadcasts weights via NCCL.  This test verifies
    the calling convention is correct — the actual NCCL broadcast
    is exercised in ``test_C243_weight_transfer.py``.
    """

    def test_weights_transfer_kind_fires_on_scale_up_existing(self):
        captured = []
        state = _make_state(
            worker_type="existing",
            scale_type="scale_up",
            weight_transfer_fn=lambda src, dst, kind: captured.append(kind),
        )
        state.run_to_completion()
        # TRANSFER_EXPERT_MAPPING (kind="expert_mapping") +
        # TRANSFER_WEIGHTS (kind="weights").
        assert "weights" in captured
        assert captured.index("weights") == 1  # second of two transfers

    def test_expert_mapping_transfer_kind_fires(self):
        captured = []
        state = _make_state(
            worker_type="existing",
            scale_type="scale_up",
            weight_transfer_fn=lambda src, dst, kind: captured.append(kind),
        )
        state.run_to_completion()
        assert "expert_mapping" in captured
        assert captured.index("expert_mapping") == 0  # first of two

    def test_scale_down_has_no_weight_transfers(self):
        """Scale-down doesn't transfer weights — only barriers."""
        captured = []
        # Scale-down existing (removing engine path).
        state = _make_state(
            worker_type="removing",
            scale_type="scale_down",
            weight_transfer_fn=lambda src, dst, kind: captured.append(kind),
        )
        state.run_to_completion()
        assert captured == []  # no transfers, only barriers


# ---------------------------------------------------------------------------
# 8. End-to-end: policy → state machine
# ---------------------------------------------------------------------------


class TestEndToEndPolicyToStateMachine:
    """``decide_and_create`` → ``run_to_completion`` drives a full
    policy-driven sequence."""

    def test_kv_pressure_triggers_scale_up(self):
        from my_vllm.config.vllm import VllmConfig
        from my_vllm.distributed.elastic_ep import (
            ElasticEPScalingState,
            ScaleUpExistingEngineState,
        )
        from my_vllm.distributed.elastic_ep.rebalance import (
            ThresholdEPLoadPolicy,
            decide_and_create,
        )

        vllm_cfg = VllmConfig()
        sc = vllm_cfg.scheduler_config
        sc.kv_aware = True
        vllm_cfg.scheduler_config = sc

        # High KV pressure → scale_up by 1.
        signal_args = {
            "current_dp_size": 4,
            "target_dp_size": None,
            "kv_cache_utilization": 0.95,  # above default up_threshold=0.85
            "pending_prefill_tokens": 1000,
            "num_active_requests": 64,
            "elapsed_seconds": 120.0,  # past cold-start hysteresis
        }
        state = decide_and_create(
            vllm_cfg,
            type("Signal", (), signal_args)(),
            policy=ThresholdEPLoadPolicy(),
        )
        assert isinstance(state, ElasticEPScalingState)
        assert state.scale_type == "scale_up"
        assert state.worker_type == "existing"
        assert state.new_parallel_config.data_parallel_size == 5

        # Drive to completion with no-op callbacks.
        steps = state.run_to_completion()
        assert state.is_complete
        assert state.state == ScaleUpExistingEngineState.COMPLETE
        assert steps == 8

    def test_low_kv_triggers_scale_down(self):
        from my_vllm.config.vllm import VllmConfig
        from my_vllm.distributed.elastic_ep import (
            ElasticEPScalingState,
            ScaleDownRemainingEngineState,
        )
        from my_vllm.distributed.elastic_ep.rebalance import (
            ThresholdEPLoadPolicy,
            decide_and_create,
        )

        vllm_cfg = VllmConfig()
        signal_args = {
            "current_dp_size": 4,
            "target_dp_size": None,
            "kv_cache_utilization": 0.2,  # below down_threshold=0.4
            "pending_prefill_tokens": 0,
            "num_active_requests": 4,
            "elapsed_seconds": 120.0,
        }
        state = decide_and_create(
            vllm_cfg,
            type("Signal", (), signal_args)(),
            policy=ThresholdEPLoadPolicy(),
        )
        assert isinstance(state, ElasticEPScalingState)
        assert state.scale_type == "scale_down"
        assert state.worker_type == "remaining"
        assert state.new_parallel_config.data_parallel_size == 3

        steps = state.run_to_completion()
        assert state.state == ScaleDownRemainingEngineState.COMPLETE
        assert steps == 3


# ---------------------------------------------------------------------------
# 9. max_steps guard
# ---------------------------------------------------------------------------


class TestRunToCompletionGuard:
    """``run_to_completion(max_steps=...)`` caps infinite loops."""

    def test_max_steps_returns_stable_count(self):
        state = _make_state(worker_type="existing", scale_type="scale_up")
        # 9 states → 8 steps required.  Cap at 100 → still completes.
        steps = state.run_to_completion(max_steps=100)
        assert steps == 8
        assert state.is_complete

    def test_zero_max_steps_no_advance(self):
        state = _make_state(worker_type="existing", scale_type="scale_up")
        # max_steps=0 → zero transitions, state remains at entry.
        steps = state.run_to_completion(max_steps=0)
        assert steps == 0
        assert not state.is_complete

    def test_max_steps_caps_at_exact_limit(self):
        """8 steps required; cap at 5 → exactly 5 taken, COMPLETE not reached."""
        state = _make_state(worker_type="existing", scale_type="scale_up")
        steps = state.run_to_completion(max_steps=5)
        assert steps == 5
        assert not state.is_complete

    def test_default_max_steps_sufficient_for_all_paths(self):
        """Default 64 is comfortably above the longest path (8 steps)."""
        for worker_type, scale_type in [
            ("existing", "scale_up"),
            ("new", "scale_up"),
            ("remaining", "scale_down"),
            ("removing", "scale_down"),
        ]:
            state = _make_state(worker_type=worker_type, scale_type=scale_type)
            steps = state.run_to_completion()  # default 64
            assert steps <= 64
            assert state.is_complete


# ---------------------------------------------------------------------------
# 10. NoOp policy + decide_and_create hold-path
# ---------------------------------------------------------------------------


class TestNoOpPolicyHoldPath:
    """``NoOpEPRebalancePolicy`` always returns hold → no state machine."""

    def test_no_op_returns_none(self):
        from my_vllm.config.vllm import VllmConfig
        from my_vllm.distributed.elastic_ep.rebalance import (
            NoOpEPRebalancePolicy,
            decide_and_create,
        )

        vllm_cfg = VllmConfig()
        signal_args = {
            "current_dp_size": 4,
            "target_dp_size": None,
            "kv_cache_utilization": 0.9,
            "pending_prefill_tokens": 0,
            "num_active_requests": 32,
            "elapsed_seconds": 60.0,
        }
        state = decide_and_create(
            vllm_cfg,
            type("Signal", (), signal_args)(),
            policy=NoOpEPRebalancePolicy(),
        )
        # NoOp → hold → state machine is None.
        assert state is None