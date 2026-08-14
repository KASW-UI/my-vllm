"""test_C230_elastic_ep.py — C230 elastic EP rebalance + state machine.

Verifies the C230 path:

  1. **State machine** — `ElasticEPScalingState` advances through
     every prescribed sequence of states for the four
     (scale_type, worker_type) combinations:
       * ``("scale_up",   "new")``      → 4 states (PRE_KV_INIT
                                                      → PREPARE
                                                      → EPLB_RESHUFFLE
                                                      → COMPLETE)
       * ``("scale_up",   "existing")`` → 9 states
       * ``("scale_down", "remaining")`` → 4 states
       * ``("scale_down", "removing")``  → 3 states
     Total = 20 transitions; matches vLLM
     ``vllm/distributed/elastic_ep/elastic_state.py``.

  2. **`ParallelConfig.data_parallel_size`** + `clone_with_dp_size()`
     work correctly (the field added to support the rebalance
     decision).

  3. **`EPRebalancePolicy`** default behaviour:
     * `NoOpEPRebalancePolicy` always returns `hold`.
     * `ThresholdEPLoadPolicy`:
        * Threshold validation (`up >= down`, `min <= max`).
        * Explicit `target_dp_size` is always honored.
        * Utilization-driven `scale_up` when kv_util ≥ up_threshold.
        * Utilization-driven `scale_down` when kv_util ≤ down_threshold.
        * `hold` within the thresholds.
        * Hysteresis: refuses to decide until elapsed
          ≥ `min_downtime_seconds`.

  4. **`create_scaling_state_for_decision`** bridges the policy
     output to a state machine:
     * `hold` → `None`.
     * `scale_up` with `worker_type="existing"`.
     * `scale_down` with `worker_type="remaining"`.
     * Skips construction when `new_dp_size == current_dp_size`.

  5. **`decide_and_create`** convenience: end-to-end signal →
     decision → state machine; default policy is
     `ThresholdEPLoadPolicy`.

  6. **Worker type semantics** for `scale_up` vs `scale_down`:
     * `scale_up`  → controls `existing` engines (the survivors).
     * `scale_down` → controls `remaining` engines (the survivors).
     * `new` / `removing` worker types are valid initial states
       but not produced by the policy (they live on the other side
       of the scaling event).
"""

import pytest

from my_vllm.config.parallel import ParallelConfig
from my_vllm.config.vllm import VllmConfig
from my_vllm.distributed.elastic_ep import (
    ElasticEPScalingState,
    EPRebalanceDecision,
    EPRebalancePolicy,
    EPRebalanceSignal,
    NoOpEPRebalancePolicy,
    ScaleDownRemovingEngineState,
    ScaleDownRemainingEngineState,
    ScaleUpExistingEngineState,
    ScaleUpNewEngineState,
    ThresholdEPLoadPolicy,
    create_scaling_state_for_decision,
    decide_and_create,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def vllm_config_dp4() -> VllmConfig:
    vc = VllmConfig()
    vc.parallel_config.data_parallel_size = 4
    return vc


# ---------------------------------------------------------------------------
# 1. State machine — full sequences
# ---------------------------------------------------------------------------


class TestStateMachineSequences:
    """Verify the 4 (scale_type, worker_type) state sequences."""

    @staticmethod
    def _walk_states(es: ElasticEPScalingState) -> list[int]:
        """Step the state machine to COMPLETE; return the sequence
        of state values traversed (including initial state).
        """
        states = [int(es.state)]
        for _ in range(50):  # safety bound
            if es.is_complete:
                break
            es.progress()
            states.append(int(es.state))
        return states

    def test_scale_up_new_engine(self, vllm_config_dp4):
        # PRE_KV_INIT → PREPARE → EPLB_RESHUFFLE → COMPLETE (4 states)
        es = ElasticEPScalingState(
            vllm_config=vllm_config_dp4,
            new_parallel_config=vllm_config_dp4.parallel_config.clone_with_dp_size(5),
            worker_type="new",
            scale_type="scale_up",
        )
        assert es.state == ScaleUpNewEngineState.PRE_KV_INIT
        states = self._walk_states(es)
        assert states == [0, 1, 2, 3]
        assert es.is_complete
        assert es.state == ScaleUpNewEngineState.COMPLETE

    def test_scale_up_existing_engine(self, vllm_config_dp4):
        # WAIT_NEW_CORE_ENGINES_INIT → CREATE_STANDBY_GROUPS
        #   → TRANSFER_EXPERT_MAPPING → WAIT_NEW_CORE_ENGINES_WEIGHTS_INIT
        #   → TRANSFER_WEIGHTS → SYNC_KV_CACHE_MEMORY_SIZE
        #   → SWITCH_AND_PREPARE → EPLB_RESHUFFLE → COMPLETE (9 states)
        es = ElasticEPScalingState(
            vllm_config=vllm_config_dp4,
            new_parallel_config=vllm_config_dp4.parallel_config.clone_with_dp_size(5),
            worker_type="existing",
            scale_type="scale_up",
        )
        assert es.state == ScaleUpExistingEngineState.WAIT_NEW_CORE_ENGINES_INIT
        states = self._walk_states(es)
        assert states == [0, 1, 2, 3, 4, 5, 6, 7, 8]
        assert es.is_complete

    def test_scale_down_remaining_engine(self, vllm_config_dp4):
        # PREPARE → EPLB_RESHUFFLE → SWITCH_AND_PREPARE → COMPLETE
        es = ElasticEPScalingState(
            vllm_config=vllm_config_dp4,
            new_parallel_config=vllm_config_dp4.parallel_config.clone_with_dp_size(3),
            worker_type="remaining",
            scale_type="scale_down",
        )
        assert es.state == ScaleDownRemainingEngineState.PREPARE
        states = self._walk_states(es)
        assert states == [0, 1, 2, 3]
        assert es.is_complete

    def test_scale_down_removing_engine(self, vllm_config_dp4):
        # PREPARE → EPLB_RESHUFFLE → COMPLETE
        es = ElasticEPScalingState(
            vllm_config=vllm_config_dp4,
            new_parallel_config=vllm_config_dp4.parallel_config.clone_with_dp_size(3),
            worker_type="removing",
            scale_type="scale_down",
        )
        assert es.state == ScaleDownRemovingEngineState.PREPARE
        states = self._walk_states(es)
        assert states == [0, 1, 2]
        assert es.is_complete

    def test_total_transitions_match_vllm(self, vllm_config_dp4):
        # State sequences — n_states per (scale_type, worker_type):
        #   scale_up   new       = 4 states → 3 transitions
        #   scale_up   existing  = 9 states → 8 transitions
        #   scale_down remaining = 4 states → 3 transitions
        #   scale_down removing  = 3 states → 2 transitions
        # Total transitions = 3 + 8 + 3 + 2 = 16.
        # (Each transition = one progress() call that advances state;
        # at the COMPLETE state, progress() returns True but doesn't
        # advance, and is_complete short-circuits the loop.)
        expected_transitions_per_state = {
            ("scale_up", "new"): 3,
            ("scale_up", "existing"): 8,
            ("scale_down", "remaining"): 3,
            ("scale_down", "removing"): 2,
        }
        for (
            scale_type,
            worker_type,
        ), expected in expected_transitions_per_state.items():
            new_dp = 5 if scale_type == "scale_up" else 3
            es = ElasticEPScalingState(
                vllm_config=vllm_config_dp4,
                new_parallel_config=vllm_config_dp4.parallel_config.clone_with_dp_size(
                    new_dp
                ),
                worker_type=worker_type,
                scale_type=scale_type,
            )
            n = 0
            for _ in range(50):
                if es.is_complete:
                    break
                es.progress()
                n += 1
            assert n == expected, (
                f"{scale_type}/{worker_type}: expected {expected} "
                f"transitions, got {n}"
            )

    def test_progress_returns_true_at_completion(self, vllm_config_dp4):
        es = ElasticEPScalingState(
            vllm_config=vllm_config_dp4,
            new_parallel_config=vllm_config_dp4.parallel_config.clone_with_dp_size(5),
            worker_type="new",
            scale_type="scale_up",
        )
        # Drive one step from PRE_KV_INIT (idx 0) → idx 1.
        result = es.progress()
        assert result is True  # not yet complete
        # Drive PREPARE → EPLB_RESHUFFLE → COMPLETE.
        result = es.progress()
        assert result is True
        result = es.progress()
        assert result is True  # this step reaches COMPLETE
        # Already at COMPLETE → still True.
        result = es.progress()
        assert result is True

    def test_elapsed_seconds_increases(self, vllm_config_dp4):
        es = ElasticEPScalingState(
            vllm_config=vllm_config_dp4,
            new_parallel_config=vllm_config_dp4.parallel_config.clone_with_dp_size(5),
            worker_type="new",
            scale_type="scale_up",
        )
        e0 = es.elapsed_seconds
        es.progress()
        e1 = es.elapsed_seconds
        # Progress is fast, but elapsed_seconds should not decrease.
        assert e1 >= e0


# ---------------------------------------------------------------------------
# 2. ParallelConfig.data_parallel_size + clone_with_dp_size
# ---------------------------------------------------------------------------


class TestParallelConfigDPField:
    def test_default_dp_size_is_one(self):
        pc = ParallelConfig()
        assert pc.data_parallel_size == 1

    def test_clone_with_dp_size(self):
        pc = ParallelConfig(tensor_parallel_size=2, data_parallel_size=3)
        clone = pc.clone_with_dp_size(7)
        assert clone.data_parallel_size == 7
        assert clone.tensor_parallel_size == 2
        # Original is unchanged.
        assert pc.data_parallel_size == 3

    def test_clone_is_independent(self):
        # Mutating the clone doesn't affect the original (deep copy).
        pc = ParallelConfig(data_parallel_size=3)
        clone = pc.clone_with_dp_size(7)
        clone.data_parallel_size = 99
        assert pc.data_parallel_size == 3


# ---------------------------------------------------------------------------
# 3. EPRebalancePolicy default implementations
# ---------------------------------------------------------------------------


class TestNoOpPolicy:
    def test_always_holds(self):
        policy = NoOpEPRebalancePolicy()
        for util in [0.0, 0.5, 0.9, 0.99]:
            signal = EPRebalanceSignal(
                current_dp_size=4, kv_cache_utilization=util
            )
            d = policy.decide(signal)
            assert d.is_hold
            assert d.new_dp_size == 4


class TestThresholdPolicyValidation:
    def test_down_gt_up_raises(self):
        with pytest.raises(ValueError, match="down_threshold"):
            ThresholdEPLoadPolicy(up_threshold=0.4, down_threshold=0.85)

    def test_min_gt_max_raises(self):
        with pytest.raises(ValueError, match="min_dp_size"):
            ThresholdEPLoadPolicy(min_dp_size=8, max_dp_size=4)

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError, match="down_threshold"):
            ThresholdEPLoadPolicy(up_threshold=0.85, down_threshold=-0.1)


class TestThresholdPolicyDecisions:
    def test_scale_up_on_high_util(self):
        policy = ThresholdEPLoadPolicy(
            up_threshold=0.85, down_threshold=0.4
        )
        signal = EPRebalanceSignal(
            current_dp_size=4,
            kv_cache_utilization=0.9,
            elapsed_seconds=60.0,
        )
        d = policy.decide(signal)
        assert d.scale_type == "scale_up"
        assert d.new_dp_size == 5
        assert "up_threshold" in d.reason

    def test_scale_down_on_low_util(self):
        policy = ThresholdEPLoadPolicy(
            up_threshold=0.85, down_threshold=0.4
        )
        signal = EPRebalanceSignal(
            current_dp_size=4,
            kv_cache_utilization=0.3,
            elapsed_seconds=60.0,
        )
        d = policy.decide(signal)
        assert d.scale_type == "scale_down"
        assert d.new_dp_size == 3
        assert "down_threshold" in d.reason

    def test_hold_within_thresholds(self):
        policy = ThresholdEPLoadPolicy(
            up_threshold=0.85, down_threshold=0.4
        )
        signal = EPRebalanceSignal(
            current_dp_size=4,
            kv_cache_utilization=0.6,
            elapsed_seconds=60.0,
        )
        d = policy.decide(signal)
        assert d.is_hold
        assert d.new_dp_size == 4
        assert "within" in d.reason

    def test_cold_start_hysteresis_keeps_holding(self):
        # At 5s the policy must NOT decide to scale, regardless of util.
        policy = ThresholdEPLoadPolicy(
            up_threshold=0.85, down_threshold=0.4, min_downtime_seconds=30.0
        )
        signal = EPRebalanceSignal(
            current_dp_size=4,
            kv_cache_utilization=0.99,  # would otherwise trigger scale_up
            elapsed_seconds=5.0,
        )
        d = policy.decide(signal)
        assert d.is_hold
        assert d.metadata["min_downtime_seconds"] == 30.0

    def test_hysteresis_overridden_by_explicit_target(self):
        # Cold start + user-specified target → obey user, ignore hysteresis.
        policy = ThresholdEPLoadPolicy(min_downtime_seconds=30.0)
        signal = EPRebalanceSignal(
            current_dp_size=4,
            target_dp_size=8,
            kv_cache_utilization=0.5,
            elapsed_seconds=5.0,
        )
        d = policy.decide(signal)
        assert d.scale_type == "scale_up"
        assert d.new_dp_size == 8
        assert "user_target" in d.reason

    def test_explicit_target_smaller(self):
        policy = ThresholdEPLoadPolicy()
        signal = EPRebalanceSignal(
            current_dp_size=8,
            target_dp_size=4,
            kv_cache_utilization=0.5,
            elapsed_seconds=60.0,
        )
        d = policy.decide(signal)
        assert d.scale_type == "scale_down"
        assert d.new_dp_size == 4

    def test_target_at_max_dp_capped(self):
        policy = ThresholdEPLoadPolicy(max_dp_size=16)
        signal = EPRebalanceSignal(
            current_dp_size=8,
            target_dp_size=64,  # absurdly large
            elapsed_seconds=60.0,
        )
        d = policy.decide(signal)
        assert d.scale_type == "scale_up"
        assert d.new_dp_size == 16  # capped to max_dp_size

    def test_target_at_min_dp_floored(self):
        policy = ThresholdEPLoadPolicy(min_dp_size=1)
        signal = EPRebalanceSignal(
            current_dp_size=4,
            target_dp_size=0,  # invalid
            elapsed_seconds=60.0,
        )
        d = policy.decide(signal)
        assert d.scale_type == "scale_down"
        assert d.new_dp_size == 1  # floored to min_dp_size

    def test_already_at_max_no_scale_up(self):
        policy = ThresholdEPLoadPolicy(max_dp_size=8)
        signal = EPRebalanceSignal(
            current_dp_size=8,
            kv_cache_utilization=0.99,
            elapsed_seconds=60.0,
        )
        d = policy.decide(signal)
        assert d.is_hold
        assert d.new_dp_size == 8

    def test_already_at_min_no_scale_down(self):
        policy = ThresholdEPLoadPolicy(min_dp_size=1)
        signal = EPRebalanceSignal(
            current_dp_size=1,
            kv_cache_utilization=0.05,
            elapsed_seconds=60.0,
        )
        d = policy.decide(signal)
        assert d.is_hold


# ---------------------------------------------------------------------------
# 4. create_scaling_state_for_decision — policy → state machine bridge
# ---------------------------------------------------------------------------


class TestScalingStateFactory:
    def test_hold_returns_none(self, vllm_config_dp4):
        decision = EPRebalanceDecision(
            scale_type="hold", new_dp_size=4, reason="test"
        )
        es = create_scaling_state_for_decision(vllm_config_dp4, decision)
        assert es is None

    def test_scale_up_existing_for_positive_decision(self, vllm_config_dp4):
        decision = EPRebalanceDecision(
            scale_type="scale_up", new_dp_size=5, reason="test"
        )
        es = create_scaling_state_for_decision(vllm_config_dp4, decision)
        assert es is not None
        assert es.scale_type == "scale_up"
        assert es.worker_type == "existing"
        assert es.new_parallel_config.data_parallel_size == 5
        assert es.state == ScaleUpExistingEngineState.WAIT_NEW_CORE_ENGINES_INIT

    def test_scale_down_remaining_for_negative_decision(
        self, vllm_config_dp4
    ):
        decision = EPRebalanceDecision(
            scale_type="scale_down", new_dp_size=3, reason="test"
        )
        es = create_scaling_state_for_decision(vllm_config_dp4, decision)
        assert es is not None
        assert es.scale_type == "scale_down"
        assert es.worker_type == "remaining"
        assert es.new_parallel_config.data_parallel_size == 3
        assert es.state == ScaleDownRemainingEngineState.PREPARE

    def test_target_equals_current_returns_none(self, vllm_config_dp4):
        decision = EPRebalanceDecision(
            scale_type="hold", new_dp_size=4, reason="test"
        )
        # Even if scale_type were 'scale_up' but new_dp == current,
        # the factory short-circuits to None.
        decision = EPRebalanceDecision(
            scale_type="scale_up", new_dp_size=4, reason="test"
        )
        es = create_scaling_state_for_decision(vllm_config_dp4, decision)
        assert es is None


# ---------------------------------------------------------------------------
# 5. decide_and_create — end-to-end
# ---------------------------------------------------------------------------


class TestDecideAndCreate:
    def test_high_util_creates_scale_up_state_machine(self, vllm_config_dp4):
        signal = EPRebalanceSignal(
            current_dp_size=4,
            kv_cache_utilization=0.9,
            elapsed_seconds=60.0,
        )
        es = decide_and_create(vllm_config_dp4, signal)
        assert es is not None
        assert es.scale_type == "scale_up"

    def test_low_util_creates_scale_down_state_machine(self, vllm_config_dp4):
        signal = EPRebalanceSignal(
            current_dp_size=4,
            kv_cache_utilization=0.3,
            elapsed_seconds=60.0,
        )
        es = decide_and_create(vllm_config_dp4, signal)
        assert es is not None
        assert es.scale_type == "scale_down"

    def test_mid_util_returns_none(self, vllm_config_dp4):
        signal = EPRebalanceSignal(
            current_dp_size=4,
            kv_cache_utilization=0.6,
            elapsed_seconds=60.0,
        )
        es = decide_and_create(vllm_config_dp4, signal)
        assert es is None

    def test_no_op_policy_always_returns_none(self, vllm_config_dp4):
        signal = EPRebalanceSignal(
            current_dp_size=4,
            kv_cache_utilization=0.99,
            elapsed_seconds=60.0,
        )
        es = decide_and_create(
            vllm_config_dp4, signal, policy=NoOpEPRebalancePolicy()
        )
        assert es is None

    def test_explicit_target_drives_decision(self, vllm_config_dp4):
        signal = EPRebalanceSignal(
            current_dp_size=2,
            target_dp_size=8,
            kv_cache_utilization=0.5,
            elapsed_seconds=60.0,
        )
        es = decide_and_create(vllm_config_dp4, signal)
        assert es is not None
        assert es.scale_type == "scale_up"
        assert es.new_parallel_config.data_parallel_size == 8


# ---------------------------------------------------------------------------
# 6. Custom policy — proves the abstraction is extensible
# ---------------------------------------------------------------------------


class TestCustomPolicy:
    def test_user_defined_policy(self):
        class AlwaysScaleUpBy2(EPRebalancePolicy):
            def decide(self, signal):
                return EPRebalanceDecision(
                    scale_type="scale_up",
                    new_dp_size=signal.current_dp_size + 2,
                    reason="always_double_up",
                )

        vc = VllmConfig()
        vc.parallel_config.data_parallel_size = 2
        signal = EPRebalanceSignal(current_dp_size=2, elapsed_seconds=60.0)
        es = decide_and_create(vc, signal, policy=AlwaysScaleUpBy2())
        assert es is not None
        assert es.new_parallel_config.data_parallel_size == 4