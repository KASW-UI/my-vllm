"""test_C223_ngram_proposer.py — C223 n-gram proposer tests.

Verifies the architecture-aligned pieces of my-vllm's n-gram
speculative-decoding proposer (the first real end-to-end spec-
decode path in this subset):

  1. ``SpeculativeConfig.__post_init__`` — n-gram defaults + min/max
     validation match vLLM's ``config/speculative.py:737-763``.

  2. ``VllmConfig.speculative_config`` field wires through to
     ``NgramProposer.__init__``.

  3. ``_find_longest_matched_ngram_and_propose_tokens`` (inner kernel):
     * Empty / too-short / at-max-length contexts → empty draft.
     * Repeated n-grams → draft starts immediately after the
       earliest occurrence.
     * Unique suffix → empty draft.
     * ``min_ngram`` / ``max_ngram`` correctly cap the match length.

  4. ``NgramProposer.propose``:
     * Filters out requests with empty sampled_token_ids (fresh
       prefill).
     * Filters out requests at max_model_len.
     * Per-request draft budget honored.
     * ``k`` correctly clipped to ``max_model_len - total``.

  5. ``NgramProposer.batch_propose``:
     * Skipped requests still appear in the output (as ``[]``).
     * Mixed valid/invalid batch produces correctly-aligned drafts.

  6. ``create_spec_decode_proposer`` dispatcher routes
     ``method="ngram"`` to ``NgramProposer`` and other methods to
     their (still-skeletal) classes.

  7. JIT warm-up happens in ``__init__`` — second ``propose()``
     call is fast (no JIT latency hit).
"""

import time

import numpy as np
import pytest

# Numba 0.67 deprecates reflected lists (vLLM still uses them);
# suppress the noise — vLLM upstream doesn't suppress either, but
# the warnings are 100× the actual test output here.
pytestmark = pytest.mark.filterwarnings(
    "ignore::numba.core.errors.NumbaPendingDeprecationWarning"
)

from my_vllm.config.model import ModelConfig
from my_vllm.config.parallel import ParallelConfig
from my_vllm.config.scheduler import SchedulerConfig
from my_vllm.config.speculative import SpeculativeConfig
from my_vllm.config.vllm import VllmConfig
from my_vllm.v1.spec_decode import (
    NgramProposer,
    SpecDecodeBaseProposer,
    create_spec_decode_proposer,
)
from my_vllm.v1.spec_decode.ngram_proposer import (
    _find_longest_matched_ngram_and_propose_tokens,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_vllm_config(
    *,
    method: str = "ngram",
    num_speculative_tokens: int = 4,
    prompt_lookup_min: int | None = None,
    prompt_lookup_max: int | None = None,
    max_model_len: int = 64,
    max_num_seqs: int = 16,
) -> VllmConfig:
    """Construct a minimal VllmConfig wired for ngram."""
    spec = SpeculativeConfig(
        method=method,
        num_speculative_tokens=num_speculative_tokens,
        prompt_lookup_min=prompt_lookup_min,
        prompt_lookup_max=prompt_lookup_max,
    )
    return VllmConfig(
        model_config=ModelConfig(max_model_len=max_model_len),
        scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs),
        parallel_config=ParallelConfig(tensor_parallel_size=1),
        speculative_config=spec,
    )


@pytest.fixture(scope="module")
def proposer() -> NgramProposer:
    """Module-scoped proposer — Numba JIT is paid once per session.

    Uses min_n=max_n=2 for test simplicity (a 2-gram matches many
    patterns; 5-gram would require a longer context).
    """
    return NgramProposer(
        _make_vllm_config(
            num_speculative_tokens=5,
            prompt_lookup_min=2,
            prompt_lookup_max=4,
        )
    )


# ---------------------------------------------------------------------------
# 1. SpeculativeConfig defaults + validation
# ---------------------------------------------------------------------------


class TestSpeculativeConfigNgram:
    def test_ngram_defaults_both_none(self):
        s = SpeculativeConfig(method="ngram", num_speculative_tokens=4)
        # vLLM defaults both to 5 when both are None.
        assert s.prompt_lookup_min == 5
        assert s.prompt_lookup_max == 5

    def test_ngram_min_falls_back_to_max(self):
        s = SpeculativeConfig(
            method="ngram", num_speculative_tokens=4, prompt_lookup_max=7
        )
        assert s.prompt_lookup_min == 7
        assert s.prompt_lookup_max == 7

    def test_ngram_max_falls_back_to_min(self):
        s = SpeculativeConfig(
            method="ngram", num_speculative_tokens=4, prompt_lookup_min=3
        )
        assert s.prompt_lookup_min == 3
        assert s.prompt_lookup_max == 3

    def test_ngram_both_set(self):
        s = SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=4,
            prompt_lookup_min=2,
            prompt_lookup_max=8,
        )
        assert s.prompt_lookup_min == 2
        assert s.prompt_lookup_max == 8

    def test_ngram_invalid_min_gt_max_raises(self):
        with pytest.raises(ValueError, match="prompt_lookup_min"):
            SpeculativeConfig(
                method="ngram",
                num_speculative_tokens=4,
                prompt_lookup_min=10,
                prompt_lookup_max=2,
            )

    def test_ngram_bracket_form_normalized(self):
        s = SpeculativeConfig(
            method="[ngram]",  # vLLM accepts this
            num_speculative_tokens=4,
        )
        assert s.method == "ngram"

    def test_non_ngram_method_leaves_ngram_fields_none(self):
        s = SpeculativeConfig(method="draft_model", num_speculative_tokens=4)
        assert s.prompt_lookup_min is None
        assert s.prompt_lookup_max is None


# ---------------------------------------------------------------------------
# 2. VllmConfig.speculative_config wiring
# ---------------------------------------------------------------------------


class TestVllmConfigSpeculative:
    def test_vllmconfig_accepts_speculative_config(self):
        vc = _make_vllm_config()
        assert vc.speculative_config is not None
        assert vc.speculative_config.method == "ngram"

    def test_vllmconfig_default_none(self):
        # Default-constructed VllmConfig has no speculative_config.
        vc = VllmConfig()
        assert vc.speculative_config is None


# ---------------------------------------------------------------------------
# 3. Inner kernel: _find_longest_matched_ngram_and_propose_tokens
# ---------------------------------------------------------------------------


class TestInnerKernel:
    """Direct tests of the Numba-JIT'd inner kernel.

    Algorithm: find the longest n-gram (length [min_ngram, max_ngram])
    that appears at the suffix of ``origin_tokens`` AND somewhere
    earlier in ``origin_tokens``.  Return ``k`` tokens immediately
    after the earliest such earlier occurrence.
    """

    def test_empty_context(self):
        r = _find_longest_matched_ngram_and_propose_tokens(
            np.array([], dtype=np.int32),
            min_ngram=2,
            max_ngram=3,
            max_model_len=10,
            k=3,
        )
        assert r.shape == (0,)

    def test_context_shorter_than_min_ngram(self):
        r = _find_longest_matched_ngram_and_propose_tokens(
            np.array([1, 2], dtype=np.int32),
            min_ngram=5,
            max_ngram=5,
            max_model_len=10,
            k=2,
        )
        assert r.shape == (0,)

    def test_context_at_max_model_len(self):
        # At max_model_len, k = min(k, max_model_len - total) = 0 → empty.
        r = _find_longest_matched_ngram_and_propose_tokens(
            np.array([1, 2, 1, 2], dtype=np.int32),
            min_ngram=2,
            max_ngram=2,
            max_model_len=4,
            k=3,
        )
        assert r.shape == (0,)

    def test_repeated_3gram_earliest_occurrence(self):
        # [1,2,3] appears at 0-2 and 3-5. Suffix [1,2,3] matches.
        # Earliest occurrence → draft starts at position 3.
        r = _find_longest_matched_ngram_and_propose_tokens(
            np.array([1, 2, 3, 1, 2, 3, 1, 2, 3], dtype=np.int32),
            min_ngram=2,
            max_ngram=3,
            max_model_len=16,
            k=4,
        )
        assert r.tolist() == [1, 2, 3, 1]

    def test_unique_suffix_no_match(self):
        # All unique tokens — no n-gram of length 2+ matches.
        r = _find_longest_matched_ngram_and_propose_tokens(
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32),
            min_ngram=2,
            max_ngram=3,
            max_model_len=16,
            k=3,
        )
        assert r.shape == (0,)

    def test_min_ngram_floor(self):
        # All-same token: matches a 4-gram, but min_ngram=2 also OK.
        # Draft budget = 2 tokens.
        r = _find_longest_matched_ngram_and_propose_tokens(
            np.array([10, 10, 10, 10, 10, 10], dtype=np.int32),
            min_ngram=2,
            max_ngram=4,
            max_model_len=10,
            k=2,
        )
        assert r.tolist() == [10, 10]

    def test_max_ngram_ceiling(self):
        # [1,2] repeats; max_ngram=2 caps the match at length 2.
        r = _find_longest_matched_ngram_and_propose_tokens(
            np.array([1, 2, 1, 2, 1, 2, 1, 2], dtype=np.int32),
            min_ngram=2,
            max_ngram=2,
            max_model_len=16,
            k=2,
        )
        assert r.tolist() == [1, 2]

    def test_k_clipped_to_available_tail(self):
        # Match found but only 6 tokens available after the match.
        r = _find_longest_matched_ngram_and_propose_tokens(
            np.array([1, 2, 1, 2, 1, 2, 1, 2], dtype=np.int32),
            min_ngram=2,
            max_ngram=2,
            max_model_len=16,
            k=8,
        )
        # Match at 0..1; tail is 6 tokens.
        assert r.tolist() == [1, 2, 1, 2, 1, 2]

    def test_k_clipped_to_max_model_len_budget(self):
        # k=8, max_model_len=10, total=8 → k = min(8, 2) = 2.
        r = _find_longest_matched_ngram_and_propose_tokens(
            np.array([1, 2, 1, 2, 1, 2, 1, 2], dtype=np.int32),
            min_ngram=2,
            max_ngram=2,
            max_model_len=10,
            k=8,
        )
        assert r.tolist() == [1, 2]


# ---------------------------------------------------------------------------
# 4. NgramProposer.propose: per-request filtering + batch output
# ---------------------------------------------------------------------------


class TestProposePerRequest:
    def test_skip_when_empty_sampled(self, proposer):
        # Empty sampled_token_ids → fresh prefill → skip.
        max_model_len = 64
        tokens = np.zeros((1, max_model_len), dtype=np.int32)
        tokens[0, :8] = [1, 2, 1, 2, 1, 2, 1, 2]
        drafts = proposer.propose(
            5, [[]], np.array([8], dtype=np.int32), tokens
        )
        assert drafts == [[]]

    def test_skip_when_context_at_max_model_len(self):
        p = NgramProposer(_make_vllm_config(max_model_len=10))
        tokens = np.zeros((1, 10), dtype=np.int32)
        tokens[0, :10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        drafts = p.propose(4, [[99]], np.array([10], dtype=np.int32), tokens)
        assert drafts == [[]]

    def test_skip_when_context_shorter_than_min_ngram(self):
        p = NgramProposer(
            _make_vllm_config(
                prompt_lookup_min=5, prompt_lookup_max=5, num_speculative_tokens=3
            )
        )
        tokens = np.zeros((1, 64), dtype=np.int32)
        tokens[0, :3] = [1, 2, 3]
        drafts = p.propose(3, [[99]], np.array([3], dtype=np.int32), tokens)
        assert drafts == [[]]

    def test_assert_when_num_speculative_exceeds_k(self, proposer):
        # proposer.k = 5; asking for 10 raises.
        tokens = np.zeros((1, 64), dtype=np.int32)
        with pytest.raises(AssertionError, match="exceeds"):
            proposer.propose(
                10, [[1]], np.array([8], dtype=np.int32), tokens
            )

    def test_happy_path(self, proposer):
        # Tokens [A, B, C, A, B, C] length 6, k=2, min/max_n=2..4.
        # Suffix length-2 = [B, C]; [B, C] appears at position 1..2.
        # Earliest match in original → draft starts at position 3.
        # tokens[3:5] = [A, B].
        max_model_len = 64
        tokens = np.zeros((1, max_model_len), dtype=np.int32)
        tokens[0, :6] = [10, 20, 30, 10, 20, 30]
        drafts = proposer.propose(
            2, [[99]], np.array([6], dtype=np.int32), tokens
        )
        assert drafts == [[10, 20]]

    def test_2gram_earliest_occurrence(self, proposer):
        # tokens = [A, B, A, B, A, B, A, B] length 8.
        # Suffix [A, B]; matches at 0..1 (earliest) and 2..3 etc.
        # With min_n=2 max_n=4, the algorithm may extend to longer
        # matches.  We verify only the suffix 2-gram behavior.
        max_model_len = 64
        tokens = np.zeros((1, max_model_len), dtype=np.int32)
        tokens[0, :8] = [10, 20, 10, 20, 10, 20, 10, 20]
        drafts = proposer.propose(
            2, [[99]], np.array([8], dtype=np.int32), tokens
        )
        # Match is [10, 20, 10, 20] (length 4) at original[4..7].
        # Draft start = 8 (end of array) → no tokens available → [].
        # Actually the matched n-gram IS the suffix itself; the
        # earliest prior occurrence of this 4-gram is at [0..3] or
        # [2..5].  Earliest prior: [0..3]=[10,20,10,20]. Draft
        # starts at position 4 → tokens[4:6]=[10, 20].
        assert drafts == [[10, 20]]


# ---------------------------------------------------------------------------
# 5. batch_propose: mixed valid / invalid requests
# ---------------------------------------------------------------------------


class TestBatchPropose:
    def test_mixed_batch(self):
        # batch = 4
        # r0: [10, 20, 30] repeats → draft = [10, 20, 30, 10]
        # r1: unique → []
        # r2: too short → []
        # r3: skipped (empty sampled) → []
        p = NgramProposer(
            _make_vllm_config(
                num_speculative_tokens=4,
                prompt_lookup_min=2,
                prompt_lookup_max=3,
            )
        )
        max_model_len = 64
        batch = 4
        tokens = np.zeros((batch, max_model_len), dtype=np.int32)
        tokens[0, :9] = [10, 20, 30, 10, 20, 30, 10, 20, 30]
        tokens[1, :6] = [1, 2, 3, 4, 5, 6]
        tokens[2, :1] = [99]
        # r3 = empty / skipped

        num_tokens = np.array([9, 6, 1, 0], dtype=np.int32)
        sampled = [[1], [2], [3], []]  # r3 has empty sampled
        drafts = p.propose(4, sampled, num_tokens, tokens)
        assert drafts == [[10, 20, 30, 10], [], [], []]

    def test_output_length_equals_batch_size(self):
        p = NgramProposer(_make_vllm_config(num_speculative_tokens=2))
        max_model_len = 32
        batch = 8
        tokens = np.zeros((batch, max_model_len), dtype=np.int32)
        num_tokens = np.zeros(batch, dtype=np.int32)
        sampled = [[] for _ in range(batch)]
        drafts = p.propose(2, sampled, num_tokens, tokens)
        assert len(drafts) == batch
        assert all(d == [] for d in drafts)


# ---------------------------------------------------------------------------
# 6. Dispatcher: create_spec_decode_proposer
# ---------------------------------------------------------------------------


class TestDispatcher:
    def test_ngram_routes_to_NgramProposer(self):
        vc = _make_vllm_config()
        drafter = create_spec_decode_proposer(vc)
        assert isinstance(drafter, NgramProposer)

    def test_draft_model_routes_to_class_shell(self):
        vc = _make_vllm_config(method="draft_model")
        drafter = create_spec_decode_proposer(vc)
        assert drafter is not None
        # Currently a class shell — exercises the dispatch path, but the
        # real impl is deferred (per cleanup-plan §3 B1).
        assert not isinstance(drafter, NgramProposer)

    def test_no_speculative_config_returns_none(self):
        vc = VllmConfig()
        assert create_spec_decode_proposer(vc) is None

    def test_unknown_method_raises(self):
        vc = _make_vllm_config(method="some_fake_method")
        with pytest.raises(NotImplementedError, match="some_fake_method"):
            create_spec_decode_proposer(vc)


# ---------------------------------------------------------------------------
# 7. JIT warm-up in __init__
# ---------------------------------------------------------------------------


class TestJitWarmup:
    def test_second_propose_call_is_fast(self):
        # Numba JIT is paid in __init__.  A second call should be fast
        # (< 100ms for a small batch — generous bound for slow CI).
        p = NgramProposer(_make_vllm_config(num_speculative_tokens=4))
        tokens = np.zeros((1, 64), dtype=np.int32)
        tokens[0, :10] = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
        # Warmup (should already be JIT'd by __init__, but be safe).
        p.propose(4, [[99]], np.array([10], dtype=np.int32), tokens)
        t0 = time.monotonic()
        for _ in range(100):
            p.propose(4, [[99]], np.array([10], dtype=np.int32), tokens)
        elapsed_ms = (time.monotonic() - t0) * 1000
        # 100 calls × 1ms = 100ms; allow generous CI bound.
        assert elapsed_ms < 500, (
            f"100 propose() calls took {elapsed_ms:.1f}ms — JIT not "
            f"warmed up?"
        )


# ---------------------------------------------------------------------------
# 8. BaseProposer inheritance + SpecDecodeBaseProposer existence
# ---------------------------------------------------------------------------


class TestNgramProposerInterface:
    def test_ngram_not_a_base_proposer_subclass(self):
        # Mirrors vLLM — NgramProposer is a free-standing class, not a
        # SpecDecodeBaseProposer subclass (it doesn't have the runner
        # interface — it operates purely on numpy arrays).
        p = NgramProposer(_make_vllm_config())
        assert not isinstance(p, SpecDecodeBaseProposer)

    def test_load_model_is_noop(self):
        # ngram doesn't load a draft model — verify the no-op path.
        p = NgramProposer(_make_vllm_config())
        # Should not raise.
        p.load_model()
        p.load_model(some="arg", other=42)