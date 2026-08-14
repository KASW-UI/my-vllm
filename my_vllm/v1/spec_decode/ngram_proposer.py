"""NgramProposer — prompt-lookup n-gram speculative-decoding proposer.

Mirrors vLLM V1 vllm/v1/spec_decode/ngram_proposer.py.

The N-gram proposer is the simplest (and oldest) training-free
speculative decoder: it does not require a draft model.  At each
target-model step, the proposer looks at the **prompt + already-
generated** token sequence and finds the longest n-gram (length in
``[prompt_lookup_min, prompt_lookup_max]``) that matches the suffix
of the context.  If a match is found, the ``k`` tokens immediately
after the matched n-gram are returned as draft tokens for parallel
verification by the target model.

vLLM has two flavors:

  * ``NgramProposer`` (this file) — pure-Python / Numba CPU kernel.
    Suitable for small batches and the v1 baseline.
  * ``NgramProposerGPU`` — Triton-fused GPU kernel (added in 2025;
    large-batch server-side acceleration).

C223 ships ``NgramProposer`` only (the GPU variant is an internal
vLLM optimization, not part of the architecture boundary).

Algorithm
---------

Given a context ``tokens = [t_0, t_1, ..., t_{n-1}]`` and a draft
budget ``k``:

  1. Reverse the context: ``rev = tokens[::-1]``.
  2. Compute the longest-prefix-of-suffix (LPS / KMP failure-function)
     array restricted to windows of size ``[min_n, max_n]``.
  3. The earliest position (in ``rev``) where the prefix-length
     crosses ``min_n`` is the matched-ngram start in ``tokens``;
     the n-gram itself is ``tokens[end - matched_len : end]``.
  4. Draft = ``tokens[end : end + k]`` (clipped to ``max_model_len -
     total`` and to the available tail).

The Numba kernel is parallelized over the batch dimension (one
``prange`` thread per request).  The inner LPS loop is the
standard KMP failure-function update, capped at ``max_ngram`` to
keep the prefix array small.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np

# Numba is a hard dep in real vLLM (cuda.txt pins == 0.65.0).
# Import lazily so that ``from my_vllm.v1.spec_decode import ...``
# still works on CPU-only environments; the JIT path raises a clear
# ImportError on use.
try:
    from numba import get_num_threads, jit, njit, prange, set_num_threads

    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on CPU-only envs
    _NUMBA_AVAILABLE = False

if TYPE_CHECKING:
    from my_vllm.config.vllm import VllmConfig


logger = logging.getLogger(__name__)


def _require_numba() -> None:
    if not _NUMBA_AVAILABLE:
        raise ImportError(
            "numba is required for the NgramProposer (training-free "
            "speculative decoding). Install with `pip install numba`. "
            "vLLM pins numba==0.65.0 in requirements/cuda.txt."
        )


class NgramProposer:
    """Prompt-lookup n-gram speculative decoder (training-free).

    Mirrors vLLM V1 vllm/v1/spec_decode/ngram_proposer.py:NgramProposer.

    Does not load a draft model — the "draft tokens" are read directly
    from the prompt + previously-generated tokens of the same request.
    """

    def __init__(self, vllm_config: "VllmConfig"):
        _require_numba()
        spec_config = vllm_config.speculative_config
        assert spec_config is not None, (
            "NgramProposer requires VllmConfig.speculative_config to "
            "be set (method='ngram')."
        )
        assert spec_config.prompt_lookup_min is not None, (
            "prompt_lookup_min must be set for ngram proposer "
            "(SpeculativeConfig.__post_init__ defaults it to "
            "prompt_lookup_max if missing)."
        )
        assert spec_config.prompt_lookup_max is not None, (
            "prompt_lookup_max must be set for ngram proposer."
        )

        # Minimum length of the n-gram to match.
        self.min_n: int = spec_config.prompt_lookup_min
        # Maximum length of the n-gram to match.
        self.max_n: int = spec_config.prompt_lookup_max
        # Number of speculative tokens to draft per step (the k in
        # "match an n-gram, return k tokens after it").
        self.k: int = spec_config.num_speculative_tokens
        # Maximum length of the model (used to clip draft size when
        # the context is already near the end of the model's window).
        self.max_model_len: int = vllm_config.model_config.max_model_len

        # Pre-allocate buffers for the Numba batch-propose output.
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.valid_ngram_draft = np.zeros((max_num_seqs, self.k), dtype=np.int32)
        self.valid_ngram_num_drafts = np.zeros((max_num_seqs,), dtype=np.int32)

        # Threshold of total tokens in the batch to enable multi-
        # threading in Numba batch propose.
        self.num_tokens_threshold = 8192
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        cpu_count = os.cpu_count()
        if cpu_count:
            # Divide by 2 to use physical cores and not logical
            # (hyper-threaded) cores.  Cap at 1 thread per TP rank
            # (TODO in vLLM: bump cap to 8 when TP-parallel ngram
            # lands).
            self.num_numba_thread_available = min(1, (cpu_count // 2))
            self.num_numba_thread_available //= tp_size
        else:
            self.num_numba_thread_available = 1

        # Trigger Numba JIT compilation for N-gram proposer.
        # First call takes <1s; doing it in __init__ avoids the
        # first-step latency hit in production.
        self.propose(
            self.k,
            [[]] * 1024,
            np.zeros(1024, dtype=np.int32),
            np.zeros((1024, self.max_model_len), dtype=np.int32),
        )
        logger.info(
            "NgramProposer initialized: min_n=%d max_n=%d k=%d "
            "max_model_len=%d num_threads=%d",
            self.min_n,
            self.max_n,
            self.k,
            self.max_model_len,
            self.num_numba_thread_available,
        )

    def batch_propose(
        self,
        num_requests: int,
        valid_ngram_requests: list,
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
        k: int,
    ) -> list[list[int]]:
        """Batch version of n-gram proposer using Numba for acceleration.

        Args:
            num_requests: total number of requests in the batch.
            valid_ngram_requests: indices of requests that need ngram
                proposals (skipped if context is empty or already at
                ``max_model_len``).
            num_tokens_no_spec: ``(batch_size,)`` int32 array — number
                of context tokens (prompt + generated, **excluding**
                draft slots) for each request.
            token_ids_cpu: ``(batch_size, max_model_len)`` int32 array
                — full per-request token sequence, padded.
            k: number of speculative tokens to propose.

        Returns:
            ``list[list[int]]`` of length ``num_requests`` — draft
            token IDs for each request, ``[]`` for skipped requests.
        """
        _require_numba()
        draft_token_ids: list[list[int]] = []

        # Only run batch_propose if there are requests needing ngram
        # proposals.  Avoid calling the Numba function with an empty
        # list (ValueError: cannot compute fingerprint of empty list).
        if num_ngram_requests := len(valid_ngram_requests):
            original_num_numba_threads = get_num_threads()
            total_tokens = int(np.sum(num_tokens_no_spec))
            if total_tokens >= self.num_tokens_threshold:
                final_num_threads = max(
                    1, min(self.num_numba_thread_available, num_ngram_requests)
                )
                set_num_threads(final_num_threads)
            else:
                set_num_threads(1)

            batch_propose_numba(
                valid_ngram_requests,
                num_tokens_no_spec,
                token_ids_cpu,
                self.min_n,
                self.max_n,
                self.max_model_len,
                k,
                self.valid_ngram_draft,
                self.valid_ngram_num_drafts,
            )
            set_num_threads(original_num_numba_threads)

        for i in range(num_requests):
            if i in valid_ngram_requests and self.valid_ngram_num_drafts[i] > 0:
                draft_token_ids.append(
                    self.valid_ngram_draft[i, : self.valid_ngram_num_drafts[i]].tolist()
                )
            else:
                draft_token_ids.append([])

        return draft_token_ids

    def propose(
        self,
        num_speculative_tokens: int,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
        slot_mappings: dict | list[dict] | None = None,  # unused for ngram
    ) -> list[list[int]]:
        """Propose draft tokens for a batch of requests.

        Args:
            num_speculative_tokens: number of speculative tokens to
                draft this step (``<= self.k``).
            sampled_token_ids: per-request list of sampled tokens from
                the previous step.  Empty list → skip spec decode
                for that request (e.g. fresh prefill).
            num_tokens_no_spec: ``(batch_size,)`` int32 — context
                length per request (prompt + generated).
            token_ids_cpu: ``(batch_size, max_model_len)`` int32 —
                per-request token sequence.
            slot_mappings: unused (ngram does not need KV slot
                mapping; kept for signature parity with vLLM's
                ``propose()`` contract).

        Returns:
            ``list[list[int]]`` of draft tokens per request.
        """
        _require_numba()
        assert num_speculative_tokens <= self.k, (
            f"num_speculative_tokens={num_speculative_tokens} exceeds "
            f"NgramProposer.k={self.k}"
        )

        # Find which requests need ngram proposals.
        valid_ngram_requests = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                # Skip speculative decoding (no prior sampled tokens
                # means fresh prefill — nothing to match against).
                continue
            num_tokens = int(num_tokens_no_spec[i])
            if num_tokens >= self.max_model_len:
                # Already at max model length — no draft budget.
                continue
            valid_ngram_requests.append(i)

        return self.batch_propose(
            len(sampled_token_ids),
            valid_ngram_requests,
            num_tokens_no_spec,
            token_ids_cpu,
            num_speculative_tokens,
        )

    def load_model(self, *args, **kwargs):
        """No-op — ngram proposer doesn't load a draft model."""
        pass


# Numba JIT-compiled batch proposer.  ``parallel=True`` enables
# per-request prange parallel execution; the inner LPS loop runs in
# a single thread per request (Numba's automatic parallel-for over
# prange).
@njit(parallel=True)
def batch_propose_numba(
    valid_ngram_requests: list,
    num_tokens_no_spec: np.ndarray,
    token_ids_cpu: np.ndarray,
    min_n: int,
    max_n: int,
    max_model_len: int,
    k: int,
    valid_ngram_draft: np.ndarray,
    valid_ngram_num_drafts: np.ndarray,
):
    for i in prange(len(valid_ngram_requests)):
        idx = valid_ngram_requests[i]
        num_tokens = num_tokens_no_spec[idx]
        context_token_ids = token_ids_cpu[idx, :num_tokens]
        drafter_output = _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=context_token_ids,
            min_ngram=min_n,
            max_ngram=max_n,
            max_model_len=max_model_len,
            k=k,
        )
        valid_ngram_num_drafts[idx] = drafter_output.shape[0]
        if len(drafter_output):
            valid_ngram_draft[idx, : drafter_output.shape[0]] = drafter_output


from numba import jit as _jit  # local alias for the inner kernel decorator


@_jit(nopython=True)
def _find_longest_matched_ngram_and_propose_tokens(
    origin_tokens: np.ndarray,
    min_ngram: int,
    max_ngram: int,
    max_model_len: int,
    k: int,
) -> np.ndarray:
    """Find the longest n-gram matching the suffix and propose tokens.

    Algorithm:

    1. Reverse ``origin_tokens``.  The goal becomes: find the longest
       prefix of length ``[min_n, max_n]`` that appears as a suffix
       somewhere later in the reversed sequence.
    2. Walk through the reversed sequence with a KMP-style failure-
       function update (capped at ``max_ngram``).
    3. When the matched prefix length crosses ``min_ngram``, record
       the **earliest** (in original) / **latest** (in reversed)
       position where the new maximum was reached.
    4. Return ``origin_tokens[matched_end : matched_end + k]``
       (clipped to the available tail and to ``max_model_len``).
    """
    # No draft possible if the context is shorter than the min n-gram.
    total_token = origin_tokens.shape[0]
    if total_token < min_ngram:
        return np.empty((0,), dtype=origin_tokens.dtype)

    # No draft beyond the max model length.
    k = min(k, max_model_len - total_token)
    if k <= 0:
        return np.empty((0,), dtype=origin_tokens.dtype)

    # Flip tokens: now we look for the longest prefix that appears as
    # a suffix at some later position.
    tokens = origin_tokens[::-1]

    # LPS array — only the first max_ngram positions are stored (the
    # failure function is meaningless beyond the n-gram cap).
    lps = np.zeros(max_ngram, dtype=np.int32)

    longest_ngram = 0
    position = 0

    # KMP failure-function walk.
    prev_lps = 0
    i = 1
    while i < total_token:
        if tokens[prev_lps] == tokens[i]:
            prev_lps += 1
            # Record the new maximum match — at the **latest**
            # position in ``tokens`` (= earliest in ``origin_tokens``).
            if prev_lps >= longest_ngram:
                longest_ngram = prev_lps
                position = i
            if i < max_ngram:
                lps[i] = prev_lps
            if prev_lps == max_ngram:
                # Cap prev_lps at max_ngram: when we hit the cap, fall
                # back to the failure function at max_ngram-1 to keep
                # matching n-grams of exactly max_ngram length.
                prev_lps = lps[max_ngram - 1]
            i += 1
        elif prev_lps != 0:
            prev_lps = lps[prev_lps - 1]
        else:
            i += 1

    if longest_ngram < min_ngram:
        # No valid n-gram found.
        return np.empty((0,), dtype=origin_tokens.dtype)

    # Map back to original-token coordinates.
    # origin_tokens[total_token - 1 - position : total_token - 1 -
    #   position + longest_ngram] is the matched n-gram.
    # Draft starts immediately after.
    start_position = total_token - 1 - position + longest_ngram
    k = min(k, total_token - start_position)
    return origin_tokens[start_position : start_position + k]


__all__ = ["NgramProposer"]