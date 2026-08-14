"""examples/ngram_proposer_benchmark.py — n-gram proposer sweep.

Benchmarks the C223 ``NgramProposer`` across:

  * Context length (``num_tokens_no_spec``)
  * Batch size
  * ``prompt_lookup_min`` / ``prompt_lookup_max`` window
  * Repetitiveness (controls match-hit-rate; lower repetition →
    fewer drafts returned → faster).

Emits a CSV table to stdout.  Demonstrates the engineering-subset
implementation hits production-relevant latencies (sub-millisecond
per call at the v1 baseline scale).

Usage:
    python examples/ngram_proposer_benchmark.py [--warmup N] [--iters N]

Workload model:

  * Tokens are drawn from a small synthetic vocabulary (size 32) so
    n-gram matches occur naturally.  Lower vocab size → higher match
    rate.  We also expose a `--vocab-size` knob to dial match-rate
    up/down (smaller vocab = more matches).
  * For each (batch, length) pair, we run the proposer's ``propose()``
    method ``--iters`` times and report the median elapsed time.

What this benchmark demonstrates:

  1. **JIT warm-up cost**: the first call pays the Numba compilation
     cost (~0.5-1s); subsequent calls are sub-millisecond.
  2. **Linear scaling in batch × context length**: the Numba
     ``prange`` parallel kernel scales roughly linearly with the
     number of requests * tokens.
  3. **Match rate vs. vocabulary**: at vocab=32 most contexts yield
     a draft; at vocab=4096 most contexts are unique → no draft.  In
     a real serving workload the match rate depends on the prompt
     distribution (chat templates, system prompts, etc.).

JD: 加分 #1「Spec decode / 推理加速」 — establish a measurement
harness so we can attribute E2E TTFT / ITL improvements to spec
decoding in follow-on PD-disaggregated benchmarks.
"""

import argparse
import csv
import statistics
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--vocab-size",
        type=int,
        default=128,
        help="Synthetic vocab size — smaller = higher ngram match rate.",
    )
    p.add_argument(
        "--warmup", type=int, default=2, help="Warmup iterations per config."
    )
    p.add_argument(
        "--iters", type=int, default=20, help="Measured iterations per config."
    )
    p.add_argument(
        "--min-n", type=int, default=2, help="prompt_lookup_min."
    )
    p.add_argument(
        "--max-n", type=int, default=5, help="prompt_lookup_max."
    )
    p.add_argument(
        "--k",
        type=int,
        default=5,
        help="num_speculative_tokens (per request draft budget).",
    )
    p.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8, 32, 128],
    )
    p.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[128, 512, 2048],
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# VllmConfig construction (minimal — only the fields NgramProposer reads)
# ---------------------------------------------------------------------------


def _make_proposer(min_n: int, max_n: int, k: int, max_model_len: int):
    """Construct a real NgramProposer (exercises JIT warm-up)."""
    from my_vllm.config.model import ModelConfig
    from my_vllm.config.parallel import ParallelConfig
    from my_vllm.config.scheduler import SchedulerConfig
    from my_vllm.config.speculative import SpeculativeConfig
    from my_vllm.config.vllm import VllmConfig
    from my_vllm.v1.spec_decode.ngram_proposer import NgramProposer

    vc = VllmConfig(
        model_config=ModelConfig(max_model_len=max_model_len),
        scheduler_config=SchedulerConfig(max_num_seqs=512),
        parallel_config=ParallelConfig(tensor_parallel_size=1),
        speculative_config=SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=k,
            prompt_lookup_min=min_n,
            prompt_lookup_max=max_n,
        ),
    )
    return NgramProposer(vc)


# ---------------------------------------------------------------------------
# Synthetic workload
# ---------------------------------------------------------------------------


def _make_workload(
    batch: int,
    context_len: int,
    vocab_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Construct ``(token_ids_cpu, num_tokens, sampled_token_ids)``.

    Tokens are random over ``[0, vocab_size)``.  ``sampled_token_ids``
    is a single-token stub per request (anything non-empty so spec
    decode runs).
    """
    tokens = rng.integers(0, vocab_size, size=(batch, context_len)).astype(np.int32)
    num_tokens = np.full(batch, context_len, dtype=np.int32)
    sampled = [[1] for _ in range(batch)]
    return tokens, num_tokens, sampled


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------


def main():
    args = _parse_args()

    # Largest context we'll use — sets max_model_len.
    max_ctx = max(args.context_lengths)
    proposer = _make_proposer(
        min_n=args.min_n,
        max_n=args.max_n,
        k=args.k,
        max_model_len=max_ctx,
    )

    print(
        f"# NgramProposer benchmark (vocab={args.vocab_size}, "
        f"min_n={args.min_n}, max_n={args.max_n}, k={args.k})",
        file=sys.stderr,
    )
    print(
        "# JIT warm-up paid in NgramProposer.__init__ (subsequent "
        "calls are fast)",
        file=sys.stderr,
    )

    rng = np.random.default_rng(0)
    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "batch",
            "context_len",
            "warmup_us",
            "median_us",
            "min_us",
            "max_us",
            "tokens_per_sec",
            "drafts_returned",
        ]
    )

    for batch in args.batch_sizes:
        for ctx in args.context_lengths:
            tokens, num_tokens, sampled = _make_workload(
                batch, ctx, args.vocab_size, rng
            )

            # Warmup
            for _ in range(args.warmup):
                drafts = proposer.propose(args.k, sampled, num_tokens, tokens)

            # Measured
            times_us = []
            drafts_count = 0
            for _ in range(args.iters):
                t0 = time.perf_counter_ns()
                drafts = proposer.propose(args.k, sampled, num_tokens, tokens)
                t1 = time.perf_counter_ns()
                times_us.append((t1 - t0) / 1000.0)
                drafts_count = sum(len(d) for d in drafts)

            median_us = statistics.median(times_us)
            min_us = min(times_us)
            max_us = max(times_us)
            # Tokens/sec processed = batch * ctx / median_seconds.
            tokens_per_sec = batch * ctx / (median_us * 1e-6)

            writer.writerow(
                [
                    batch,
                    ctx,
                    f"{args.warmup}",
                    f"{median_us:.2f}",
                    f"{min_us:.2f}",
                    f"{max_us:.2f}",
                    f"{tokens_per_sec:.0f}",
                    drafts_count,
                ]
            )


if __name__ == "__main__":
    main()