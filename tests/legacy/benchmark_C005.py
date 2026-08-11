"""Benchmark: C004 (single request) vs C005 (static batch)"""

import time

from my_vllm.entrypoints.llm import load_tokenizer
from my_vllm.model_executor.model_loader import load_model
from my_vllm.entrypoints.llm import LLM
from my_vllm.v1.request import Request

MODEL = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m"
PROMPTS = [
    "The capital of France is",
    "Once upon a time in a land far away",
    "Artificial intelligence has transformed",
    "The history of computing begins with",
]
GEN_LEN = 40


def main():
    tokenizer = load_tokenizer(MODEL)
    warmup = load_model(MODEL)
    LLM(warmup)

    # ── Single ──
    singles = []
    for i, p in enumerate(PROMPTS):
        req = Request(request_id=f"s-{i}")
        req.prompt_token_ids = tokenizer.encode(p)
        singles.append(req)

    model_single = load_model(MODEL)
    llm_single = LLM(model_single)
    t0 = time.time()
    for req in singles:
        llm_single.generate(req, max_new_tokens=GEN_LEN)
    t_single = time.time() - t0
    tok_single = sum(len(r.output_token_ids) for r in singles)

    # ── Batch ──
    batches = []
    for i, p in enumerate(PROMPTS):
        req = Request(request_id=f"b-{i}")
        req.prompt_token_ids = tokenizer.encode(p)
        batches.append(req)

    model_batch = load_model(MODEL)
    llm_batch = LLM(model_batch)
    t0 = time.time()
    llm_batch.generate_batch(batches, max_new_tokens=GEN_LEN)
    t_batch = time.time() - t0
    tok_batch = sum(len(r.output_token_ids) for r in batches)

    speedup = t_single / t_batch if t_batch > 0 else float("inf")

    print(f"Model: {MODEL.split('/')[-1]}")
    print(f"Requests: {len(PROMPTS)}  |  Generate length: {GEN_LEN}")
    print()
    print("=" * 60)
    print(f"{'Method':<20} {'Tokens':>8} {'Total (s)':>10} {'ms/token':>10}")
    print("-" * 60)
    print(
        f"{'C004 (single × N)':<20} {tok_single:>8} {t_single:>10.2f} "
        f"{t_single / max(tok_single, 1) * 1000:>10.1f}"
    )
    print(
        f"{'C005 (batch)':<20} {tok_batch:>8} {t_batch:>10.2f} "
        f"{t_batch / max(tok_batch, 1) * 1000:>10.1f}"
    )
    print("=" * 60)
    print(f"Speedup: {speedup:.1f}×")
    print(f"Time saved: {(t_single - t_batch):.1f}s")


if __name__ == "__main__":
    main()
