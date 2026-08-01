"""Benchmark: C002 (no KV cache) vs C003 (with KV cache)"""

import time
import torch

from my_vllm.runtime.tokenizer import load_tokenizer
from my_vllm.runtime.model_loader import load_model
from my_vllm.generation.sampler import greedy_sample
from my_vllm.cache.kv_cache import KVCache

MODEL = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m"
PROMPT = (
    "In the field of artificial intelligence, large language models have "
    "revolutionized natural language processing. These models, based on the "
    "transformer architecture, use self-attention mechanisms to process text "
    "sequences. During inference, the model generates tokens one by one, "
    "conditioned on all previously generated tokens. This autoregressive "
    "process can be computationally expensive, especially for long sequences. "
    "To optimize this, KV cache techniques store the key and value tensors "
    "from previous steps, avoiding redundant recomputation. However, even with "
    "KV caching, serving multiple requests concurrently presents significant "
    "challenges in resource management and scheduling. Efficient memory "
    "allocation, batch processing, and request prioritization become critical "
    "for production systems serving thousands of users simultaneously. "
    "Modern serving systems like vLLM address these challenges through "
    "innovative techniques such as PagedAttention, which applies virtual "
    "memory and paging concepts to the attention mechanism, enabling "
    "efficient memory sharing and zero waste. This approach dramatically "
    "improves throughput while maintaining low latency, making it possible "
    "to serve large language models at scale with high efficiency."
)
GEN_LEN = 100
EOS = 2


def benchmark_c002_no_cache(model, input_ids):
    """C002 approach: forward full sequence every step."""
    generated = []
    cur = input_ids.to(model.device)
    t0 = time.time()
    for _ in range(GEN_LEN):
        logits = model(cur, use_cache=False).logits
        nxt = greedy_sample(logits).item()
        if nxt == EOS:
            break
        generated.append(nxt)
        cur = torch.cat([cur, torch.tensor([[nxt]], device=model.device)], dim=1)
    elapsed = time.time() - t0
    return len(generated), elapsed


def benchmark_c003_kv_cache(model, input_ids):
    """C003 approach: prefill + decode with KV cache."""
    generated = []
    cache = KVCache()
    t0 = time.time()
    logits = cache.prefill(model, input_ids.to(model.device))
    nxt = greedy_sample(logits).item()
    if nxt == EOS:
        return 0, time.time() - t0
    generated.append(nxt)
    for _ in range(1, GEN_LEN):
        logits = cache.decode(model, torch.tensor([nxt], device=model.device))
        nxt = greedy_sample(logits).item()
        if nxt == EOS:
            break
        generated.append(nxt)
    elapsed = time.time() - t0
    return len(generated), elapsed


def main():
    tokenizer = load_tokenizer(MODEL)
    hf_model = load_model(MODEL)
    input_ids = tokenizer.encode(PROMPT, return_tensors="pt")
    n_prompt = input_ids.shape[1]

    print(f"Model: {MODEL.split('/')[-1]}")
    print(f"Prompt: {PROMPT!r} ({n_prompt} tokens)")
    print(f"Generate length: {GEN_LEN}")
    print()

    # Warmup
    hf_model(input_ids=input_ids.to(hf_model.device), use_cache=False)
    hf_model(input_ids=input_ids.to(hf_model.device), use_cache=True)

    # C002 — no KV cache
    hf_model_copy = load_model(MODEL)
    n_c002, t_c002 = benchmark_c002_no_cache(hf_model_copy, input_ids)
    ms_per_token_c002 = (t_c002 / n_c002) * 1000 if n_c002 > 0 else 0

    # C003 — with KV cache
    n_c003, t_c003 = benchmark_c003_kv_cache(hf_model, input_ids)
    ms_per_token_c003 = (t_c003 / n_c003) * 1000 if n_c003 > 0 else 0

    speedup = t_c002 / t_c003 if t_c003 > 0 else float("inf")

    print("=" * 55)
    print(f"{'Method':<20} {'Tokens':>8} {'Total (s)':>10} {'ms/token':>10}")
    print("-" * 55)
    print(
        f"{'C002 (no KV cache)':<20} {n_c002:>8} {t_c002:>10.2f} {ms_per_token_c002:>10.1f}"
    )
    print(
        f"{'C003 (KV cache)':<20} {n_c003:>8} {t_c003:>10.2f} {ms_per_token_c003:>10.1f}"
    )
    print("=" * 55)
    print(f"Speedup: {speedup:.1f}x faster with KV cache")
    print(f"Total time reduction: {(t_c002 - t_c003):.1f}s")


if __name__ == "__main__":
    main()
