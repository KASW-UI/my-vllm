"""C031 benchmark — measure prefill latency and decode throughput.

Runs 3 prompt lengths against opt-125m and reports tokens/sec.
"""

import time
import sys

from my_vllm.model_executor.model_loader import load_model
from my_vllm.entrypoints.llm import load_tokenizer

MODEL = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m"


def benchmark():
    tokenizer = load_tokenizer(MODEL)
    model = load_model(MODEL)
    device = model.device
    dtype = next(model.parameters()).dtype

    prompts = [
        ("short", "Hello world"),
        ("medium", "The capital of France is a city known for its history and culture"),
        ("long", "Artificial intelligence and deep learning have revolutionized "
          "the field of natural language processing in recent years, enabling "
          "models to understand and generate human language with remarkable "
          "accuracy and fluency across many tasks"),
    ]

    results = []
    for name, text in prompts:
        tokens = tokenizer.encode(text)
        if not isinstance(tokens, list):
            tokens = list(tokens)

        # Prefill benchmark
        import torch
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=True)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0) * 1000
        prefill_tok_ms = prefill_ms / len(tokens) if tokens else 0

        # Decode benchmark (5 steps)
        cache = out.past_key_values
        next_token = torch.tensor([[tokens[-1]]], dtype=torch.long, device=device)
        t0 = time.perf_counter()
        for _ in range(5):
            with torch.no_grad():
                out = model(input_ids=next_token, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(-1)
        torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - t0) * 1000
        decode_tok_s = 5 / (decode_ms / 1000) if decode_ms > 0 else 0

        results.append({
            "name": name,
            "prompt_tokens": len(tokens),
            "prefill_total_ms": round(prefill_ms, 2),
            "prefill_ms_per_token": round(prefill_tok_ms, 2),
            "decode_5_steps_ms": round(decode_ms, 2),
            "decode_tokens_per_sec": round(decode_tok_s, 1),
        })
        print(f"  {name}: {len(tokens)} tokens, prefill={prefill_ms:.1f}ms, decode={decode_tok_s:.1f} tok/s")

    return results


if __name__ == "__main__":
    import json
    data = benchmark()
    print("\n" + json.dumps(data, indent=2))
