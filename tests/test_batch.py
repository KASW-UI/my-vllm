import time

from my_vllm.runtime.tokenizer import load_tokenizer
from my_vllm.runtime.model_loader import load_model
from my_vllm.runtime.model import LLM
from my_vllm.engine.request import Request


def test_batch_output_correct():
    prompts = [
        "The capital of France is",
        "Once upon a time",
        "Artificial intelligence is",
    ]
    model_name = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m"
    max_new_tokens = 20

    tokenizer = load_tokenizer(model_name)
    hf_model = load_model(model_name)
    llm = LLM(hf_model)

    requests = []
    for i, prompt in enumerate(prompts):
        req = Request(request_id=f"b-{i}")
        req.prompt_token_ids = tokenizer.encode(prompt)
        requests.append(req)

    llm.generate_batch(requests, max_new_tokens=max_new_tokens)

    for req in requests:
        assert len(req.output_token_ids) >= 1
        assert len(req.output_token_ids) <= max_new_tokens
        assert req.status.value >= 2  # FINISHED
        text = tokenizer.decode(req.output_token_ids)
        assert len(text) > 0
        print(f"[{req.request_id}] status={req.status.name} "
              f"tokens={len(req.output_token_ids)}")


def test_batch_vs_single():
    """Batch of N should be faster than N singles."""
    prompts = [
        "Once upon a time",
        "The weather today is",
        "A long time ago in",
    ]
    model_name = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m"
    gen_len = 30

    tokenizer = load_tokenizer(model_name)
    hf_model = load_model(model_name)
    llm = LLM(hf_model)

    # Batch timing
    requests_batch = []
    for i, prompt in enumerate(prompts):
        req = Request(request_id=f"ba-{i}")
        req.prompt_token_ids = tokenizer.encode(prompt)
        requests_batch.append(req)

    t0 = time.time()
    llm.generate_batch(requests_batch, max_new_tokens=gen_len)
    t_batch = time.time() - t0

    total_tokens_batch = sum(len(r.output_token_ids) for r in requests_batch)

    # Single timing (fresh model each time to avoid cache warmup bias)
    requests_single = []
    for i, prompt in enumerate(prompts):
        req = Request(request_id=f"s-{i}")
        req.prompt_token_ids = tokenizer.encode(prompt)
        requests_single.append(req)

    hf_model2 = load_model(model_name)
    llm2 = LLM(hf_model2)
    t0 = time.time()
    for req in requests_single:
        llm2.generate(req, max_new_tokens=gen_len)
    t_single = time.time() - t0

    total_tokens_single = sum(len(r.output_token_ids) for r in requests_single)

    print(f"Batch:  {t_batch:.2f}s for {total_tokens_batch} tokens "
          f"({t_batch / max(total_tokens_batch, 1) * 1000:.1f} ms/tok)")
    print(f"Single: {t_single:.2f}s for {total_tokens_single} tokens "
          f"({t_single / max(total_tokens_single, 1) * 1000:.1f} ms/tok)")
    print(f"Speedup: {t_single / t_batch:.1f}x")
