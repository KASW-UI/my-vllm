import time

from my_vllm.runtime.tokenizer import load_tokenizer
from my_vllm.runtime.model_loader import load_model
from my_vllm.runtime.model import LLM


def test_kv_cache_speedup():
    prompt = "Once upon a time, in a distant land, there lived a wise"
    model_name = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m"
    gen_len = 30

    tokenizer = load_tokenizer(model_name)
    hf_model = load_model(model_name)
    llm = LLM(hf_model)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    t0 = time.time()
    output_ids = llm.generate(
        input_ids, max_new_tokens=gen_len
    )
    kv_time = time.time() - t0

    generated_text = tokenizer.decode(output_ids.tolist())
    assert len(output_ids) <= gen_len
    assert len(generated_text) > 0

    avg_ms_per_token = (kv_time / len(output_ids)) * 1000
    print(f"Prompt: {prompt!r} ({input_ids.shape[1]} tokens)")
    print(f"Generated {len(output_ids)} tokens in {kv_time:.2f}s "
          f"({avg_ms_per_token:.1f} ms/token)")
    print(f"Output: {prompt + generated_text!r}")
    assert avg_ms_per_token < 1000, (
        f"KV cache too slow: {avg_ms_per_token:.0f} ms/token"
    )
