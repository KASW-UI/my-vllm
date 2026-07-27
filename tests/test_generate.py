from my_vllm.runtime.tokenizer import load_tokenizer
from my_vllm.runtime.model_loader import load_model
from my_vllm.runtime.model import LLM


def test_generate():
    prompt = "Once upon a time"
    model_name = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m"
    max_new_tokens = 20

    tokenizer = load_tokenizer(model_name)
    hf_model = load_model(model_name)
    llm = LLM(hf_model)

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = llm.generate(
        input_ids, max_new_tokens=max_new_tokens
    )

    assert output_ids.ndim == 1, f"Expected 1D tensor, got shape {output_ids.shape}"
    assert 1 <= len(output_ids) <= max_new_tokens, (
        f"Expected 1..{max_new_tokens} tokens, got {len(output_ids)}"
    )

    generated_text = tokenizer.decode(output_ids.tolist())
    assert len(generated_text) > 0

    print(f"Prompt: {prompt!r}")
    print(f"Generated ({len(output_ids)} tokens): {prompt + generated_text!r}")
