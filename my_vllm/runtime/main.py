import torch

from .tokenizer import load_tokenizer
from .model_loader import load_model
from .model import LLM


def run(
    prompt: str,
    model_name: str = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m",
    max_new_tokens: int = 50,
):
    print(f"Loading model: {model_name}")
    tokenizer = load_tokenizer(model_name)

    print("Loading weights ...")
    hf_model = load_model(model_name)
    llm = LLM(hf_model)

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print(f"Input: {prompt!r} ({input_ids.shape[1]} tokens)")
    print(f"EOS token id: {tokenizer.eos_token_id}")
    print()

    print("Generating ...")
    output_ids = llm.generate(input_ids, max_new_tokens=max_new_tokens)

    generated_text = tokenizer.decode(output_ids.tolist())
    full_text = prompt + generated_text
    print(f"Output: {full_text!r}")
    print(f"Generated {len(output_ids)} tokens")

    return output_ids


if __name__ == "__main__":
    run("Hello, my name is")
