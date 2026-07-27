import torch

from .tokenizer import load_tokenizer
from .model_loader import load_model
from .model import LLM


def run(
    prompt: str,
    model_name: str = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m",
):
    print(f"Loading model: {model_name}")
    tokenizer = load_tokenizer(model_name)

    print("Loading weights ...")
    hf_model = load_model(model_name)
    llm = LLM(hf_model)

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print(f"Input: {prompt!r} ({input_ids.shape[1]} tokens)")

    print("Running forward pass ...")
    logits = llm.forward(input_ids)

    print(f"Logits shape: {logits.shape}  (vocab_size={logits.shape[-1]})")
    print()

    last_logits = logits[0, -1, :]
    top_probs, top_indices = torch.topk(
        torch.softmax(last_logits, dim=-1), 5
    )
    print("Top-5 next-token predictions:")
    for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
        token_text = tokenizer.decode([idx.item()])
        print(f"  {i + 1}. {token_text!r:20s} (id={idx.item():5d}, prob={prob.item():.4f})")

    return logits


if __name__ == "__main__":
    run("Hello, my name is")
