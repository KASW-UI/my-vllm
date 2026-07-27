import torch

from my_vllm.runtime.tokenizer import load_tokenizer
from my_vllm.runtime.model_loader import load_model
from my_vllm.runtime.model import LLM


def test_forward():
    prompt = "Hello, my name is"
    model_name = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m"

    tokenizer = load_tokenizer(model_name)
    hf_model = load_model(model_name)
    llm = LLM(hf_model)

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    logits = llm.forward(input_ids)

    assert isinstance(logits, torch.Tensor)
    assert logits.ndim == 3
    assert logits.shape[0] == 1
    assert logits.shape[1] == input_ids.shape[1]
    assert logits.shape[2] == 50272

    last_logits = logits[0, -1, :]
    top_id = torch.argmax(last_logits).item()
    top_token = tokenizer.decode([top_id])
    assert isinstance(top_token, str) and len(top_token) > 0

    print(f"Forward pass OK: logits shape={logits.shape}, top token={top_token!r}")
