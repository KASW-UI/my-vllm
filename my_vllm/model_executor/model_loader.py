"""Model loader — loads HuggingFace models.

C032: inlined implementation, no longer re-exports from runtime/.
"""

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def load_model(model_name: str, dtype: str = "bfloat16"):
    config = AutoConfig.from_pretrained(model_name)
    torch_dtype = getattr(torch, dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, config=config, torch_dtype=torch_dtype,
    )
    model = model.to(device)
    model.eval()
    return model
