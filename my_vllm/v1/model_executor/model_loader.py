"""Model loader — loads HuggingFace models.

Mirrors vLLM model_executor/model_loader/ architecture (educational
single-function simplification).  Real vLLM has a full package with
BaseModelLoader ABC and 10+ implementations.

C032: inlined implementation, no longer re-exports from runtime/.
"""

import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM

_LOCAL_MODEL_PATH = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m"


def load_model(model_name: str, dtype: str = "bfloat16"):
    model_path = model_name
    if not os.path.isdir(model_name):
        model_path = _LOCAL_MODEL_PATH

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    torch_dtype = getattr(torch, dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, torch_dtype=torch_dtype,
        local_files_only=True,
    )
    model = model.to(device)
    model.eval()
    return model
