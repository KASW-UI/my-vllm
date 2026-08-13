"""Model loader — loads vLLM-native models with weight loading.

Aligns with vLLM model_executor/model_loader/ architecture.
Loads the vLLM-native OPTForCausalLM, loads weights from the
HuggingFace checkpoint via load_weights(), and places the model
on the target device.
"""

import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from my_vllm.model_executor.models.opt import OPTForCausalLM

_LOCAL_MODEL_PATH = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m"


def load_model(model_name, dtype: str = "bfloat16"):
    if not isinstance(model_name, str):
        model_name = _LOCAL_MODEL_PATH
    model_path = model_name
    if not os.path.isdir(model_name):
        model_path = _LOCAL_MODEL_PATH

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    torch_dtype = getattr(torch, dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = OPTForCausalLM(config)
    model = model.to(device)
    model = model.to(torch_dtype)
    model.eval()

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, torch_dtype=torch_dtype,
        local_files_only=True,
    )
    weights = list(hf_model.state_dict().items())
    loaded = model.load_weights(weights)
    del hf_model

    num_expected = sum(1 for _ in model.state_dict().items())
    if len(loaded) < num_expected:
        import logging
        logger = logging.getLogger(__name__)
        missing = num_expected - len(loaded)
        logger.warning("Loaded %d/%d weights (%d missing — may be OK for tied embeddings)",
                       len(loaded), num_expected, missing)

    return model
