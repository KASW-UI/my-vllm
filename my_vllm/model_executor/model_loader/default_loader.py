"""Default model loader — HuggingFace weight streaming.

Mirrors vLLM V1 model_executor/model_loader/default_loader.py:1.

C132: extracted from the former single-file
`model_executor/model_loader.py` into the package layout matching
vLLM. The full vLLM `DefaultModelLoader` handles safetensors /
pytorch / sharded_state formats and TP-aware weight sharding; this
implementation handles the single-process HF→OPT path used by
my-vLLM's C132 baseline.
"""

import logging
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from my_vllm.model_executor.models.opt import OPTForCausalLM

from .base_loader import BaseModelLoader
from .weight_utils import hf_state_dict_to_model_state

_LOCAL_MODEL_PATH = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m"
logger = logging.getLogger(__name__)


class DefaultModelLoader(BaseModelLoader):
    """Load OPT (or other HF architecture) via state_dict streaming."""

    def load_model(self, model_name, dtype: str = "bfloat16"):
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
        loaded = hf_state_dict_to_model_state(hf_model, model)
        del hf_model

        num_expected = sum(1 for _ in model.state_dict().items())
        if len(loaded) < num_expected:
            missing = num_expected - len(loaded)
            logger.warning(
                "Loaded %d/%d weights (%d missing — may be OK for tied embeddings)",
                len(loaded), num_expected, missing,
            )

        return model