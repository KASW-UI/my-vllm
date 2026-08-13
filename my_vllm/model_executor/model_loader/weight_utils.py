"""Weight utilities — checkpoint → model state_dict streaming.

Mirrors vLLM V1 model_executor/model_loader/weight_utils.py:1.

C132: extracted from the former single-file
`model_executor/model_loader.py` into the package layout matching
vLLM. vLLM's `weight_utils.py` handles format dispatch
(safetensors / pytorch / sharded_state), HF downloads, and TP-aware
weight sharding; my-vLLM's slice handles only the
HF → OPT state_dict streaming used by `DefaultModelLoader`.
"""

import torch.nn as nn


def hf_state_dict_to_model_state(
    hf_model: nn.Module,
    model: nn.Module,
) -> list[str]:
    """Stream weights from an HF model's state_dict into our model.

    Args:
        hf_model: HuggingFace model with loaded weights
            (from `AutoModelForCausalLM.from_pretrained`).
        model: target my-vLLM model (e.g. `OPTForCausalLM`).

    Returns:
        List of weight names successfully loaded into `model`.
        Tied embeddings may produce a smaller list than the model's
        full state_dict, which is expected.
    """
    weights = list(hf_model.state_dict().items())
    loaded = model.load_weights(weights)
    return loaded