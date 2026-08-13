"""Model loader utilities — model class registry and HF arch mapping.

Mirrors vLLM V1 model_executor/model_loader/utils.py:1.

C132: extracted from the former single-file
`model_executor/model_loader.py` into the package layout matching
vLLM. vLLM's `utils.py` also has `get_model_architecture` and
`get_architecture_class_name` for the broader registry; my-vLLM
keeps only the OPT entry as the only supported architecture.
"""

from my_vllm.model_executor.models.opt import OPTForCausalLM

_MODEL_REGISTRY: dict[str, type] = {
    "OPTForCausalLM": OPTForCausalLM,
}


def get_model_cls(model_type: str) -> type | None:
    """Return the model class registered for a given HF `model_type`.

    Args:
        model_type: HF config.model_type (e.g. "opt", "llama", ...).
            Currently only "opt" is mapped (to OPTForCausalLM);
            other values return None.

    Returns:
        The registered model class, or None if the architecture is
        not supported.
    """
    return _MODEL_REGISTRY.get(_arch_key(model_type))


def get_model_architecture(model_type: str) -> str | None:
    """Map HF `model_type` to the registered architecture key.

    Currently a pass-through (only "opt" is registered). Kept for
    API parity with vLLM's model loader utils.
    """
    if model_type in _MODEL_REGISTRY:
        return model_type
    return None


def _arch_key(model_type: str) -> str:
    """Canonicalise HF model_type into our registry key."""
    return "OPTForCausalLM" if model_type == "opt" else model_type