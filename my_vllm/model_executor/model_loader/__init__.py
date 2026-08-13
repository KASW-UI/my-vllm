"""Model loader package — model construction + weight streaming.

Mirrors vLLM V1 model_executor/model_loader/.

C132: extracted from the former single-file
`model_executor/model_loader.py` into the package layout matching
vLLM (vllm/model_executor/model_loader/{__init__.py,base_loader.py,
default_loader.py,utils.py,weight_utils.py}).

Public API preserves the legacy `load_model(model_name, dtype)`
function so GPUModelRunner / model_states/default callers continue
to work without modification.
"""

from .base_loader import BaseModelLoader
from .default_loader import DefaultModelLoader
from .utils import get_model_architecture, get_model_cls
from .weight_utils import hf_state_dict_to_model_state


def load_model(model_name, dtype: str = "bfloat16"):
    """Load a model using the default loader.

    Convenience wrapper preserving the legacy single-file API used
    by `my_vllm.v1.worker.gpu_model_runner` and
    `my_vllm.v1.worker.gpu.model_states.default`.

    Args:
        model_name: HF model id or local path.
        dtype: torch dtype name (e.g. "bfloat16", "float16").

    Returns:
        The constructed model on the target device in eval mode.
    """
    return DefaultModelLoader().load_model(model_name, dtype)


__all__ = [
    "BaseModelLoader",
    "DefaultModelLoader",
    "get_model_cls",
    "get_model_architecture",
    "hf_state_dict_to_model_state",
    "load_model",
]