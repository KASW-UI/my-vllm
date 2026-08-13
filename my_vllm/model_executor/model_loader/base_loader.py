"""Model loader base — abstract loader interface.

Mirrors vLLM V1 model_executor/model_loader/base_loader.py:1.

C132: extracted from the former single-file
`model_executor/model_loader.py` into the package layout matching
vLLM (vllm/model_executor/model_loader/{__init__.py,base_loader.py,
default_loader.py,utils.py,weight_utils.py}).
"""

from abc import ABC, abstractmethod


class BaseModelLoader(ABC):
    """Abstract base for model loaders.

    Mirrors vLLM V1 BaseModelLoader (simplified — single
    `load_model` method; the full vLLM loader also has
    `download_model` and TP-aware weight sharding, neither of which
    are in scope for my-vLLM's single-process, single-GPU path).
    """

    @abstractmethod
    def load_model(self, model_name: str, dtype: str = "bfloat16"):
        """Load a model and return it ready for inference.

        Args:
            model_name: HF model id or local path.
            dtype: torch dtype name (e.g. "bfloat16", "float16").

        Returns:
            The constructed model on the target device in eval mode.
        """