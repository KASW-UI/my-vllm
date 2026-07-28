"""Model loader — re-export from runtime/model_loader.py.

C031: model_executor/ namespace for model loading infrastructure.
Bridges to the existing runtime implementation.
"""

from my_vllm.runtime.model_loader import load_model  # noqa: F401
