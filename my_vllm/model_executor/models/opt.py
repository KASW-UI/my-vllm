"""OPT model — re-export from runtime/models/opt.py.

C031: model_executor/models/ namespace for model implementations.
Bridges to the existing runtime models.
"""

from my_vllm.runtime.models.opt import OPTForCausalLM  # noqa: F401
