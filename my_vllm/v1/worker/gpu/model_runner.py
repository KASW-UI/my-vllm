"""GPUModelRunner — vLLM-canonical location.

Mirrors vLLM V1 v1/worker/gpu/model_runner.py.

C136: re-export shim. The concrete `GPUModelRunner` implementation
lives in `my_vllm.v1.worker.gpu_model_runner.py` (a sibling of
`gpu/`, not a child of it). vLLM's pattern is to keep the canonical
runner under `gpu/model_runner.py`; my-vllm's split keeps the
implementation in `gpu_model_runner.py` for backward compatibility
(5 callers reference it directly — see the search in C136 commit)
and provides this file as the vLLM-aligned import path.

Future refactors can move the full implementation body into this
file without changing any caller:

  from my_vllm.v1.worker.gpu.model_runner import GPUModelRunner  # vLLM path
  from my_vllm.v1.worker.gpu_model_runner import GPUModelRunner  # legacy path

both resolve to the same class via the re-export below.
"""

from my_vllm.v1.worker.gpu_model_runner import (
    InputBatch,
    GPUModelRunner,
    _str_to_torch_dtype,
)

__all__ = [
    "GPUModelRunner",
    "InputBatch",
    "_str_to_torch_dtype",
]