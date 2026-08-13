"""LoRA Triton ops package — placeholder.

Mirrors vLLM V1 lora/ops/triton_ops/.

C144: my-vllm's Dense-Linear LoRA path uses the pure-torch
implementation in `lora/ops/torch_ops/lora_ops.py`. The Triton
backend (SGMV / BGMV kernels) is out of scope per
docs/alignment-plan.md §8 ("Dense Linear 的一条 LoRA 前向
路径即可证明理解，不追求全 backend").

A follow-up can port the Triton SGMV / BGMV kernels from vLLM
when my-vllm's per-request adapter dispatch is wired up.
"""