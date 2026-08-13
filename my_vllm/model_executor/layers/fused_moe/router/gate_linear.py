"""GateLinear — router gate linear layer for MoE — aligns with vLLM."""

import torch
from torch.nn.parameter import Parameter

from my_vllm.model_executor.layers.linear import ReplicatedLinear
from my_vllm.model_executor.custom_op import PluggableLayer


@PluggableLayer.register("gate_linear")
class GateLinear(ReplicatedLinear):
    """Gate linear layer for MoE router — aligns with vLLM GateLinear."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        out_dtype: torch.dtype | None = None,
        params_dtype: torch.dtype | None = None,
        force_fp32_compute: bool = False,
        prefix: str = "",
    ):
        self.out_dtype = out_dtype or torch.float32
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        super().__init__(
            input_size,
            output_size,
            bias=bias,
            params_dtype=params_dtype,
            prefix=prefix,
        )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        return super().forward(x)
