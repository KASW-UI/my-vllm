"""Tensor Parallel linear layers (C013 → C050).

C013: world_size=1 degrade to regular nn.Linear.
C050: real torch.distributed all_gather/all_reduce via parallel_state.

Aligns with vLLM vllm/model_executor/layers/linear.py:
ColumnParallelLinear — shard output dim, all_gather to reconstruct.
RowParallelLinear — shard input dim, all_reduce to sum partials.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from my_vllm.v1.distributed.parallel_state import (
    all_gather,
    all_reduce,
    get_tp_world_size,
    get_tp_rank,
)


class ColumnParallelLinear(nn.Module):
    """Linear with column-wise weight sharding — aligns with vLLM.

    Weight [input, output] → each rank holds 1/tp_size of output cols.
    Forward: Y_i = X A_i → all_gather → Y = [Y_1, ..., Y_p].
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = True,
    ):
        super().__init__()
        tp_size = get_tp_world_size()
        self.tp_size = tp_size
        self.gather_output = gather_output

        output_per_partition = output_size // tp_size
        self.weight = nn.Parameter(
            torch.empty(output_per_partition, input_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(output_per_partition))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = self.weight.shape[0]
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_parallel = F.linear(x, self.weight, self.bias)
        if self.gather_output and self.tp_size > 1:
            return all_gather(output_parallel, dim=-1)
        return output_parallel


class RowParallelLinear(nn.Module):
    """Linear with row-wise weight sharding — aligns with vLLM.

    Weight [input, output] → each rank holds 1/tp_size of input rows.
    Forward: Y_i = X_i A_i → all_reduce sum → Y = Σ Y_i.
    Bias applied only on rank 0 (vLLM pattern).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        reduce_results: bool = True,
    ):
        super().__init__()
        tp_size = get_tp_world_size()
        tp_rank = get_tp_rank()
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        input_per_partition = input_size // tp_size
        self.weight = nn.Parameter(
            torch.empty(output_size, input_per_partition)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = self.weight.shape[0]
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_is_parallel:
            input_parallel = x
        else:
            chunk_size = x.shape[-1] // self.tp_size
            start = self.tp_rank * chunk_size
            input_parallel = x[..., start:start + chunk_size].contiguous()

        bias_ = None if (self.tp_rank > 0) else self.bias
        output_parallel = F.linear(input_parallel, self.weight, bias_)

        if self.reduce_results and self.tp_size > 1:
            return all_reduce(output_parallel)
        return output_parallel
