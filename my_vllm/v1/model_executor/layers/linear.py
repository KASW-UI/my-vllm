"""Tensor Parallel linear layers (C013).

When world_size=1: degrade to regular nn.Linear.
When world_size>1: shard weights across GPUs, all-gather / all-reduce.
"""

import torch
import torch.nn as nn

from my_vllm.v1.distributed.parallel_state import (
    all_gather,
    all_reduce,
    get_tp_world_size,
    get_tp_rank,
)


class ColumnParallelLinear(nn.Module):
    """Linear with column-wise weight sharding.

    Weight [input, output] → shard along output dim.
    Forward: all_gather(local_w × input) to reconstruct full output.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
    ):
        super().__init__()
        tp_size = get_tp_world_size()
        tp_rank = get_tp_rank()
        self.tp_size = tp_size

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
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.weight.shape[0]
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_out = nn.functional.linear(x, self.weight, self.bias)
        if self.tp_size == 1:
            return local_out
        return all_gather(local_out, dim=-1)


class RowParallelLinear(nn.Module):
    """Linear with row-wise weight sharding.

    Weight [input, output] → shard along input dim.
    Each rank computes partial dot-product; all_reduce sums them.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
    ):
        super().__init__()
        tp_size = get_tp_world_size()
        tp_rank = get_tp_rank()
        self.tp_size = tp_size

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
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.weight.shape[0]
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_out = nn.functional.linear(x, self.weight)
        if self.tp_size == 1:
            return local_out + self.bias if self.bias is not None else local_out
        out = all_reduce(local_out)
        if self.bias is not None:
            out = out + self.bias
        return out
