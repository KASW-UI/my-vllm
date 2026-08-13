"""Tensor Parallel linear layers — aligns with vLLM vllm/model_executor/layers/linear.py.

Architecture preserved (class hierarchy, method signatures, forward logic, TP sharding, GQA):
  LinearBase (PluggableLayer) → ReplicatedLinear / ColumnParallelLinear /
  MergedColumnParallelLinear / QKVParallelLinear / RowParallelLinear

Engineering details skipped: quantization methods (always unquantized), weight_loader_v2,
BasevLLMParameter, block scale adjustments, marlin/bnb sharding, DCP, MinimaxM3.
"""

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from my_vllm.distributed.parallel_state import (
    all_gather,
    all_reduce,
    get_tp_rank,
    get_tp_world_size,
)
from my_vllm.model_executor.custom_op import PluggableLayer


def divide(n: int, d: int) -> int:
    """Divide n by d, asserting exact divisibility."""
    assert n % d == 0, f"{n} not divisible by {d}"
    return n // d


# --------------------------------------------------------------------------
# UnquantizedLinearMethod — always used, no quantization dispatch
# --------------------------------------------------------------------------

class UnquantizedLinearMethod:
    """Linear method without quantization — real vLLM architecture."""

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        total_output = sum(output_partition_sizes)
        weight = Parameter(
            torch.empty(total_output, input_size_per_partition, dtype=params_dtype)
        )
        layer.register_parameter("weight", weight)

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return F.linear(x, layer.weight, bias)


# --------------------------------------------------------------------------
# LinearBase — real vLLM base class for all linear layers
# --------------------------------------------------------------------------

class LinearBase(PluggableLayer):
    """Base linear layer — aligns with vLLM LinearBase.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure (ignored — always unquantized).
        prefix: Prefix for parameter names.
        return_bias: If true, return bias together with outputs in forward pass.
        disable_tp: If true, tensor parallelism will be disabled for this layer.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
        tp_rank: int | None = None,
        tp_size: int | None = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.has_bias = bias
        self.skip_bias_add = skip_bias_add
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.quant_config = quant_config
        self.prefix = prefix
        self.quant_method = UnquantizedLinearMethod()
        self.return_bias = return_bias
        self.disable_tp = disable_tp
        if disable_tp:
            self.tp_rank, self.tp_size = 0, 1
        else:
            self.tp_rank = tp_rank if tp_rank is not None else get_tp_rank()
            self.tp_size = tp_size if tp_size is not None else get_tp_world_size()


# --------------------------------------------------------------------------
# ReplicatedLinear — same weight on all ranks (no TP sharding)
# --------------------------------------------------------------------------

class ReplicatedLinear(LinearBase):
    """Replicated linear layer — aligns with vLLM ReplicatedLinear.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict.
        return_bias: If true, return bias together with outputs in forward pass.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = self.output_sizes
        else:
            self.output_partition_sizes = [output_size]

        super().__init__(
            input_size,
            output_size,
            bias,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

        self.quant_method.create_weights(
            self,
            self.input_size,
            self.output_partition_sizes,
            self.input_size,
            self.output_size,
            self.params_dtype,
            weight_loader=self.weight_loader,
        )

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size, dtype=self.params_dtype)
            )
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)
        assert param.size() == loaded_weight.size(), (
            f"Tried to load weights of size {loaded_weight.size()}"
            f" to a parameter of size {param.size()}"
        )
        param.data.copy_(loaded_weight)

    def forward(
        self,
        input_: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        bias = self.bias if not self.skip_bias_add else None
        output = self.quant_method.apply(self, input_, bias)
        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def extra_repr(self) -> str:
        s = f"in_features={self.input_size}"
        s += f", output_features={self.output_size}"
        s += f", bias={self.bias is not None}"
        return s


# --------------------------------------------------------------------------
# ColumnParallelLinear — output-dimension sharding
# --------------------------------------------------------------------------

class ColumnParallelLinear(LinearBase):
    """Linear layer with column parallelism — aligns with vLLM.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Args:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias.
        gather_output: If true, call all-gather on output.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict.
        return_bias: If true, return bias together with outputs in forward pass.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
        tp_rank: int | None = None,
        tp_size: int | None = None,
    ):
        if disable_tp:
            tp_rank, tp_size = 0, 1
        else:
            tp_rank = tp_rank if tp_rank is not None else get_tp_rank()
            tp_size = tp_size if tp_size is not None else get_tp_world_size()
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                divide(o, tp_size) for o in self.output_sizes
            ]

        super().__init__(
            input_size,
            output_size,
            bias,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

        self.gather_output = gather_output

        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=self.weight_loader,
        )

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size_per_partition, dtype=params_dtype)
            )
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)
        param_data = param.data
        if param is self.weight:
            shard_size = param_data.shape[0]
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def forward(
        self,
        input_: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        bias = self.bias if not self.skip_bias_add else None
        output_parallel = self.quant_method.apply(self, input_, bias)
        if self.gather_output and self.tp_size > 1:
            output = all_gather(output_parallel, dim=-1)
        else:
            output = output_parallel
        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def extra_repr(self) -> str:
        s = f"in_features={self.input_size}"
        s += f", output_features={self.output_size_per_partition}"
        s += f", bias={self.bias is not None}"
        s += f", tp_size={self.tp_size}"
        s += f", gather_output={self.gather_output}"
        return s


# --------------------------------------------------------------------------
# MergedColumnParallelLinear — multiple column-parallel weights fused
# --------------------------------------------------------------------------

class MergedColumnParallelLinear(ColumnParallelLinear):
    """Packed linear layers with column parallelism — aligns with vLLM.

    Similar to ColumnParallelLinear, but the weight matrix is concatenated
    along the output dimension. When the weight matrix is loaded, the
    different partitions are sharded separately. Used e.g. for gate_up_proj
    in SwiGLU MLPs.

    Args:
        input_size: input dimension of the linear layer.
        output_sizes: list of output dimensions of the linear layer.
        bias: If true, add bias.
        gather_output: If true, call all-gather on output.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict.
        return_bias: If true, return bias together with outputs in forward pass.
        disable_tp: If true, treat as Replicated.
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        self.output_sizes = output_sizes
        self.tp_size = get_tp_world_size() if not disable_tp else 1
        self.tp_rank = get_tp_rank() if not disable_tp else 0
        assert all(o % self.tp_size == 0 for o in output_sizes)
        super().__init__(
            input_size=input_size,
            output_size=sum(output_sizes),
            bias=bias,
            gather_output=gather_output,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | None = None,
    ):
        param_data = param.data
        if loaded_shard_id is None:
            if param is self.weight:
                shard_size = param_data.shape[0]
                start_idx = self.tp_rank * shard_size
                loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
            assert param_data.shape == loaded_weight.shape
            param_data.copy_(loaded_weight)
            return

        assert loaded_shard_id < len(self.output_sizes)
        shard_offset = sum(self.output_sizes[:loaded_shard_id])
        shard_size = self.output_sizes[loaded_shard_id]
        shard_offset //= self.tp_size
        shard_size //= self.tp_size

        if param is self.weight:
            param_data = param_data.narrow(0, shard_offset, shard_size)
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


# --------------------------------------------------------------------------
# QKVParallelLinear — GQA-aware QKV fused projection
# --------------------------------------------------------------------------

class QKVParallelLinear(ColumnParallelLinear):
    """Linear layers for the attention's QKV transformation — aligns with vLLM.

    The weight matrix is concatenated along the output dimension. The layer
    is parallelized along the head dimension. When the number of key/value
    heads is smaller than the number of query heads (GQA/MQA), the key/value
    head may be replicated while the query heads are partitioned.

    Args:
        hidden_size: input hidden state size of the transformer.
        head_size: size of each attention head.
        total_num_heads: total number of attention query heads.
        total_num_kv_heads: total number of attention key/value heads.
        bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict.
        return_bias: If true, return bias together with outputs in forward pass.
        disable_tp: If true, disable TP sharding.
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
        v_head_size: int | None = None,
    ):
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.v_head_size = v_head_size if v_head_size is not None else head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads

        tp_size = get_tp_world_size() if not disable_tp else 1
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1

        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,
            self.num_kv_heads * self.head_size * tp_size,
            self.num_kv_heads * self.v_head_size * tp_size,
        ]

        super().__init__(
            input_size=self.hidden_size,
            output_size=sum(self.output_sizes),
            bias=bias,
            gather_output=False,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

    def validate_shard_id(self, shard_id: Any) -> bool:
        """Validate that shard_id is one of {'q', 'k', 'v', None}.

        Mirrors vllm QKVParallelLinear.validate_shard_id (TypeIs narrowing
        omitted — my-vllm is unquantized-only).
        """
        if shard_id in {"q", "k", "v"} or shard_id is None:
            return True
        raise ValueError(
            "Shard id for QKVParallelLinear should be 'q', 'k', or 'v', "
            f"got shard id {shard_id}."
        )

    def _get_shard_offset_mapping(self, loaded_shard_id: str) -> int:
        return {
            "q": 0,
            "k": self.num_heads * self.head_size,
            "v": (self.num_heads + self.num_kv_heads) * self.head_size,
            "total": (self.num_heads + self.num_kv_heads) * self.head_size
            + self.num_kv_heads * self.v_head_size,
        }.get(loaded_shard_id, 0)

    def _get_shard_size_mapping(self, loaded_shard_id: str) -> int:
        return {
            "q": self.num_heads * self.head_size,
            "k": self.num_kv_heads * self.head_size,
            "v": self.num_kv_heads * self.v_head_size,
        }.get(loaded_shard_id, 0)

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ):
        param_data = param.data

        if loaded_shard_id is None:
            if param is self.weight:
                shard_size = param_data.shape[0]
                start_idx = self.tp_rank * shard_size
                loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
            assert param_data.shape == loaded_weight.shape
            param_data.copy_(loaded_weight)
            return

        assert loaded_shard_id in ("q", "k", "v")

        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)

        if param is self.weight:
            param_data = param_data.narrow(0, shard_offset, shard_size)
            if loaded_shard_id == "q":
                shard_rank = self.tp_rank
            else:
                shard_rank = self.tp_rank // self.num_kv_head_replicas
            loaded_weight = loaded_weight.narrow(
                0, shard_rank * shard_size, shard_size
            )

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[str]:
        """Per-layer weight loader — mirrors vllm QKVParallelLinear.load_weights.

        Iterates (name, loaded_weight) pairs; for each, dispatches to the
        Parameter's weight_loader with the loaded weight's `shard_id`
        attribute (q/k/v/None). The yielded names let the caller track
        which weights were consumed.

        Quantization paths (PackedColumnParameter, PerTensorScaleParameter,
        BlockQuantScaleParameter) are out of scope for my-vllm.
        """
        import logging
        logger = logging.getLogger(__name__)
        for name, loaded_weight in weights:
            shard_id = getattr(loaded_weight, "shard_id", None)
            self.validate_shard_id(shard_id)
            param: Parameter
            if "." in name:
                submodule, _, attr = name.rpartition(".")
                param = getattr(self.get_submodule(submodule), attr, self)
            else:
                param = getattr(self, name, self)
            if param is None and name == "bias":
                continue
            param.weight_loader(param, loaded_weight, shard_id)
            yield name


# --------------------------------------------------------------------------
# RowParallelLinear — input-dimension sharding
# --------------------------------------------------------------------------

class RowParallelLinear(LinearBase):
    """Linear layer with row parallelism — aligns with vLLM.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension.

    Args:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, input is already split across GPUs.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        reduce_results: If true, call all-reduce on output.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict.
        return_bias: If true, return bias together with outputs in forward pass.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        reduce_results: bool = True,
        quant_config: None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        self.tp_rank = get_tp_rank() if not disable_tp else 0
        self.tp_size = get_tp_world_size() if not disable_tp else 1
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size
        self.output_partition_sizes = [output_size]

        super().__init__(
            input_size,
            output_size,
            bias,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=self.weight_loader,
        )

        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError(
                "When not reduce the results, adding bias to the "
                "results can lead to incorrect results"
            )

        if bias:
            self.bias = Parameter(torch.empty(self.output_size, dtype=params_dtype))
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)
        param_data = param.data
        if param is self.weight:
            shard_size = param_data.shape[1]
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(1, start_idx, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def forward(
        self,
        input_: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        if self.input_is_parallel:
            input_parallel = input_
        else:
            chunk_size = input_.shape[-1] // self.tp_size
            start = self.tp_rank * chunk_size
            input_parallel = input_[..., start:start + chunk_size].contiguous()

        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method.apply(self, input_parallel, bias_)

        if self.reduce_results and self.tp_size > 1:
            output = all_reduce(output_parallel)
        else:
            output = output_parallel

        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def extra_repr(self) -> str:
        s = f"in_features={self.input_size_per_partition}"
        s += f", output_features={self.output_size}"
        s += f", bias={self.bias is not None}"
        s += f", tp_size={self.tp_size}"
        s += f", reduce_results={self.reduce_results}"
        return s
