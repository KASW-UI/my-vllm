"""Vocabulary parallel embedding — aligns with vLLM vllm/model_executor/layers/vocab_parallel_embedding.py.

Architecture preserved: VocabParallelEmbedding (PluggableLayer), ParallelLMHead,
UnquantizedEmbeddingMethod, VocabParallelEmbeddingShardIndices.

Engineering details skipped: quantization methods (always unquantized), LoRA layout,
BasevLLMParameter, set_weight_attrs, packed_dim, torch.compile.
"""

from dataclasses import dataclass

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

DEFAULT_VOCAB_PADDING_SIZE = 64


def divide(n: int, d: int) -> int:
    assert n % d == 0, f"{n} not divisible by {d}"
    return n // d


def pad_vocab_size(vocab_size: int, pad_to: int = DEFAULT_VOCAB_PADDING_SIZE) -> int:
    return ((vocab_size + pad_to - 1) // pad_to) * pad_to


# ---------------------------------------------------------------------------
# UnquantizedEmbeddingMethod
# ---------------------------------------------------------------------------

class UnquantizedEmbeddingMethod:
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
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("weight", weight)

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return F.linear(x, layer.weight, bias)

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_, layer.weight)

    def tie_weights(self, layer: nn.Module, embed_tokens: "VocabParallelEmbedding"):
        layer.weight = embed_tokens.weight
        return layer


# ---------------------------------------------------------------------------
# VocabParallelEmbeddingShardIndices
# ---------------------------------------------------------------------------

@dataclass
class VocabParallelEmbeddingShardIndices:
    """Indices for a shard of a vocab parallel embedding — aligns with vLLM."""

    padded_org_vocab_start_index: int
    padded_org_vocab_end_index: int
    padded_added_vocab_start_index: int
    padded_added_vocab_end_index: int
    org_vocab_start_index: int
    org_vocab_end_index: int
    added_vocab_start_index: int
    added_vocab_end_index: int

    @property
    def num_org_elements(self) -> int:
        return self.org_vocab_end_index - self.org_vocab_start_index

    @property
    def num_added_elements(self) -> int:
        return self.added_vocab_end_index - self.added_vocab_start_index

    @property
    def num_org_elements_padded(self) -> int:
        return self.padded_org_vocab_end_index - self.padded_org_vocab_start_index

    @property
    def num_added_elements_padded(self) -> int:
        return self.padded_added_vocab_end_index - self.padded_added_vocab_start_index

    @property
    def num_org_vocab_padding(self) -> int:
        return self.num_org_elements_padded - self.num_org_elements

    @property
    def num_added_vocab_padding(self) -> int:
        return self.num_added_elements_padded - self.num_added_elements

    @property
    def num_elements_padded(self) -> int:
        return self.num_org_elements_padded + self.num_added_elements_padded

    def __post_init__(self):
        assert self.padded_org_vocab_start_index <= self.padded_org_vocab_end_index
        assert self.padded_added_vocab_start_index <= self.padded_added_vocab_end_index
        assert self.org_vocab_start_index <= self.org_vocab_end_index
        assert self.added_vocab_start_index <= self.added_vocab_end_index


def _vocab_range_from_per_partition_size(
    per_partition_vocab_size: int, rank: int, offset: int = 0
) -> tuple[int, int]:
    index_f = rank * per_partition_vocab_size
    index_l = index_f + per_partition_vocab_size
    return index_f + offset, index_l + offset


def _vocab_range_from_global_size(
    global_vocab_size: int, rank: int, world_size: int, offset: int = 0
) -> tuple[int, int]:
    per_partition_vocab_size = divide(global_vocab_size, world_size)
    return _vocab_range_from_per_partition_size(
        per_partition_vocab_size, rank, offset=offset
    )


# ---------------------------------------------------------------------------
# VocabParallelEmbedding
# ---------------------------------------------------------------------------

class VocabParallelEmbedding(PluggableLayer):
    """Embedding parallelized in the vocabulary dimension — aligns with vLLM.

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        params_dtype: type of the parameters.
        org_num_embeddings: original vocabulary size (without LoRA).
        padding_size: padding size for the vocabulary.
        quant_config: quant config (ignored).
        prefix: full name of the layer in the state dict.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config: None = None,
        prefix: str = "",
    ):
        super().__init__()

        tp_rank = get_tp_rank()
        self.tp_size = get_tp_world_size()
        self.num_embeddings = num_embeddings
        self.padding_size = padding_size
        self.org_vocab_size = org_num_embeddings or num_embeddings
        num_added_embeddings = num_embeddings - self.org_vocab_size
        self.org_vocab_size_padded = pad_vocab_size(
            self.org_vocab_size, self.padding_size
        )
        self.num_embeddings_padded = pad_vocab_size(
            self.org_vocab_size_padded + num_added_embeddings, self.padding_size
        )
        assert self.org_vocab_size_padded <= self.num_embeddings_padded

        self.shard_indices = self._get_indices(
            self.num_embeddings_padded,
            self.org_vocab_size_padded,
            self.num_embeddings,
            self.org_vocab_size,
            tp_rank,
            self.tp_size,
        )
        self.embedding_dim = embedding_dim

        self.quant_method = UnquantizedEmbeddingMethod()

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.num_added_embeddings = self.num_embeddings - self.org_vocab_size
        self.num_embeddings_per_partition = divide(
            self.num_embeddings_padded, self.tp_size
        )
        assert (
            self.shard_indices.num_elements_padded == self.num_embeddings_per_partition
        )
        self.num_org_embeddings_per_partition = (
            self.shard_indices.org_vocab_end_index
            - self.shard_indices.org_vocab_start_index
        )
        self.num_added_embeddings_per_partition = (
            self.shard_indices.added_vocab_end_index
            - self.shard_indices.added_vocab_start_index
        )

        self.quant_method.create_weights(
            self,
            self.embedding_dim,
            [self.num_embeddings_per_partition],
            self.embedding_dim,
            self.num_embeddings_padded,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader,
        )

    @classmethod
    def _get_indices(
        cls,
        vocab_size_padded: int,
        org_vocab_size_padded: int,
        vocab_size: int,
        org_vocab_size: int,
        tp_rank: int,
        tp_size: int,
    ) -> VocabParallelEmbeddingShardIndices:
        num_added_embeddings_padded = vocab_size_padded - org_vocab_size_padded
        padded_org_vocab_start_index, padded_org_vocab_end_index = (
            _vocab_range_from_global_size(org_vocab_size_padded, tp_rank, tp_size)
        )
        padded_added_vocab_start_index, padded_added_vocab_end_index = (
            _vocab_range_from_global_size(
                num_added_embeddings_padded, tp_rank, tp_size, offset=org_vocab_size
            )
        )
        org_vocab_start_index = min(padded_org_vocab_start_index, org_vocab_size)
        org_vocab_end_index = min(padded_org_vocab_end_index, org_vocab_size)
        added_vocab_start_index = min(padded_added_vocab_start_index, vocab_size)
        added_vocab_end_index = min(padded_added_vocab_end_index, vocab_size)
        return VocabParallelEmbeddingShardIndices(
            padded_org_vocab_start_index,
            padded_org_vocab_end_index,
            padded_added_vocab_start_index,
            padded_added_vocab_end_index,
            org_vocab_start_index,
            org_vocab_end_index,
            added_vocab_start_index,
            added_vocab_end_index,
        )

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        start_idx = self.shard_indices.org_vocab_start_index
        shard_size = self.shard_indices.org_vocab_end_index - start_idx

        assert loaded_weight.shape[0] == self.org_vocab_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param[:loaded_weight.shape[0]].data.copy_(loaded_weight)
        param[loaded_weight.shape[0]:].data.fill_(0)

    def _get_masked_input_and_mask(
        self, input_: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        org_vocab_start = self.shard_indices.org_vocab_start_index
        org_vocab_end = self.shard_indices.org_vocab_end_index
        num_org_vocab_padding = self.shard_indices.num_org_vocab_padding
        added_vocab_start = self.shard_indices.added_vocab_start_index
        added_vocab_end = self.shard_indices.added_vocab_end_index

        org_vocab_mask = (input_ >= org_vocab_start) & (input_ < org_vocab_end)
        added_vocab_mask = (input_ >= added_vocab_start) & (input_ < added_vocab_end)
        added_offset = (
            added_vocab_start
            - (org_vocab_end - org_vocab_start)
            - num_org_vocab_padding
        )
        valid_offset = (org_vocab_start * org_vocab_mask) + (
            added_offset * added_vocab_mask
        )
        vocab_mask = org_vocab_mask | added_vocab_mask
        input_ = vocab_mask * (input_ - valid_offset)
        return input_, ~vocab_mask

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            masked_input, input_mask = self._get_masked_input_and_mask(input_)
        else:
            masked_input = input_
            input_mask = None

        output_parallel = self.quant_method.embedding(self, masked_input.long())

        if self.tp_size > 1:
            output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)

        output = all_reduce(output_parallel)
        return output

    def extra_repr(self) -> str:
        s = f"num_embeddings={self.num_embeddings_per_partition}"
        s += f", embedding_dim={self.embedding_dim}"
        s += f", org_vocab_size={self.org_vocab_size}"
        s += f", num_embeddings_padded={self.num_embeddings_padded}"
        s += f", tp_size={self.tp_size}"
        return s


# ---------------------------------------------------------------------------
# ParallelLMHead
# ---------------------------------------------------------------------------

class ParallelLMHead(VocabParallelEmbedding):
    """Parallelized LM head — aligns with vLLM ParallelLMHead.

    Output logits weight matrices used in the Sampler.

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        bias: whether to use bias.
        params_dtype: type of the parameters.
        org_num_embeddings: original vocabulary size.
        padding_size: padding size for the vocabulary.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config: None = None,
        prefix: str = "",
    ):
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype,
            org_num_embeddings,
            padding_size,
            quant_config,
            prefix,
        )
        self.quant_config = quant_config
        if bias:
            self.bias = Parameter(
                torch.empty(self.num_embeddings_per_partition, dtype=self.params_dtype),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

    def tie_weights(self, embed_tokens: VocabParallelEmbedding):
        return self.quant_method.tie_weights(self, embed_tokens)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        del input_
        raise RuntimeError("LMHead's weights should be used in the sampler.")
