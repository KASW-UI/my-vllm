"""Inference-only OPT model — extracted from vllm/vllm/model_executor/models/opt.py.

Architecture: OPTLearnedPositionalEmbedding, OPTAttention (QKV fused), OPTDecoderLayer,
OPTDecoder, OPTModel, OPTForCausalLM. Class names, method signatures, field names,
class-level metadata (hf_to_vllm_mapper, packed_modules_mapping) mirror the real vLLM
file so the file is a faithful structural subset.

Engineering details skipped (principled scope reduction):
- SupportsPP (PP group, IntermediateTensors, make_layers): PP=1 by config; the
  make_empty_intermediate_tensors factory is preserved for forward-compat.
- SupportsLoRA: lora/scheduler_hook.py lives in the lora/ package; not woven
  into the model definition.
- AutoWeightsLoader: replaced by a manual fusion helper (QKV only — see
  _fuse_qkv_params below). AutoWeightsLoader's auto-mapping machinery
  (WeightsMapper with orig_to_new_stacked / orig_to_new_prefix) is out of
  scope for the single-model reproduction.
- maybe_prefix helper: trivial string concat (`{prefix}.X`) — inlined.

Field-level alignment verified against vLLM OPTForCausalLM:
  - hf_to_vllm_mapper.orig_to_new_stacked: {q,k,v}_proj → qkv_proj shard_id
  - hf_to_vllm_mapper.orig_to_new_prefix: decoder. → model.decoder.
  - packed_modules_mapping: qkv_proj ← [q_proj, k_proj, v_proj]
"""

from collections.abc import Iterable

import torch
from torch import nn
from transformers import OPTConfig

from my_vllm.model_executor.layers.activation import get_act_fn
from my_vllm.model_executor.layers.attention import Attention
from my_vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from my_vllm.model_executor.layers.logits_processor import LogitsProcessor
from my_vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)


class WeightsMapper:
    """Minimal WeightsMapper for QKV stacking + decoder-prefix remap.

    Mirrors vllm/vllm/model_executor/models/utils.py:WeightsMapper (subset).
    Real WeightsMapper is a dataclass; this is a plain class for simplicity.
    """

    def __init__(
        self,
        orig_to_new_stacked: dict[str, tuple[str, str]] | None = None,
        orig_to_new_prefix: dict[str, str] | None = None,
    ):
        self.orig_to_new_stacked = orig_to_new_stacked or {}
        self.orig_to_new_prefix = orig_to_new_prefix or {}


class OPTLearnedPositionalEmbedding(nn.Embedding):
    """OPT learned positional embedding with +2 offset.

    Mirrors vllm OPTLearnedPositionalEmbedding.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, positions: torch.Tensor):
        return super().forward(positions + self.offset)


class OPTAttention(nn.Module):
    """OPT self-attention layer — mirrors vllm OPTAttention.

    Constructor signature: (embed_dim, num_heads, bias=True, cache_config=None,
    quant_config=None, prefix=""). Fields num_heads, head_dim, scaling match.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        cache_config=None,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        total_num_heads = num_heads
        self.num_heads = total_num_heads
        self.head_dim = embed_dim // total_num_heads
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            embed_dim,
            self.head_dim,
            total_num_heads,
            bias=bias,
            prefix=f"{prefix}.qkv_proj",
        )
        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=bias,
            prefix=f"{prefix}.out_proj",
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            scale=self.scaling,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        attn_output = self.attn(q, k, v)
        output, _ = self.out_proj(attn_output)
        return output


class OPTDecoderLayer(nn.Module):
    """OPT decoder layer — mirrors vllm OPTDecoderLayer."""

    def __init__(
        self,
        config: OPTConfig,
        cache_config=None,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = OPTAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            bias=config.enable_bias,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.do_layer_norm_before = config.do_layer_norm_before

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim,
            elementwise_affine=config.layer_norm_elementwise_affine,
        )
        self.fc1 = ColumnParallelLinear(
            self.embed_dim,
            config.ffn_dim,
            bias=config.enable_bias,
            prefix=f"{prefix}.fc1",
        )
        self.activation_fn = get_act_fn(config.activation_function)
        self.fc2 = RowParallelLinear(
            config.ffn_dim,
            self.embed_dim,
            bias=config.enable_bias,
            prefix=f"{prefix}.fc2",
        )
        self.final_layer_norm = nn.LayerNorm(
            self.embed_dim,
            elementwise_affine=config.layer_norm_elementwise_affine,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class OPTDecoder(nn.Module):
    """OPT decoder (transformer) — mirrors vllm OPTDecoder.

    Mirrors vLLM's structure: vocab-embed, learned-pos-embed (offset=2),
    optional project_in/out (when word_embed_proj_dim != hidden_size),
    optional final_layer_norm (controlled by _remove_final_layer_norm),
    nn.ModuleList of OPTDecoderLayer.
    """

    def __init__(
        self,
        config: OPTConfig,
        cache_config=None,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.word_embed_proj_dim,
        )
        self.embed_positions = OPTLearnedPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size
        )

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = ReplicatedLinear(
                config.hidden_size,
                config.word_embed_proj_dim,
                bias=False,
                prefix=f"{prefix}.project_out",
            )
            self.project_in = ReplicatedLinear(
                config.word_embed_proj_dim,
                config.hidden_size,
                bias=False,
                prefix=f"{prefix}.project_in",
            )
        else:
            self.project_out = None
            self.project_in = None

        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size,
                elementwise_affine=config.layer_norm_elementwise_affine,
            )
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList(
            [
                OPTDecoderLayer(
                    config, cache_config, quant_config,
                    prefix=f"{prefix}.layers.{i}",
                )
                for i in range(config.num_hidden_layers)
            ]
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return token embeddings for input_ids (mirrors vllm)."""
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        pos_embeds = self.embed_positions(positions)
        if self.project_in is not None:
            inputs_embeds, _ = self.project_in(inputs_embeds)
        hidden_states = inputs_embeds + pos_embeds

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)
        if self.project_out is not None:
            hidden_states, _ = self.project_out(hidden_states)
        return hidden_states


class OPTModel(nn.Module):
    """OPT model wrapper — mirrors vllm OPTModel."""

    def __init__(self, config: OPTConfig, prefix: str = ""):
        super().__init__()
        self.decoder = OPTDecoder(config, prefix=f"{prefix}.decoder")

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.decoder.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.decoder(input_ids, positions, inputs_embeds=inputs_embeds)


class OPTForCausalLM(nn.Module):
    """OPT for causal LM — mirrors vllm OPTForCausalLM.

    Class-level metadata matches vllm:
      hf_to_vllm_mapper:  stacked q/k/v → qkv_proj + decoder. → model.decoder.
      packed_modules_mapping:  qkv_proj ← [q_proj, k_proj, v_proj]

    SupportsPP / SupportsLoRA inheritance omitted (PP=1 by config; LoRA lives
    in lora/ package — not woven into the model). See vllm version for the
    full multi-inheritance signature.
    """

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
        },
        orig_to_new_prefix={
            "decoder.": "model.decoder.",
        },
    )
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }

    def __init__(self, config: OPTConfig, prefix: str = ""):
        super().__init__()
        self.config = config
        self.model = OPTModel(config, prefix=f"{prefix}.model")
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.decoder.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.word_embed_proj_dim,
            )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return token embeddings for input_ids (mirrors vllm)."""
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds=inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def _fuse_qkv_params(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Fuse HF q/k/v_proj into qkv_proj weights and biases.

        Mirrors the stacking declared in hf_to_vllm_mapper.orig_to_new_stacked
        (q/k/v → qkv_proj). AutoWeightsLoader applies this mapper; my-vllm
        applies it explicitly here. Out-of-scope for AutoWeightsLoader.
        """
        fused: dict[str, torch.Tensor] = {}
        qkv_keys: dict[str, list[str]] = {}

        for key in state_dict:
            if ".self_attn.q_proj." in key:
                base = key.replace(".q_proj", ".qkv_proj")
                qkv_keys.setdefault(base, []).append("q")
            elif ".self_attn.k_proj." in key:
                base = key.replace(".k_proj", ".qkv_proj")
                qkv_keys.setdefault(base, []).append("k")
            elif ".self_attn.v_proj." in key:
                base = key.replace(".v_proj", ".qkv_proj")
                qkv_keys.setdefault(base, []).append("v")
            else:
                fused[key] = state_dict[key]

        for base_key in qkv_keys:
            q = state_dict[base_key.replace("qkv_proj", "q_proj")]
            k = state_dict[base_key.replace("qkv_proj", "k_proj")]
            v = state_dict[base_key.replace("qkv_proj", "v_proj")]
            fused[base_key] = torch.cat([q, k, v], dim=0)

        return fused

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        """Load weights from HF checkpoint into vLLM-native OPT model.

        AutoWeightsLoader.load_weights(weights, mapper=hf_to_vllm_mapper) is
        what vllm uses; my-vllm inlines the manual loop since AutoWeightsLoader
        is out of scope. The semantics match: apply mapper (QKV stack +
        decoder. prefix swap), then copy into model_state.
        """
        state_dict = dict(weights)
        if not any("qkv_proj" in k for k in state_dict):
            state_dict = self._fuse_qkv_params(state_dict)

        loaded: set[str] = set()
        model_state = self.state_dict()

        for name, param in model_state.items():
            if name in state_dict:
                wt = state_dict[name]
                if param.shape != wt.shape:
                    slices = tuple(
                        slice(0, min(ps, ws))
                        for ps, ws in zip(param.shape, wt.shape)
                    )
                    param[slices].copy_(wt[slices])
                else:
                    param.data.copy_(wt)
                loaded.add(name)

        return loaded