"""Inference-only LLaMA model — aligns with vLLM vllm/model_executor/models/llama.py.

Architecture preserved: LlamaMLP, LlamaAttention, LlamaDecoderLayer, LlamaModel,
LlamaForCausalLM. Layers: MergedColumnParallelLinear, QKVParallelLinear,
RowParallelLinear, RMSNorm, RotaryEmbedding, SiluAndMul, Attention, 
VocabParallelEmbedding, ParallelLMHead, LogitsProcessor.

Engineering details skipped: PP (IntermediateTensors, PPMissingLayer), LoRA,
Eagle/Eagle3, Quantization, AutoWeightsLoader, WeightsMapper.
"""

from collections.abc import Iterable

import torch
from torch import nn
from transformers import LlamaConfig

from my_vllm.model_executor.layers.activation import SiluAndMul
from my_vllm.model_executor.layers.attention import Attention
from my_vllm.model_executor.layers.layernorm import RMSNorm
from my_vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from my_vllm.model_executor.layers.logits_processor import LogitsProcessor
from my_vllm.model_executor.layers.rotary_embedding import get_rope
from my_vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)


class LlamaMLP(nn.Module):
    """LLaMA MLP with SwiGLU — aligns with vLLM LlamaMLP."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        bias: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                f"Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class LlamaAttention(nn.Module):
    """LLaMA self-attention with GQA and RoPE — aligns with vLLM LlamaAttention."""

    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 8192,
        bias: bool = False,
        bias_o_proj: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads)

        head_dim = getattr(config, "head_dim", None)
        self.head_dim = head_dim or self.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias_o_proj,
            prefix=f"{prefix}.o_proj",
        )

        self._init_rotary_emb(config)

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    def _init_rotary_emb(self, config: LlamaConfig) -> None:
        is_neox_style = True
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=getattr(config, "rope_parameters", None),
            is_neox_style=is_neox_style,
        )


class LlamaDecoderLayer(nn.Module):
    """LLaMA decoder layer with residual streaming — aligns with vLLM."""

    def __init__(
        self,
        config: LlamaConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False
        )
        bias_o_proj = attention_bias
        if hasattr(config, "qkv_bias"):
            attention_bias = config.qkv_bias

        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
            max_position_embeddings=max_position_embeddings,
            bias=attention_bias,
            bias_o_proj=bias_o_proj,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions, hidden_states=hidden_states
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LlamaModel(nn.Module):
    """LLaMA model — aligns with vLLM LlamaModel."""

    def __init__(
        self,
        config: LlamaConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, prefix=f"{prefix}.layers.{i}")
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, residual
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    """LLaMA for causal language modeling — aligns with vLLM LlamaForCausalLM.

    Supports weight loading from HuggingFace checkpoints with QKV fusion
    and gate_up fusion.
    """

    def __init__(
        self,
        config: LlamaConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config, prefix=f"{prefix}.model")

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            config.vocab_size, scale=logit_scale
        )

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

    def _fuse_weights(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Fuse QKV and gate_up weights from HF checkpoint."""
        fused: dict[str, torch.Tensor] = {}
        qkv_keys: dict[str, list[str]] = {}
        gate_up_keys: dict[str, list[str]] = {}

        for key in state_dict:
            if ".self_attn.q_proj." in key:
                base = key.replace(".q_proj", ".qkv_proj")
                qkv_keys.setdefault(base, []).append(("q", key))
            elif ".self_attn.k_proj." in key:
                base = key.replace(".k_proj", ".qkv_proj")
                qkv_keys.setdefault(base, []).append(("k", key))
            elif ".self_attn.v_proj." in key:
                base = key.replace(".v_proj", ".qkv_proj")
                qkv_keys.setdefault(base, []).append(("v", key))
            elif ".mlp.gate_proj." in key:
                base = key.replace(".gate_proj", ".gate_up_proj")
                gate_up_keys.setdefault(base, []).append((0, key))
            elif ".mlp.up_proj." in key:
                base = key.replace(".up_proj", ".gate_up_proj")
                gate_up_keys.setdefault(base, []).append((1, key))
            elif not any(
                key.startswith(p)
                for p in [
                    ".self_attn.q_proj",
                    ".self_attn.k_proj",
                    ".self_attn.v_proj",
                    ".mlp.gate_proj",
                    ".mlp.up_proj",
                ]
            ):
                fused[key] = state_dict[key]

        for base_key, components in qkv_keys.items():
            parts = {shard: state_dict[k] for shard, k in components}
            fused[base_key] = torch.cat(
                [parts["q"], parts["k"], parts["v"]], dim=0
            )

        for base_key, components in gate_up_keys.items():
            parts = {shard: state_dict[k] for shard, k in components}
            fused[base_key] = torch.cat(
                [parts[0], parts[1]], dim=0
            )

        return fused

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        state_dict = dict(weights)
        if not any("qkv_proj" in k for k in state_dict):
            state_dict = self._fuse_weights(state_dict)

        loaded: set[str] = set()
        model_state = self.state_dict()

        for name, param in model_state.items():
            if name in state_dict:
                param.data.copy_(state_dict[name])
                loaded.add(name)

        return loaded
