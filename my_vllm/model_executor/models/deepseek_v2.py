"""Inference-only DeepSeek V2/V3 model — aligns with vLLM vllm/model_executor/models/deepseek_v2.py.

Architecture preserved (MHA path + MoE + shared experts):
  DeepseekAttention (MHA, like V1), DeepseekV2MLP (dense), DeepseekV2MoE,
  DeepseekV2DecoderLayer, DeepseekV2Model, DeepseekV2ForCausalLM.

Engineering details skipped: MLA, V32 sparse indexer, PP, EP, SP, LoRA,
Eagle/Eagle3, EPLB, quantization.
"""

from collections.abc import Iterable

import torch
from torch import nn
from transformers import PretrainedConfig

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
from my_vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    GateLinear,
    MoERunner,
)


class DeepseekAttention(nn.Module):
    """MHA-based self-attention (DeepSeek V1 / non-MLA path) — aligns with vLLM."""

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        max_position_embeddings: int = 8192,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads
        self.total_num_kv_heads = num_kv_heads or num_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads)

        head_dim = getattr(config, "head_dim", None) or getattr(
            config, "qk_rope_head_dim", None
        )
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
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
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

    def _init_rotary_emb(self, config: PretrainedConfig) -> None:
        is_neox_style = True
        rope_params = getattr(config, "rope_parameters", None)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_params is None and rope_scaling is not None:
            rope_type = rope_scaling.get("type", "default")
            if rope_type in ("yarn", "deepseek_yarn", "llama3", "dynamic"):
                rope_params = rope_scaling
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=rope_params,
            is_neox_style=is_neox_style,
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


class DeepseekV2MLP(nn.Module):
    """Dense MLP with SwiGLU — aligns with vLLM DeepseekV2MLP."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class DeepseekV2MoE(nn.Module):
    """Mixture-of-Experts — aligns with vLLM DeepseekV2MoE."""

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = getattr(config, "n_shared_experts", 0)
        self.routed_scaling_factor = getattr(
            config, "routed_scaling_factor", 1.0
        )
        top_k = config.num_experts_per_tok
        moe_inter = config.moe_intermediate_size

        # Router gate
        self.gate = GateLinear(
            input_size=self.hidden_size,
            output_size=self.n_routed_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )

        # Shared experts (optional)
        self.shared_experts = None
        if self.n_shared_experts > 0:
            self.shared_experts = DeepseekV2MLP(
                hidden_size=self.hidden_size,
                intermediate_size=(
                    moe_inter * self.n_shared_experts
                    if self.n_shared_experts > 1
                    else moe_inter
                ),
                prefix=f"{prefix}.shared_experts",
            )

        # FusedMoE factory
        self.experts: MoERunner = FusedMoE(
            num_experts=self.n_routed_experts,
            top_k=top_k,
            hidden_size=self.hidden_size,
            intermediate_size=moe_inter,
            renormalize=config.norm_topk_prob,
            use_grouped_topk=True,
            num_expert_group=getattr(config, "n_group", 1),
            topk_group=getattr(config, "topk_group", 1),
            scoring_func=getattr(config, "scoring_func", "softmax"),
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=getattr(
                self.gate, "e_score_correction_bias", None
            ),
            activation=config.hidden_act,
            prefix=f"{prefix}.experts",
            gate=self.gate,
            shared_experts=self.shared_experts,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits, _ = self.gate(hidden_states)
        return self.experts(hidden_states, router_logits)


class DeepseekV2DecoderLayer(nn.Module):
    """Decoder layer — aligns with vLLM DeepseekV2DecoderLayer."""

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = 0  # reserved
        self.n_routed_experts = getattr(config, "n_routed_experts", None)
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        first_k_dense_replace = getattr(config, "first_k_dense_replace", 0)

        # Attention (MHA path)
        self.self_attn = DeepseekAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
            max_position_embeddings=getattr(
                config, "max_position_embeddings", 8192
            ),
            prefix=f"{prefix}.self_attn",
        )

        # MLP or MoE (depends on config — we build both for testing)
        self.is_moe_layer = self.n_routed_experts is not None
        self.mlp = DeepseekV2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size
            if self.is_moe_layer
            else config.intermediate_size,
            hidden_act=config.hidden_act,
            prefix=f"{prefix}.mlp",
        )

        self.rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
        self.input_layernorm = RMSNorm(
            self.hidden_size, eps=self.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            self.hidden_size, eps=self.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions, hidden_states=hidden_states
        )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class DeepseekV2Model(nn.Module):
    """DeepSeek V2/V3 model — aligns with vLLM DeepseekV2Model."""

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        n_layers = config.num_hidden_layers
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        first_k_dense_replace = getattr(config, "first_k_dense_replace", 0)
        n_routed_experts = getattr(config, "n_routed_experts", None)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer_prefix = f"{prefix}.layers.{i}"

            # Determine if this is a MoE layer
            is_moe = (
                n_routed_experts is not None
                and i >= first_k_dense_replace
                and i % moe_layer_freq == 0
            )

            if is_moe:
                self.layers.append(
                    DeepseekV2MoEDecoderLayer(config, prefix=layer_prefix)
                )
            else:
                self.layers.append(
                    DeepseekV2DecoderLayer(config, prefix=layer_prefix)
                )

        self.norm = RMSNorm(
            config.hidden_size,
            eps=getattr(config, "rms_norm_eps", 1e-6),
        )

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


class DeepseekV2MoEDecoderLayer(nn.Module):
    """Decoder layer with MoE FFN — aligns with vLLM layer + DeepseekV2MoE."""

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = DeepseekAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
            max_position_embeddings=getattr(
                config, "max_position_embeddings", 8192
            ),
            prefix=f"{prefix}.self_attn",
        )

        self.mlp = DeepseekV2MoE(config, prefix=f"{prefix}.mlp")

        self.rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
        self.input_layernorm = RMSNorm(
            self.hidden_size, eps=self.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            self.hidden_size, eps=self.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions, hidden_states=hidden_states
        )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class DeepseekV2ForCausalLM(nn.Module):
    """DeepSeek V2/V3 for causal LM — aligns with vLLM DeepseekV2ForCausalLM."""

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.model = DeepseekV2Model(config, prefix=f"{prefix}.model")

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
        )
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

        self.logits_processor = LogitsProcessor(config.vocab_size)

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

    def _fuse_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        fused: dict[str, torch.Tensor] = {}
        qkv_keys: dict[str, list[tuple[str, str]]] = {}
        gate_up_keys: dict[str, list[tuple[int, str]]] = {}

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

        for base_key, parts in qkv_keys.items():
            sorted_parts = sorted(parts, key=lambda x: {"q": 0, "k": 1, "v": 2}[x[0]])
            fused[base_key] = torch.cat(
                [state_dict[k] for _, k in sorted_parts], dim=0
            )

        for base_key, parts in gate_up_keys.items():
            sorted_parts = sorted(parts, key=lambda x: x[0])
            fused[base_key] = torch.cat(
                [state_dict[k] for _, k in sorted_parts], dim=0
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
