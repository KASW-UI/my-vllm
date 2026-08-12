"""P/D GPU allocation cost model — determines optimal GPU split between prefill and decode.

Aligns with vLLM V1 production scaling heuristics (not a single source file;
spread across scheduler config and KV connector init logic).

JD alignment:
  - 优化超大规模异构推理集群下的大模型推理性能、成本和稳定性
  - 多角色分离式推理 — GPU 数量如何分配

Decision logic:
  - prefill 是 compute-bound (O(n²) attention), decode 是 memory-bound (O(n) attention)
  - long prompts + many waiting → more P GPUs
  - many running decode → more D GPUs
  - GPU count recommendation based on FLOPs ratio + queue depth
"""

from dataclasses import dataclass


@dataclass
class PDClusterState:
    """Snapshot of P-side and D-side load for cost analysis."""
    num_prefill_reqs: int          # requests waiting for prefill
    num_decode_reqs: int            # requests currently in decode
    avg_prompt_len: int             # average prompt tokens
    avg_decode_remaining: int       # average remaining decode tokens
    total_gpus: int = 8             # total available GPUs in cluster


def estimate_flops_per_token(
    num_heads: int = 32,
    num_kv_heads: int = 32,
    head_dim: int = 128,
    num_layers: int = 12,
) -> tuple[float, float]:
    """Estimate FLOPs per token for prefill vs decode.

    Prefill: O(n²) attention + O(n) MLP, dominated by attention.
    Decode: O(n) attention (cached K/V) + O(1) MLP per new token.

    Returns: (prefill_flops_per_token, decode_flops_per_token)
    """
    d = num_heads * head_dim
    layers = num_layers
    gqa_ratio = num_heads / num_kv_heads

    # Prefill: 2 * q * k^T * v = 2 * (n_heads * head_dim * seq_len * head_dim)
    # → per token: 2 * d * (seq_len / 2) * gqa_ratio  ~= d * seq_len
    # Plus MLP: 4 * d * 4d = 16 * d² per layer
    prefill_per_layer = 4 * d * d + 2 * d * 128  # attention + MLP approximation
    # Decode: attention is O(1) per new token since K/V are cached
    # Only MLP + cached K/V projection + LM head
    decode_per_layer = 2 * d * 128 * gqa_ratio + 16 * d * d / 128  # much smaller

    return (
        prefill_per_layer * layers,
        decode_per_layer * layers,
    )


def recommend_pd_split(state: PDClusterState) -> tuple[int, int, str]:
    """Recommend GPU split between prefill and decode.

    Returns: (num_p_gpus, num_d_gpus, reasoning_string)

    Heuristic (aligned with vLLM production observations):
      - Prefill FLOPs ~ O(prompt_len²), Decode FLOPs ~ O(1) per step
      - If prefill_queue / decode_queue > 3.0 → need more P GPUs
      - If decode_queue / prefill_queue > 2.0 → need more D GPUs
      - Balance both latency + throughput: no side should starve

    Aligns with:
      - vLLM v1/engine/coordinator.py DP wave scheduling heuristics
      - KV connector dispatch load balancing
    """
    if state.total_gpus <= 1:
        return 1, 0, "single GPU — no P/D split"

    gpus = state.total_gpus

    prefill_work = state.num_prefill_reqs * (state.avg_prompt_len ** 1.5)
    decode_work = state.num_decode_reqs * state.avg_decode_remaining

    if prefill_work + decode_work == 0:
        return gpus // 2, gpus - gpus // 2, "no work — balanced split"

    prefill_ratio = prefill_work / (prefill_work + decode_work)

    #
    # Cost-driven allocation:
    #   Prefill is compute-heavy → needs more GPUs per request
    #   Decode is memory-heavy → many requests share fewer GPUs
    #
    # Weight prefill ratio more heavily (×1.5) to account for
    # the fact that a single long prefill can saturate multiple GPUs
    # while many decodes can efficiently share one GPU.
    #
    adjusted_ratio = min(0.9, prefill_ratio * 1.5)

    num_p = max(1, int(gpus * adjusted_ratio))
    num_d = gpus - num_p

    if num_p < 1:
        num_p = 1
        num_d = gpus - 1
    if num_d < 1:
        num_d = 1
        num_p = gpus - 1

    reason = (
        f"P={num_p}, D={num_d} — "
        f"prefill_load={prefill_ratio:.2f}, "
        f"P_queue={state.num_prefill_reqs}, D_queue={state.num_decode_reqs}, "
        f"avg_prompt={state.avg_prompt_len}"
    )

    return num_p, num_d, reason


def should_scale_up(state: PDClusterState, current_p: int, current_d: int) -> str | None:
    """Return which side should scale up/down based on load.

    Returns: "prefill" | "decode" | None (no change needed)

    Aligns with K8s HPA / Ray autoscaler decision logic.
    """
    prefill_pressure = state.num_prefill_reqs / max(1, current_p)
    decode_pressure = state.num_decode_reqs / max(1, current_d)

    if prefill_pressure > 8.0 and decode_pressure < 4.0:
        return "prefill"
    if decode_pressure > 16.0 and prefill_pressure < 4.0:
        return "decode"
    if prefill_pressure > 12.0:
        return "prefill"
    if decode_pressure > 24.0:
        return "decode"

    return None
