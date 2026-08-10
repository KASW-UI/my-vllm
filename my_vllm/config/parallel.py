"""ParallelConfig + Multi-GPU topology/cost model. Aligns with vLLM ParallelConfig.

C086: TP×PP×EP×CP combination analysis — topology layout, communication cost model,
compatibility matrix.
"""

from dataclasses import dataclass


@dataclass
class ParallelConfig:
    """Configuration for 4D parallelism.

    Aligns with vLLM ParallelConfig (simplified).
    """

    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1
    context_parallel_size: int = 1

    @property
    def world_size(self) -> int:
        return (
            self.tensor_parallel_size
            * self.pipeline_parallel_size
            * self.expert_parallel_size
            * self.context_parallel_size
        )


# --- Communication cost model ---

def communication_cost(
    tp: int = 1, pp: int = 1, ep: int = 1, cp: int = 1,
    hidden_dim: int = 4096,
    seq_len: int = 2048,
    num_heads: int = 32,
    head_dim: int = 128,
    top_k: int = 2,
    num_layers: int = 32,
    dtype_bytes: int = 2,
) -> dict[str, float]:
    """Estimate per-step communication volume in bytes.

    TP: O(hidden_dim × seqlen) per layer → frequent, small
    PP: O(hidden_dim × seqlen) per stage → infrequent, large
    EP: O(top_k × hidden_dim × seqlen) per MoE → occasional
    CP: O(num_heads × head_dim × seqlen/cp_size) → batch, medium
    """
    n = seq_len
    d = hidden_dim
    h = num_heads
    hd = head_dim
    k = top_k
    L = num_layers

    tp_cost = 2 * n * d * dtype_bytes * L * (tp - 1) / tp
    pp_cost = n * d * dtype_bytes * 2 * (pp - 1)
    ep_cost = n * d * dtype_bytes * k * 2 * (ep - 1) / ep
    cp_cost = 2 * h * hd * n * dtype_bytes * 2 / cp

    return {
        "tp_bytes": tp_cost,
        "pp_bytes": pp_cost,
        "ep_bytes": ep_cost,
        "cp_bytes": cp_cost,
        "total_bytes": tp_cost + pp_cost + ep_cost + cp_cost,
    }


# --- Compatibility matrix ---

COMPAT_MATRIX = {
    "TP×PP": "✅ Each PP stage internally uses TP",
    "TP×EP": "✅ Each EP rank internally uses TP",
    "TP×CP": "✅ Each CP rank internally uses TP",
    "PP×CP": "⚠️ Ring state sync needed across stages",
    "EP×CP": "⚠️ All-to-All overlap with ring communication is complex",
}

# --- Example topology (8 GPUs) ---

TOPOLOGY_8GPU_TP2_PP2_EP2 = """
TP=2, PP=2, EP=2:
  GPU 0,1: stage 0, expert_group 0 (TP=2)
  GPU 2,3: stage 1, expert_group 0 (TP=2)
  GPU 4,5: stage 0, expert_group 1 (TP=2)
  GPU 6,7: stage 1, expert_group 1 (TP=2)

Order: PP major, then EP, then TP.
GPU 0: pp_rank=0, ep_rank=0, tp_rank=0
GPU 1: pp_rank=0, ep_rank=0, tp_rank=1
GPU 2: pp_rank=1, ep_rank=0, tp_rank=0
...
"""
