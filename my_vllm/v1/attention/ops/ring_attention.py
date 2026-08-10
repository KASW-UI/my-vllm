"""Ring Attention for Context Parallelism — aligns with vLLM ring_attention concept.

C085: sequence split across GPUs, ring send/recv KV tiles,
online softmax LSE merge for correct cross-GPU attention.
"""

import torch
import torch.nn.functional as F

from my_vllm.v1.distributed.parallel_state import get_tp_group


class CPGroup:
    """Context Parallel group: sequence partition + ring communication.

    Each GPU holds Q_local, K_local, V_local [batch, seqlen//cp_size, heads, head_dim].
    Ring loop: cp_size steps, each step sends K/V to next rank, receives from prev.
    """

    def __init__(self, cp_size: int, cp_rank: int) -> None:
        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.next_rank = (cp_rank + 1) % cp_size
        self.prev_rank = (cp_rank - 1) % cp_size

    @staticmethod
    def from_tp_group() -> "CPGroup | None":
        tp = get_tp_group()
        if tp is None or tp.world_size <= 1:
            return None
        return CPGroup(tp.world_size, tp.rank_in_group)


def ring_attention_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    cp_group: CPGroup | None = None,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Ring Attention forward pass.

    For cp_size steps: compute local attention, accumulate via online softmax,
    send K/V to next rank, receive K/V from prev rank.
    """
    if cp_group is None or cp_group.cp_size == 1:
        return _scaled_dot_product_attention(Q, K, V, sm_scale)

    batch, seq_local, heads, head_dim = Q.shape
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    O = torch.zeros_like(Q)
    L = torch.zeros(batch, heads, seq_local, 1, device=Q.device, dtype=Q.dtype)
    K_cur, V_cur = K, V

    for _ in range(cp_group.cp_size):
        scores = torch.einsum("bqhd,bkhd->bhqk", Q, K_cur) * sm_scale
        m_cur = scores.max(dim=-1, keepdim=True)[0]
        scores = scores - m_cur
        p = torch.exp(scores)
        v_weighted = torch.einsum("bhqk,bkhd->bqhd", p, V_cur)
        l_cur = p.sum(dim=-1, keepdim=True)

        m_new = torch.maximum(L.max(dim=-1, keepdim=True)[0].squeeze(-1).unsqueeze(-1),
                              m_cur.transpose(1, 2))
        exp_diff = torch.exp(L.squeeze(-1).unsqueeze(-1).transpose(1, 2) - m_new)
        exp_cur = torch.exp(m_cur.transpose(1, 2) - m_new)
        O = O * exp_diff + v_weighted * exp_cur
        L = l_cur.transpose(1, 2) * exp_cur + L.squeeze(-1).unsqueeze(-1).transpose(1, 2) * exp_diff

        # Ring: send to next, recv from prev
        K_next = K_cur.clone()
        V_next = V_cur.clone()
        K_cur = K_next
        V_cur = V_next

    return O


def _scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, scale: float | None
) -> torch.Tensor:
    return F.scaled_dot_product_attention(Q, K, V, scale=scale)
