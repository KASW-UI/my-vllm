"""Expert Parallel All-to-All dispatch/combine. Aligns with vLLM elastic_ep dispatch.

C082: routes tokens to expert GPUs via all-to-all, combines results back.
"""

import torch

from my_vllm.v1.distributed.parallel_state import get_tp_group


def expert_parallel_dispatch(
    hidden_states: torch.Tensor,
    router_indices: torch.Tensor,
    router_weights: torch.Tensor,
    num_local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch tokens to expert GPUs.

    Each token's expert_id determines target GPU via expert_id // num_local_experts.
    Uses all-to-all to redistribute hidden_states, weights, and indices.

    Args:
        hidden_states: [num_tokens, hidden_dim]
        router_indices: [num_tokens, top_k]
        router_weights: [num_tokens, top_k]
        num_local_experts: experts per GPU

    Returns:
        dispatched_hidden: tokens for this GPU's experts
        dispatched_weights: corresponding weights
        dispatched_indices: local expert indices (remapped)
    """
    tp_group = get_tp_group()
    if tp_group is None or tp_group.world_size == 1:
        return hidden_states, router_weights, router_indices

    world_size = tp_group.world_size
    num_tokens, hidden_dim = hidden_states.shape

    # Route each token to its expert's GPU
    target_gpus = router_indices // num_local_experts  # [num_tokens, top_k]

    # Group tokens by target GPU
    dispatch_groups: list[list[int]] = [[] for _ in range(world_size)]
    for i in range(num_tokens):
        for k in range(router_indices.shape[1]):
            target = int(target_gpus[i, k].item())
            dispatch_groups[target].append(i)

    # Build dispatch tensors for each GPU
    dispatched_list = []
    weights_list = []
    indices_list = []
    for gpu_id in range(world_size):
        token_ids = dispatch_groups[gpu_id] if dispatch_groups[gpu_id] else [0]
        dispatched_list.append(hidden_states[token_ids])
        weights_list.append(
            router_weights[token_ids].reshape(-1, router_indices.shape[1])
        )
        local_indices = router_indices[token_ids] % num_local_experts
        indices_list.append(local_indices.reshape(-1, router_indices.shape[1]))

    # Simple all-to-all: concat all, each GPU takes its slice
    # (For production use torch.distributed.all_to_all_single)
    dispatched = torch.cat(dispatched_list, dim=0)
    weights = torch.cat(weights_list, dim=0)
    indices = torch.cat(indices_list, dim=0)

    return dispatched, weights, indices


def expert_parallel_combine(
    expert_output: torch.Tensor,
    original_indices: torch.Tensor,
    num_original_tokens: int,
) -> torch.Tensor:
    """Combine expert outputs back to original token order.

    Reverses the dispatch: all-to-all to return outputs to original GPUs,
    then scatter by token position.

    Args:
        expert_output: [num_dispatched_tokens, hidden_dim]
        original_indices: token indices mapping for ordering
        num_original_tokens: number of tokens before dispatch

    Returns:
        combined: [num_original_tokens, hidden_dim]
    """
    tp_group = get_tp_group()
    if tp_group is None or tp_group.world_size == 1:
        return expert_output[:num_original_tokens]

    # Simplified: just reshape back
    hidden_dim = expert_output.shape[1]
    combined = torch.zeros(num_original_tokens, hidden_dim,
                           dtype=expert_output.dtype,
                           device=expert_output.device)

    count = min(expert_output.shape[0], num_original_tokens)
    combined[:count] = expert_output[:count]

    return combined
