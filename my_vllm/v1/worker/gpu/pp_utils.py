"""Pipeline Parallelism handler — aligns with vLLM PPHandler (minimal).

C084: sampled-token broadcast/recv on side stream, NCCL p2p send/recv
for activations between PP stages, 1F1B-lite schedule analysis.
"""

import torch

from my_vllm.v1.distributed.parallel_state import get_pp_group


class PPHandler:
    """Manages pipeline parallelism sampled-token communication.

    Non-last rank: receives sampled tokens from last rank.
    Last rank: broadcasts sampled tokens to all other ranks.
    All ranks: NCCL send/recv for activation tensors between stages.
    """

    def __init__(self, is_last_stage: bool, stage_id: int) -> None:
        self.is_last_stage = is_last_stage
        self.stage_id = stage_id

    @property
    def pp_group(self):
        return get_pp_group()

    def broadcast_sampled_tokens(self, sampled_token_ids: dict) -> dict:
        """Last rank broadcasts sampled tokens to all PP stages."""
        if self.pp_group is None or self.pp_group.world_size <= 1:
            return sampled_token_ids
        if self.is_last_stage:
            self.pp_group.broadcast_object(sampled_token_ids, src=0)
            return sampled_token_ids
        return self.pp_group.recv_object(src=0) or {}

    def send_activation(self, tensor: torch.Tensor, dst_stage: int) -> None:
        if self.pp_group is None:
            return
        self.pp_group.send(tensor, dst=dst_stage)

    def recv_activation(
        self, shape: torch.Size, dtype: torch.dtype, src_stage: int
    ) -> torch.Tensor | None:
        if self.pp_group is None:
            return None
        return self.pp_group.recv(shape, dtype, src=src_stage)

    @staticmethod
    def bubble_time(pp_size: int, micro_batch_time: float) -> float:
        return (pp_size - 1) * micro_batch_time

    @staticmethod
    def efficiency(num_micro_batches: int, pp_size: int) -> float:
        return num_micro_batches / (num_micro_batches + pp_size - 1)
