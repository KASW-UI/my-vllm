"""Legacy sampling bridge — re-exports from v1/sample.

C024: sampling logic moved to v1/sample/sampler.py (nn.Module).
This module kept for backward compatibility.
"""

import torch


def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    """Take the argmax of the last position's logits.

    Legacy function; prefer Sampler (nn.Module) from v1.sample.
    """
    return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
