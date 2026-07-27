import torch


def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    """Take the argmax of the last position's logits."""
    return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
