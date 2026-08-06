import torch
from my_vllm.v1.distributed.parallel_state import init_distributed
from my_vllm.v1.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)


def test_tp_layers_world_size_1():
    """When world_size=1, TP layers degrade to regular nn.Linear."""
    init_distributed(world_size=1, rank=0)

    col = ColumnParallelLinear(16, 32, bias=True)
    x = torch.randn(2, 16)
    y = col(x)
    assert y.shape == (2, 32)

    row = RowParallelLinear(16, 32, bias=True)
    y2 = row(x)
    assert y2.shape == (2, 32)

    # Verify column parallel produces same output as regular linear
    # with the same weight
    lin = torch.nn.Linear(16, 32, bias=True)
    lin.weight.data.copy_(col.weight.data)
    lin.bias.data.copy_(col.bias.data)
    expected = lin(x)
    assert torch.allclose(y, expected, atol=1e-5)

    print("TP layers OK (world_size=1)")


def test_tp_untrained_forward():
    """Random weights should produce finite outputs."""
    init_distributed(world_size=1, rank=0)

    col = ColumnParallelLinear(64, 128)
    x = torch.randn(4, 64)
    y = col(x)
    assert torch.isfinite(y).all()
    assert y.shape == (4, 128)

    row = RowParallelLinear(64, 128)
    y2 = row(x)
    assert torch.isfinite(y2).all()
    assert y2.shape == (4, 128)
    print("TP forward produces finite outputs")
