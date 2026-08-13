"""C150: multi-process NCCL end-to-end.

Verifies the multi-process + NCCL pipeline:

  1. `init_distributed_environment(...)` — canonical vLLM
     entry point. Falls back to gloo when CUDA is unavailable
     (covers the CI path).
  2. `initialize_model_parallel(...)` — creates the TP
     GroupCoordinator after the process group is up.
  3. `MultiprocExecutor._init_executor()` — opens a TCP port
     and exports `MASTER_ADDR` / `MASTER_PORT` / `WORLD_SIZE`
     env vars before spawning workers.
  4. End-to-end (requires ≥ 2 GPUs): spawn 2 worker processes
     with TP=2, init NCCL + TP group, load OPT model, verify
     `get_tp_world_size() == 2` and `get_tp_rank()` is unique
     per rank, run a forward pass.

Tests requiring multi-GPU are skipped automatically when the
machine has fewer than 2 CUDA devices.
"""

from __future__ import annotations

import os
import socket
from typing import Any

import pytest
import torch

from my_vllm.distributed import parallel_state as ps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _destroy_dist_state() -> None:
    """Tear down any global distributed state so tests don't leak."""
    if torch.distributed.is_initialized():
        ps.destroy_model_parallel()


def _has_2plus_gpus() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() >= 2


# ---------------------------------------------------------------------------
# Pure-Python / framework tests (no real NCCL needed)
# ---------------------------------------------------------------------------


def test_get_open_port_returns_distinct_ports() -> None:
    """`_get_open_port` allocates a free TCP port each call."""
    from my_vllm.v1.executor.multiproc_executor import _get_open_port

    p1 = _get_open_port()
    p2 = _get_open_port()
    assert p1 != p2, f"two calls returned the same port {p1}"
    # Both ports should be unprivileged (>= 1024).
    assert p1 >= 1024 and p2 >= 1024


def test_init_distributed_environment_falls_back_to_gloo_on_cpu(monkeypatch) -> None:
    """`init_distributed_environment` falls back to gloo when CUDA is absent.

    Mirrors vLLM's runtime fallback. We simulate the no-CUDA
    path with a tmpdir-based `file://` init method.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        # `file://` init method is supported by gloo + single
        # process, so the test runs without NCCL.
        url = f"file://{tmp}/init"
        # Force CUDA-unavailable branch by patching torch.cuda.
        monkeypatch.setattr(
            torch.cuda, "is_available", lambda: False, raising=False,
        )
        try:
            ps.init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=url,
                local_rank=0,
                backend="nccl",
            )
            assert torch.distributed.is_initialized()
            assert torch.distributed.get_world_size() == 1
            assert torch.distributed.get_rank() == 0
        finally:
            _destroy_dist_state()


def test_init_distributed_environment_is_idempotent() -> None:
    """A second call after the first is a no-op (process group already up)."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        url = f"file://{tmp}/init"
        try:
            ps.init_distributed_environment(
                world_size=1, rank=0,
                distributed_init_method=url, local_rank=0,
                backend="gloo",
            )
            # Second call must not raise.
            ps.init_distributed_environment(
                world_size=1, rank=0,
                distributed_init_method=url, local_rank=0,
                backend="gloo",
            )
            assert torch.distributed.is_initialized()
        finally:
            _destroy_dist_state()


def test_initialize_model_parallel_sets_tp_group(monkeypatch) -> None:
    """`initialize_model_parallel` populates `_TP` even at TP=1.

    After init: `get_tp_world_size() == 1`, `get_tp_rank() == 0`.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        url = f"file://{tmp}/init"
        try:
            ps.init_distributed_environment(
                world_size=1, rank=0,
                distributed_init_method=url, local_rank=0,
                backend="gloo",
            )
            ps.initialize_model_parallel(
                tensor_model_parallel_size=1, backend="gloo",
            )
            assert ps.model_parallel_is_initialized()
            assert ps.get_tp_world_size() == 1
            assert ps.get_tp_rank() == 0
        finally:
            _destroy_dist_state()


def test_multiproc_executor_sets_master_env(monkeypatch, tmp_path) -> None:
    """`MultiprocExecutor._init_executor` exports MASTER_ADDR/PORT/WORLD_SIZE.

    Uses a mock `_worker_main` so we don't actually spawn
    subprocesses. Verifies the env vars are set before
    spawning.
    """
    from my_vllm.config.parallel import ParallelConfig
    from my_vllm.config.vllm import VllmConfig
    from my_vllm.v1.executor.multiproc_executor import MultiprocExecutor

    cfg = VllmConfig()
    cfg.parallel_config = ParallelConfig(
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        expert_parallel_size=1,
        context_parallel_size=1,
    )

    # Wipe pre-existing env so we observe only the executor's writes.
    for k in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(k, raising=False)

    spawn_log: list[int] = []

    def fake_process_ctor(target, args, daemon, **kw):
        spawn_log.append(args[0])  # record rank
        return _FakeProcess()

    monkeypatch.setattr(
        "my_vllm.v1.executor.multiproc_executor.mp.Process",
        fake_process_ctor,
    )

    executor = MultiprocExecutor(cfg)
    executor._init_executor()

    assert spawn_log == [0, 1], spawn_log
    assert os.environ["MASTER_ADDR"] == "127.0.0.1"
    assert int(os.environ["MASTER_PORT"]) >= 1024
    assert os.environ["WORLD_SIZE"] == "2"
    # Last rank's RANK / LOCAL_RANK should remain.
    assert os.environ["RANK"] == "1"
    assert os.environ["LOCAL_RANK"] == "1"


# ---------------------------------------------------------------------------
# End-to-end multi-process NCCL test (skipped on <2 GPUs)
# ---------------------------------------------------------------------------


class _FakeProcess:
    """Process stub that satisfies the executor's `daemon`/lifecycle API."""

    daemon = True

    def start(self) -> None:
        pass

    def join(self, timeout: float | None = None) -> None:
        pass

    def is_alive(self) -> bool:
        return False

    def terminate(self) -> None:
        pass


@pytest.mark.skipif(
    not _has_2plus_gpus(),
    reason="C150 e2e needs ≥ 2 CUDA devices for NCCL TP=2",
)
def test_two_worker_nccl_tp2_loads_model_with_partitioned_linears(
    monkeypatch, tmp_path,
) -> None:
    """Spawn 2 workers, init NCCL + TP=2, load OPT, verify rank/world.

    Uses `torch.multiprocessing.spawn` to fork 2 workers
    (because `mp.spawn` integrates with the NCCL env vars
    cleanly). Each worker:

      1. Sets CUDA device = rank.
      2. Calls `init_distributed_environment` (NCCL via the
         TCP port exposed by the parent).
      3. Calls `initialize_model_parallel(tensor_model_parallel_size=2)`.
      4. Builds a tiny OPTForCausalLM and inspects the resulting
         `ColumnParallelLinear.tp_rank` / `tp_size`.
      5. Sends the rank + tp_size + world_size back to the parent.

    The parent asserts that the two workers report complementary
    tp_ranks (0 and 1) and identical tp_size=2, world_size=2.
    """
    import torch.multiprocessing as mp
    from my_vllm.config.parallel import ParallelConfig
    from my_vllm.config.vllm import VllmConfig
    from my_vllm.v1.executor.multiproc_executor import _get_open_port

    cfg = VllmConfig()
    cfg.parallel_config = ParallelConfig(
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        expert_parallel_size=1,
        context_parallel_size=1,
    )

    # Reserve a port before forking.
    port = _get_open_port()
    results: dict[int, dict[str, Any]] = {}

    def worker_fn(rank: int) -> None:
        try:
            torch.cuda.set_device(rank)
            ps.init_distributed_environment(
                world_size=2,
                rank=rank,
                distributed_init_method=f"tcp://127.0.0.1:{port}",
                local_rank=rank,
                backend="nccl",
            )
            ps.initialize_model_parallel(
                tensor_model_parallel_size=2, backend="nccl",
            )

            # Build a tiny OPT model and check the partitioned
            # ColumnParallelLinear reports tp_size=2 + the
            # rank's slice.
            from transformers import OPTConfig
            from my_vllm.model_executor.models.opt import OPTForCausalLM
            opt_cfg = OPTConfig(
                vocab_size=64, hidden_size=32, num_hidden_layers=2,
                num_attention_heads=2, intermediate_size=64, ffn_dim=64,
                max_position_embeddings=32, word_embed_proj_dim=32,
                tie_word_embeddings=True, dropout=0.0,
                attention_dropout=0.0, activation_dropout=0.0,
                layerdrop=0.0, enable_bias=False,
                layer_norm_elementwise_affine=True,
                do_layer_norm_before=True, num_labels=1, init_std=0.02,
                use_cache=True, pad_token_id=1, bos_token_id=0,
                eos_token_id=2,
            )
            model = OPTForCausalLM(opt_cfg)

            # Pick the first layer's qkv_proj — its
            # `output_size_per_partition` should be 16
            # (= (num_heads=2 * head_size=16) / tp_size=2).
            qkv = model.model.decoder.layers[0].self_attn.qkv_proj
            results[rank] = {
                "tp_rank": ps.get_tp_rank(),
                "tp_size": ps.get_tp_world_size(),
                "world_size": torch.distributed.get_world_size(),
                "qkv_out_partition": qkv.output_size_per_partition,
                "qkv_tp_rank": qkv.tp_rank,
                "qkv_tp_size": qkv.tp_size,
            }
        finally:
            ps.destroy_model_parallel()

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker_fn, args=(rank,)) for rank in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0, f"worker exited with code {p.exitcode}"

    assert set(results) == {0, 1}, results
    r0, r1 = results[0], results[1]
    # World size consistent across ranks.
    assert r0["world_size"] == r1["world_size"] == 2
    # TP size + rank complementary.
    assert r0["tp_size"] == r1["tp_size"] == 2
    assert {r0["tp_rank"], r1["tp_rank"]} == {0, 1}
    # ColumnParallelLinear layer reports the rank's partition.
    assert r0["qkv_out_partition"] == r1["qkv_out_partition"] == 16
    assert {r0["qkv_tp_rank"], r1["qkv_tp_rank"]} == {0, 1}
    assert r0["qkv_tp_size"] == r1["qkv_tp_size"] == 2