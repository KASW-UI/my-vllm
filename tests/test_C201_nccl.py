"""test_C201_nccl.py — C201 NCCL benchmark smoke test.

Verifies that ``examples/nccl_benchmark.py`` runs end-to-end in the
degenerate single-GPU path (world_size=1), and that the algorithmic /
bus bandwidth formulas are correctly applied for ring all-reduce,
reduce-scatter, and all-gather.

For world_size=1, the bus bandwidth is ``0`` (the formula has
``(N-1) = 0``), and the algorithmic bandwidth equals ``size / t``
for all-gather / ``size/N / t`` for reduce-scatter (or ``0`` for
all-reduce since the data is summed into itself).  The smoke test
only checks shape and finiteness, not the BW values.

The multi-process path is gated behind a fixture so it only runs
when invoked with explicit env vars (similar to ``run.py``).  This
keeps the default test invocation fast and deterministic.
"""

import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_SCRIPT = REPO_ROOT / "examples" / "nccl_benchmark.py"


# ---------------------------------------------------------------------------
# 1. Single-GPU degenerate path
# ---------------------------------------------------------------------------

class TestSingleGPUDegenerate:
    """``examples/nccl_benchmark.py`` runs in world_size=1 mode (gloo
    backend) without crashing and produces a valid CSV with the
    expected shape."""

    def test_runs_and_produces_csv(self, tmp_path: Path):
        out_csv = tmp_path / "bench.csv"
        env = os.environ.copy()
        # No MASTER_ADDR / WORLD_SIZE → maybe_init_process_group falls
        # through to gloo single-rank init.
        env.pop("MASTER_ADDR", None)
        env.pop("MASTER_PORT", None)
        env.pop("WORLD_SIZE", None)
        env.pop("RANK", None)
        env.pop("LOCAL_RANK", None)

        result = subprocess.run(
            [sys.executable, str(BENCHMARK_SCRIPT),
             "--min-bytes", "1024", "--max-bytes", "65536",
             "--output", str(out_csv),
             "--warmup", "2", "--iters", "5"],
            capture_output=True, env=env, text=True, timeout=120,
        )
        assert result.returncode == 0, (
            f"benchmark failed:\nstdout={result.stdout}\nstderr={result.stderr}"
        )
        assert out_csv.exists(), "CSV output not produced"

        with out_csv.open() as f:
            rows = list(csv.DictReader(f))
        assert rows, "CSV is empty"
        # Each row must have the expected columns.
        expected_cols = {
            "op", "size_bytes", "num_elements", "dtype",
            "world_size", "rank", "elapsed_us",
            "alg_bw_GBps", "bus_bw_GBps",
        }
        actual_cols = set(rows[0].keys())
        assert expected_cols <= actual_cols, (
            f"missing columns: {expected_cols - actual_cols}"
        )

    def test_world_size_one_bw_zero_for_bus(self, tmp_path: Path):
        """At world_size=1, bus BW is mathematically 0; alg BW is the
        local memcpy rate (for all_gather) or 0 (for all_reduce /
        reduce_scatter)."""
        out_csv = tmp_path / "bench.csv"
        env = os.environ.copy()
        env.pop("MASTER_ADDR", None)
        env.pop("MASTER_PORT", None)
        env.pop("WORLD_SIZE", None)
        env.pop("RANK", None)
        env.pop("LOCAL_RANK", None)
        subprocess.run(
            [sys.executable, str(BENCHMARK_SCRIPT),
             "--min-bytes", "1024", "--max-bytes", "4096",
             "--output", str(out_csv),
             "--warmup", "2", "--iters", "5"],
            capture_output=True, env=env, text=True, timeout=120, check=True,
        )
        with out_csv.open() as f:
            rows = list(csv.DictReader(f))
        # Filter to non-error rows.
        ok_rows = [r for r in rows if float(r["elapsed_us"]) > 0]
        assert ok_rows, "no successful runs"
        for r in ok_rows:
            assert r["world_size"] == "1", (
                f"expected world_size=1, got {r['world_size']}"
            )
            assert float(r["bus_bw_GBps"]) == 0.0, (
                f"world_size=1 bus BW must be 0, got {r['bus_bw_GBps']} "
                f"(op={r['op']}, size={r['size_bytes']})"
            )

    def test_size_sweep_monotonic(self, tmp_path: Path):
        """For all_gather, larger messages should not run faster than
        smaller ones (sanity check on the timing methodology)."""
        out_csv = tmp_path / "bench.csv"
        env = os.environ.copy()
        env.pop("MASTER_ADDR", None)
        env.pop("MASTER_PORT", None)
        env.pop("WORLD_SIZE", None)
        env.pop("RANK", None)
        env.pop("LOCAL_RANK", None)
        subprocess.run(
            [sys.executable, str(BENCHMARK_SCRIPT),
             "--ops", "all_gather",
             "--dtypes", "float32",
             "--min-bytes", "1024", "--max-bytes", "4194304",
             "--output", str(out_csv),
             "--warmup", "2", "--iters", "5"],
            capture_output=True, env=env, text=True, timeout=180, check=True,
        )
        with out_csv.open() as f:
            rows = [
                r for r in csv.DictReader(f)
                if float(r["elapsed_us"]) > 0
            ]
        # Check elapsed_us is monotonically non-decreasing.
        prev = 0.0
        for r in rows:
            t = float(r["elapsed_us"])
            assert t >= prev * 0.8, (
                f"elapsed_us dropped sharply at size={r['size_bytes']}: "
                f"{prev:.1f} → {t:.1f} (timing noise > 20%)"
            )
            prev = t


# ---------------------------------------------------------------------------
# 2. Direct module invocation (smoke)
# ---------------------------------------------------------------------------

class TestBenchmarkModule:
    """Import the benchmark module and invoke ``make_sizes`` /
    ``run_sweep`` directly — useful for verifying the bandwidth
    formulas in isolation."""

    def test_make_sizes_geometric_sweep(self):
        sys.path.insert(0, str(REPO_ROOT / "examples"))
        try:
            from nccl_benchmark import make_sizes
        finally:
            sys.path.pop(0)
        sizes = make_sizes(1024, 4 * 1024 * 1024)
        assert sizes[0] == 1024
        assert sizes[-1] == 4 * 1024 * 1024
        # Each successive size is 2x previous.
        for a, b in zip(sizes, sizes[1:]):
            assert b == 2 * a, f"non-geometric: {a} → {b}"

    def test_bw_formulas(self):
        """Validate ring all-reduce / reduce-scatter / all-gather BW
        formulas at world_size=8 and a known message size."""
        sys.path.insert(0, str(REPO_ROOT / "examples"))
        try:
            from nccl_benchmark import _bw_all_reduce, _bw_reduce_scatter, _bw_all_gather
        finally:
            sys.path.pop(0)

        size_bytes = 64 * 1024 * 1024  # 64 MiB
        t = 0.01                       # 10 ms
        n = 8

        # all_reduce: alg = 7/8 * size / t
        alg, bus = _bw_all_reduce(size_bytes, t, n)
        assert abs(alg - (7 / 8) * size_bytes / t / 1e9) < 1e-6
        # bus = 2 * 7 * size / t
        assert abs(bus - 14 * size_bytes / t / 1e9) < 1e-6

        # reduce_scatter: alg = size / n / t
        alg, bus = _bw_reduce_scatter(size_bytes, t, n)
        assert abs(alg - size_bytes / n / t / 1e9) < 1e-6
        # bus = 7 * size / n / t
        assert abs(bus - 7 * size_bytes / n / t / 1e9) < 1e-6

        # all_gather: alg = size / t
        alg, bus = _bw_all_gather(size_bytes, t, n)
        assert abs(alg - size_bytes / t / 1e9) < 1e-6
        # bus = 7 * size / t
        assert abs(bus - 7 * size_bytes / t / 1e9) < 1e-6

        # world_size=1: bus BW must be 0 for all three ops.
        alg_ar, bus_ar = _bw_all_reduce(size_bytes, t, 1)
        assert bus_ar == 0.0
        alg_rs, bus_rs = _bw_reduce_scatter(size_bytes, t, 1)
        assert bus_rs == 0.0
        alg_ag, bus_ag = _bw_all_gather(size_bytes, t, 1)
        assert bus_ag == 0.0


# ---------------------------------------------------------------------------
# 3. Multi-process smoke (gated)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires ≥2 GPUs",
)
class TestMultiGPUOptional:
    """Optional multi-GPU smoke test.  Skipped on single-GPU machines
    (the default test environment).  Run with:

        MASTER_ADDR=127.0.0.1 MASTER_PORT=29600 WORLD_SIZE=2 RANK=0 \\
            pytest tests/test_C201_nccl.py::TestMultiGPUOptional -v
    """

    def test_two_gpu_run(self, tmp_path: Path):
        out_csv = tmp_path / "bench.csv"
        env = os.environ.copy()
        env["MASTER_ADDR"] = "127.0.0.1"
        env["MASTER_PORT"] = "29600"
        env["WORLD_SIZE"] = "2"
        env["RANK"] = "0"

        # Just check the script is invokable with multi-GPU env.
        # The actual 2-rank init requires torchrun; we only test
        # that the args are accepted.
        result = subprocess.run(
            [sys.executable, str(BENCHMARK_SCRIPT), "--help"],
            capture_output=True, env=env, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "all_reduce" in result.stdout
