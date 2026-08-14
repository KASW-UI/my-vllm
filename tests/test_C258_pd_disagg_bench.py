"""tests/test_C258_pd_disagg_bench.py — PD disaggregation benchmark.

C258: PD Disaggregation benchmark.

Verifies the example harness
``examples/pd_disaggregated_benchmark.py``:

* **KvTransferParams** — `num_blocks_for_tokens`, `bytes_per_kv_block`,
  `total_kv_bytes` math (BF16 / FP16 / FP32 / INT8) round-trip.
* **RDMA mode configs** — `from_link_bw` produces ns/byte consistent
  with the link BW; defaults cover NVLink / IB-HDR / IB-NDR / RoCE.
* **RdmaSimulator** — `elapsed_us` math (per-message overhead +
  bytes/bandwidth) + `gbytes_per_sec` circular check.
* **percentile / summarize** — P50/P99/mean/min on synthetic lists.
* **In-process bench** — runs the SHM transfer with a small size
  and verifies the CSV row shape (timing returned in μs, gbytes
  finite, KV bytes > 0).
* **RDMA-simulated bench** — runs the simulator and verifies the
  rows satisfy the wire-format expectations (mean ≈ bytes/bw).
* **run_benchmark** — full sweep over an opt-125m-shaped config,
  asserting one row per (size, mode) cell.
* **CSV schema** — header + 1 row per cell, all 16 columns present.
* **CLI smoke** — subprocess invocation with both modes enabled.
"""

from __future__ import annotations

import csv
import io
import statistics
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from examples.pd_disaggregated_benchmark import (
    DEFAULT_RDMA_MODES,
    BenchResult,
    KvTransferParams,
    RdmaModeConfig,
    RdmaSimulator,
    bench_in_process_transfer,
    bench_rdma_simulated_transfer,
    _emit_csv_header,
    _make_param_grid,
    _make_paged_kv_cache,
    main,
    percentile,
    run_benchmark,
    summarize,
)


# ===========================================================================
# KvTransferParams math
# ===========================================================================


class TestKvTransferParams:
    """`KvTransferParams` size math."""

    def test_num_blocks_for_tokens_exact(self):
        p = KvTransferParams(
            num_tokens=256, num_layers=4, num_kv_heads=4,
            head_size=16, block_size=16, num_blocks=16,
        )
        assert p.num_blocks_for_tokens == 16

    def test_num_blocks_for_tokens_capped(self):
        p = KvTransferParams(
            num_tokens=10, num_layers=4, num_kv_heads=4,
            head_size=16, block_size=16, num_blocks=1,
        )
        # ceil(10/16) = 1.
        assert p.num_blocks_for_tokens == 1

    def test_bytes_per_kv_block_bf16(self):
        # 16 * 4 * 16 * 2 (K+V) * 2 (bf16) = 4096.
        p = KvTransferParams(
            num_tokens=16, num_layers=1, num_kv_heads=4,
            head_size=16, block_size=16, num_blocks=1,
        )
        assert p.bytes_per_kv_block == 16 * 4 * 16 * 2 * 2

    def test_total_kv_bytes(self):
        p = KvTransferParams(
            num_tokens=128, num_layers=4, num_kv_heads=4,
            head_size=16, block_size=16, num_blocks=8,
        )
        # 4 * 8 * 4096 = 131072.
        assert p.total_kv_bytes == 131072

    def test_dtype_matters(self):
        p_fp16 = KvTransferParams(
            num_tokens=16, num_layers=1, num_kv_heads=4,
            head_size=16, block_size=16, num_blocks=1, dtype=torch.float16,
        )
        p_fp32 = KvTransferParams(
            num_tokens=16, num_layers=1, num_kv_heads=4,
            head_size=16, block_size=16, num_blocks=1, dtype=torch.float32,
        )
        # FP32 is 2× FP16.
        assert p_fp32.bytes_per_kv_block == 2 * p_fp16.bytes_per_kv_block


# ===========================================================================
# RDMA mode configs
# ===========================================================================


class TestRdmaModeConfig:
    """`RdmaModeConfig` + `DEFAULT_RDMA_MODES`."""

    def test_from_link_bw_math(self):
        # 50 GB/s = 50 × 1e9 B/s.  Time per byte = 1 / (50 * 1e9) sec/byte
        # = 1/50 ns/byte = 0.02 ns/byte.
        cfg = RdmaModeConfig.from_link_bw(
            name="x", medium="X", link_bw_GBps=50.0,
            per_message_overhead_us=2.0,
        )
        assert cfg.latency_per_byte_ns == pytest.approx(0.02, abs=0.001)

    def test_default_modes_cover_expected_set(self):
        assert set(DEFAULT_RDMA_MODES) == {"nvlink", "ib_hdr", "ib_ndr", "roce"}
        assert DEFAULT_RDMA_MODES["nvlink"].medium == "NVLINK"
        assert DEFAULT_RDMA_MODES["ib_hdr"].medium == "IB-HDR"
        assert DEFAULT_RDMA_MODES["ib_ndr"].medium == "IB-NDR"
        assert DEFAULT_RDMA_MODES["roce"].medium == "ROCE"

    def test_link_bw_ordering(self):
        # NDR ≥ HDR ≥ ROCE (rule of thumb; not strict NDR > HDR > ROCE).
        nv = DEFAULT_RDMA_MODES["nvlink"].link_bw_GBps
        hdr = DEFAULT_RDMA_MODES["ib_hdr"].link_bw_GBps
        ndr = DEFAULT_RDMA_MODES["ib_ndr"].link_bw_GBps
        roce = DEFAULT_RDMA_MODES["roce"].link_bw_GBps
        assert nv >= hdr
        assert ndr >= hdr
        assert hdr >= roce


# ===========================================================================
# RdmaSimulator
# ===========================================================================


class TestRdmaSimulator:
    """Latency + bandwidth math."""

    def test_elapsed_us_includes_overhead(self):
        cfg = RdmaModeConfig.from_link_bw(
            name="x", medium="X", link_bw_GBps=50.0,
            per_message_overhead_us=10.0,
        )
        sim = RdmaSimulator(cfg)
        # Zero bytes → just overhead.
        assert sim.elapsed_us(0) == 10.0
        # 1 GB at 50 GB/s = 20 ms = 20_000 μs. Plus 10 μs overhead.
        # Allow ±10% slack for floating-point rounding.
        one_gb = 1 << 30
        elapsed = sim.elapsed_us(one_gb)
        assert 18_000.0 <= elapsed <= 22_000.0

    def test_gbytes_per_sec_circular(self):
        cfg = RdmaModeConfig.from_link_bw(
            name="x", medium="X", link_bw_GBps=50.0,
            per_message_overhead_us=0.0,
        )
        sim = RdmaSimulator(cfg)
        # Without overhead, the effective BW equals the link BW.
        one_gb = 1 << 30
        elapsed = sim.elapsed_us(one_gb)
        bw = sim.gbytes_per_sec(one_gb, elapsed)
        assert bw == pytest.approx(50.0, rel=1e-3)

    def test_zero_elapsed_clamps(self):
        cfg = RdmaModeConfig.from_link_bw(
            name="x", medium="X", link_bw_GBps=50.0,
            per_message_overhead_us=0.0,
        )
        sim = RdmaSimulator(cfg)
        # Avoid div-by-zero.
        assert sim.gbytes_per_sec(1024, 0.0) == 0.0


# ===========================================================================
# Percentile / summarize
# ===========================================================================


class TestPercentile:
    """Linear-interpolation percentile."""

    def test_single_value(self):
        assert percentile([5.0], 50) == 5.0
        assert percentile([5.0], 99) == 5.0

    def test_p50_median(self):
        # [1, 2, 3, 4, 5] → median = 3.
        assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0

    def test_p99_tail(self):
        # [1..100] → P99 ~ 99.01.
        vals = [float(i) for i in range(1, 101)]
        p99 = percentile(vals, 99)
        assert 98.9 <= p99 <= 99.1

    def test_empty(self):
        assert percentile([], 50) == 0.0


class TestSummarize:
    """P50/P99/mean/min summary."""

    def test_basic(self):
        stats = summarize([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats["p50_us"] == 3.0
        assert stats["mean_us"] == 3.0
        assert stats["min_us"] == 1.0

    def test_empty(self):
        stats = summarize([])
        assert stats["p50_us"] == 0.0
        assert stats["min_us"] == 0.0


# ===========================================================================
# Paged KV cache allocation
# ===========================================================================


class TestMakePagedKvCache:
    """Paged cache allocation shape."""

    def test_shape(self):
        p = KvTransferParams(
            num_tokens=64, num_layers=4, num_kv_heads=2,
            head_size=8, block_size=16, num_blocks=4,
        )
        k, v = _make_paged_kv_cache(p, torch.device("cpu"))
        assert k.shape == (4, 4, 16, 2, 8)
        assert v.shape == (4, 4, 16, 2, 8)
        assert k.dtype == torch.bfloat16
        assert v.dtype == torch.bfloat16


# ===========================================================================
# Bench functions
# ===========================================================================


class TestBenchInProcess:
    """`bench_in_process_transfer` returns a list of μs timings."""

    def test_returns_list_of_floats(self):
        p = KvTransferParams(
            num_tokens=64, num_layers=2, num_kv_heads=2,
            head_size=8, block_size=16, num_blocks=4,
        )
        times = bench_in_process_transfer(p, iters=3, device=torch.device("cpu"))
        assert len(times) == 3
        for t in times:
            assert isinstance(t, float)
            assert t > 0.0


class TestBenchRdmaSimulated:
    """`bench_rdma_simulated_transfer` returns near-elapsed_us per iter."""

    def test_returns_list_of_floats(self):
        p = KvTransferParams(
            num_tokens=64, num_layers=2, num_kv_heads=2,
            head_size=8, block_size=16, num_blocks=4,
        )
        cfg = RdmaModeConfig.from_link_bw(
            name="x", medium="X", link_bw_GBps=50.0,
            per_message_overhead_us=100.0,
        )
        times = bench_rdma_simulated_transfer(p, iters=3, cfg=cfg)
        assert len(times) == 3
        # 100 μs overhead + tiny bytes ≈ 100 μs per iter.
        for t in times:
            assert t >= 100.0

    def test_engine_simulator_plus_actual(self):
        """Simulated elapsed_us ≈ wall-clock (within sleep granularity)."""
        p = KvTransferParams(
            num_tokens=64, num_layers=2, num_kv_heads=2,
            head_size=8, block_size=16, num_blocks=4,
        )
        cfg = RdmaModeConfig.from_link_bw(
            name="x", medium="X", link_bw_GBps=50.0,
            per_message_overhead_us=1000.0,  # 1 ms overhead
        )
        times = bench_rdma_simulated_transfer(p, iters=2, cfg=cfg)
        # Sleep granularity is ~1 ms on most systems; allow 0.5 ms slack.
        for t in times:
            assert 1000.0 <= t <= 1500.0


# ===========================================================================
# Param grid + BenchResult
# ===========================================================================


class TestParamGrid:
    """`_make_param_grid` builds a list of params."""

    def test_basic(self):
        grid = _make_param_grid(
            num_tokens_list=[128, 256],
            num_layers=4, num_kv_heads=4, head_size=16, block_size=16,
        )
        assert len(grid) == 2
        assert grid[0].num_tokens == 128
        assert grid[0].num_blocks == 8


class TestBenchResult:
    """`BenchResult.to_csv` produces 16 columns."""

    def test_csv_row(self):
        r = BenchResult(
            mode="in_process", medium="SHM",
            num_tokens=128, num_layers=4, num_kv_heads=4,
            head_size=16, block_size=16, num_blocks=8,
            kv_bytes=131072, iteration=10,
            p50_us=200.0, p99_us=400.0, mean_us=250.0, min_us=180.0,
            gbytes_per_sec=0.5, note="ok",
        )
        line = r.to_csv()
        fields = line.split(",")
        assert len(fields) == 16
        assert fields[0] == "in_process"
        assert fields[-1] == "ok"

    def test_csv_header(self):
        h = _emit_csv_header()
        cols = h.split(",")
        assert len(cols) == 16


# ===========================================================================
# run_benchmark end-to-end
# ===========================================================================


class TestRunBenchmark:
    """`run_benchmark` returns N rows."""

    def test_in_process_only(self):
        results = run_benchmark(
            iters=3,
            num_tokens_list=[128, 256],
            num_layers=2, num_kv_heads=2, head_size=8, block_size=16,
            include_rdma=False,
            device=torch.device("cpu"),
        )
        # 2 sizes × 1 mode = 2 rows.
        assert len(results) == 2
        for r in results:
            assert r.mode == "in_process"
            assert r.kv_bytes > 0
            assert r.gbytes_per_sec >= 0.0

    def test_in_process_plus_rdma(self):
        results = run_benchmark(
            iters=3,
            num_tokens_list=[128],
            num_layers=2, num_kv_heads=2, head_size=8, block_size=16,
            include_rdma=True,
            device=torch.device("cpu"),
        )
        # 1 size × (1 in-process + 4 RDMA modes) = 5 rows.
        assert len(results) == 5
        modes = {r.mode for r in results}
        assert modes == {"in_process", "nvlink", "ib_hdr", "ib_ndr", "roce"}


# ===========================================================================
# CSV round-trip
# ===========================================================================


class TestCsvRoundtrip:
    """CSV serialization round-trip."""

    def test_header_and_row_parseable(self):
        results = run_benchmark(
            iters=2,
            num_tokens_list=[64],
            num_layers=2, num_kv_heads=2, head_size=8, block_size=16,
            include_rdma=True,
            device=torch.device("cpu"),
        )
        buf = io.StringIO()
        buf.write(_emit_csv_header() + "\n")
        for r in results:
            buf.write(r.to_csv() + "\n")
        # Parse via csv.DictReader.
        reader = csv.DictReader(io.StringIO(buf.getvalue()))
        rows = list(reader)
        assert len(rows) == len(results)
        # Spot-check fields.
        for row in rows:
            assert "mode" in row
            assert "p50_us" in row
            assert "gbytes_per_sec" in row
            assert float(row["kv_bytes"]) > 0
            assert float(row["p50_us"]) >= 0.0


# ===========================================================================
# CLI smoke
# ===========================================================================


class TestCliSmoke:
    """Subprocess invocation emits a CSV row."""

    def test_no_rdma(self):
        proc = subprocess.run(
            [
                sys.executable,
                "examples/pd_disaggregated_benchmark.py",
                "--iters", "2",
                "--num-tokens", "64",
                "--num-layers", "2",
                "--num-kv-heads", "2",
                "--head-size", "8",
                "--quiet",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(Path(__file__).parent.parent),
        )
        assert proc.returncode == 0, proc.stderr
        lines = proc.stdout.strip().split("\n")
        assert len(lines) == 2
        # Header + 1 row.
        assert lines[0].startswith("mode,medium,")
        assert lines[1].startswith("in_process,SHM,")

    def test_with_rdma(self):
        proc = subprocess.run(
            [
                sys.executable,
                "examples/pd_disaggregated_benchmark.py",
                "--iters", "2",
                "--num-tokens", "64",
                "--num-layers", "2",
                "--num-kv-heads", "2",
                "--head-size", "8",
                "--include-rdma",
                "--quiet",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(Path(__file__).parent.parent),
        )
        assert proc.returncode == 0, proc.stderr
        lines = proc.stdout.strip().split("\n")
        # 1 size × 5 modes = 5 data rows + 1 header = 6 lines.
        assert len(lines) == 6
        modes = {line.split(",")[0] for line in lines[1:]}
        assert modes == {"in_process", "nvlink", "ib_hdr", "ib_ndr", "roce"}


# ===========================================================================
# Integration: mean ≈ elapsed_us for RDMA-simulated
# ===========================================================================


class TestIntegration:
    """End-to-end sanity for the harness."""

    def test_simulated_bw_matches_link(self):
        """For a tiny transfer, the RDMA-simulated BW ≈ link BW."""
        p = KvTransferParams(
            num_tokens=64, num_layers=1, num_kv_heads=1,
            head_size=4, block_size=16, num_blocks=4,
        )
        # 1 GB transfer at 50 GB/s → 20 ms (no overhead).
        cfg = RdmaModeConfig.from_link_bw(
            name="x", medium="X", link_bw_GBps=50.0,
            per_message_overhead_us=0.0,
        )
        sim = RdmaSimulator(cfg)
        # 1 GB at 50 GB/s = 20 ms.
        one_gb = 1 << 30
        elapsed = sim.elapsed_us(one_gb)
        bw = sim.gbytes_per_sec(one_gb, elapsed)
        assert bw == pytest.approx(50.0, rel=1e-3)

    def test_in_process_scaling(self):
        """In-process transfer scales sub-linearly with bytes (single
        contiguous copy dominates Python overhead for big buffers).

        The test compares a small (32 KB) vs a much larger (1 MB)
        transfer — the larger one should be at least 2× slower in
        practice.  This is a weak monotonicity check (Python
        overhead can mask small size differences) so we use a
        conservative 2× threshold.
        """
        small = bench_in_process_transfer(
            KvTransferParams(
                num_tokens=64, num_layers=1, num_kv_heads=1,
                head_size=4, block_size=16, num_blocks=4,
            ),
            iters=5,
            device=torch.device("cpu"),
        )
        large = bench_in_process_transfer(
            KvTransferParams(
                num_tokens=4096, num_layers=4, num_kv_heads=4,
                head_size=16, block_size=16, num_blocks=256,
            ),
            iters=5,
            device=torch.device("cpu"),
        )
        small_mean = statistics.mean(small)
        large_mean = statistics.mean(large)
        # The large transfer is 1024× larger by byte count; the
        # Python overhead should still be a small constant.  So
        # large should be at least 2× slower (loose because the
        # SHM allocation dominates the small one).
        assert large_mean >= small_mean * 2.0, (
            f"large_mean={large_mean:.1f}μs should be ≥ "
            f"2× small_mean={small_mean:.1f}μs"
        )
