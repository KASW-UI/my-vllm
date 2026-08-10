"""Benchmark suite — throughput, latency, prefill, decode, MoE, EP, TP, PP, CP, KV Connector.

C098: full-stack performance modeling.  Each module provides a benchmark function
that returns a dict of metrics; run_all.py aggregates and prints a summary table.

Usage: python benchmark/run_all.py
"""

# Re-export for convenience
from benchmark.throughput import benchmark_throughput  # noqa: F401
