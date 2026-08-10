"""Run all benchmarks — Phase 4 convergence verification (C098).

Usage: python benchmark/run_all.py
"""

from benchmark.cp import benchmark_cp_long_context
from benchmark.decode import benchmark_decode
from benchmark.ep import benchmark_ep_scaling
from benchmark.kv_connector import benchmark_kv_connector
from benchmark.latency import benchmark_latency
from benchmark.moe import benchmark_moe
from benchmark.pp import benchmark_pp_bubble
from benchmark.prefill import benchmark_prefill
from benchmark.throughput import benchmark_throughput
from benchmark.tp import benchmark_tp_speedup

BENCHMARKS = {
    "throughput": benchmark_throughput,
    "latency": benchmark_latency,
    "prefill": benchmark_prefill,
    "decode": benchmark_decode,
    "moe": benchmark_moe,
    "ep": benchmark_ep_scaling,
    "tp": benchmark_tp_speedup,
    "pp": benchmark_pp_bubble,
    "cp": benchmark_cp_long_context,
    "kv_connector": benchmark_kv_connector,
}


def run_all():
    print("=" * 60)
    print("  C098 Benchmark Suite")
    print("=" * 60)
    for name, fn in BENCHMARKS.items():
        print(f"\n--- {name} ---")
        result = fn()
        for k, v in result.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            elif isinstance(v, dict):
                print(f"  {k}:")
                for kk, kv in v.items():
                    if isinstance(kv, float):
                        print(f"    {kk}: {kv:.2f}")
                    else:
                        print(f"    {kk}: {kv}")
            else:
                print(f"  {k}: {v}")
    print("\n" + "=" * 60)
    print("  All benchmarks complete")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
