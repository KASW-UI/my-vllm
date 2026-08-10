"""Prometheus metrics — stub for C048.

Real vLLM uses multiprocess Prometheus via setup_multiprocess_prometheus().
This stub provides a minimal interface for future expansion.
"""


def setup_multiprocess_prometheus(mp_prometheus_port: int = 0) -> None:
    """Placeholder: real vLLM initializes multiprocess prometheus registry."""
    pass
