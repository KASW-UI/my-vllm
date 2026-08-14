"""Stats and Prometheus metrics for the NIXL connector.

C250: extracted from ``vllm/distributed/kv_transfer/kv_connector/v1/nixl/
stats.py`` (266 lines).  Scope reductions:

  * Prometheus metrics (``NixlPromMetrics``) — **omitted**.
    my-vllm has no Prometheus integration; the C231 / C232 era
    decided metrics are out of scope for the JD-aligned subset.
    The CLI-only ``reduce()`` path is preserved.
  * ``nixlXferTelemetry`` (a NIXL-library type) — replaced with a
    plain dataclass ``TransferTelemetry`` with the same fields.
  * numpy percentile → pure-Python percentile (numpy is a soft
    import for the rest of my-vllm but not required here).

References:
  * vLLM stats: ``vllm/distributed/kv_transfer/kv_connector/v1/nixl/stats.py``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


# ── Minimal stand-in for vllm.distributed.kv_transfer.kv_connector.v1.metrics.KVConnectorStats
# my-vllm hasn't vendored Prometheus metrics (deferred), so we define a
# minimal ABC that ``NixlKVConnectorStats`` extends.  The real
# ``KVConnectorStats`` in vLLM holds ``data`` + ``is_empty()`` +
# ``aggregate()`` + ``reduce()`` methods that we mirror here.
class KVConnectorStats:
    """Minimal stats ABC for NIXL connector.

    Mirrors ``vllm.distributed.kv_transfer.kv_connector.v1.metrics
    .KVConnectorStats`` minus the Prometheus integration.
    """

    data: dict[str, list[float | int]]

    def is_empty(self) -> bool:
        raise NotImplementedError

    def aggregate(self, other: "KVConnectorStats") -> "KVConnectorStats":
        raise NotImplementedError


@dataclass
class TransferTelemetry:
    """Per-transfer telemetry record.

    Mirrors ``nixlXferTelemetry`` from vLLM's vendored NIXL bindings.
    Used by ``NixlKVConnectorStats.record_transfer``.
    """

    xferDuration: float  # microseconds
    postDuration: float  # microseconds
    totalBytes: int
    descCount: int


def _percentile(values: list[float], p: float) -> float:
    """Pure-Python percentile (linear interpolation, matches numpy)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    k = (n - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, n - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


@dataclass
class NixlKVConnectorStats(KVConnectorStats):
    """Container for transfer performance metrics.

    Mirrors ``vllm.distributed.kv_transfer.kv_connector.v1.nixl.stats
    .NixlKVConnectorStats``.  Tracks per-transfer timing, byte counts,
    descriptor counts, and failure counters.
    """

    def __post_init__(self) -> None:
        if not getattr(self, "data", None):
            self.reset()

    def reset(self) -> None:
        self.data: dict[str, list[float | int]] = {
            "transfer_duration": [],   # seconds
            "post_duration": [],        # seconds
            "bytes_transferred": [],    # bytes
            "num_descriptors": [],      # count
            "num_failed_transfers": [],  # count
            "num_failed_notifications": [],  # count
            "num_kv_expired_reqs": [],  # count
        }

    def record_transfer(self, res: TransferTelemetry) -> None:
        """Record a successful NIXL transfer."""
        self.data["transfer_duration"].append(res.xferDuration / 1e6)
        self.data["post_duration"].append(res.postDuration / 1e6)
        self.data["bytes_transferred"].append(res.totalBytes)
        self.data["num_descriptors"].append(res.descCount)

    def record_failed_transfer(self) -> None:
        """Record a failed NIXL transfer operation."""
        self.data["num_failed_transfers"].append(1)

    def record_failed_notification(self) -> None:
        """Record a failed NIXL notification (send_notif)."""
        self.data["num_failed_notifications"].append(1)

    def record_kv_expired_req(self) -> None:
        """Record a request whose KV blocks expired before consumption."""
        self.data["num_kv_expired_reqs"].append(1)

    def clone_and_reset(self) -> "NixlKVConnectorStats":
        """Clone current stats + reset accumulator (for periodic logging)."""
        old = copy.copy(self)
        self.reset()
        return old

    def is_empty(self) -> bool:
        """True iff no metrics worth reporting in this interval."""
        return (
            self.num_successful_transfers == 0
            and len(self.data["num_failed_transfers"]) == 0
            and len(self.data["num_failed_notifications"]) == 0
            and len(self.data["num_kv_expired_reqs"]) == 0
        )

    def aggregate(self, other: "KVConnectorStats") -> "KVConnectorStats":
        """Merge another stats object into this one."""
        if not other.is_empty():
            for k, v in other.data.items():
                accumulator = self.data[k]
                assert isinstance(accumulator, list)
                accumulator.extend(v)
        return self

    def reduce(self) -> dict[str, int | float]:
        """Compute compact representative stats for CLI logging.

        Mirrors ``vllm.distributed.kv_transfer.kv_connector.v1.nixl.stats
        .NixlKVConnectorStats.reduce``.  If there were no successful
        transfers in this interval, returns a zeros dict (so callers
        can spot "all failed" vs "no activity").
        """
        if self.num_successful_transfers == 0:
            return {
                "Num successful transfers": 0,
                "Avg xfer time (ms)": 0,
                "P90 xfer time (ms)": 0,
                "Avg post time (ms)": 0,
                "P90 post time (ms)": 0,
                "Avg MB per transfer": 0,
                "Throughput (MB/s)": 0,
                "Avg number of descriptors": 0,
            }

        xfer_time = self.data["transfer_duration"]
        post_time = self.data["post_duration"]
        mb = [b / (2**20) for b in self.data["bytes_transferred"]]
        descs = self.data["num_descriptors"]
        n = len(descs)
        assert n == self.num_successful_transfers

        total_mb = sum(mb)
        avg_mb = total_mb / n
        total_time_seconds = sum(xfer_time)
        throughput_mb_s = total_mb / total_time_seconds if total_time_seconds > 0 else 0

        return {
            "Num successful transfers": n,
            "Avg xfer time (ms)": round(
                (sum(xfer_time) / n) * 1e3, 3,
            ),
            "P90 xfer time (ms)": round(
                _percentile(xfer_time, 90) * 1e3, 3,
            ),
            "Avg post time (ms)": round(
                (sum(post_time) / n) * 1e3, 3,
            ),
            "P90 post time (ms)": round(
                _percentile(post_time, 90) * 1e3, 3,
            ),
            "Avg MB per transfer": round(avg_mb, 3),
            "Throughput (MB/s)": round(throughput_mb_s, 3),
            "Avg number of descriptors": round(
                sum(descs) / n, 1,
            ),
        }

    @property
    def num_successful_transfers(self) -> int:
        return len(self.data["transfer_duration"])


__all__ = [
    "NixlKVConnectorStats",
    "TransferTelemetry",
]
