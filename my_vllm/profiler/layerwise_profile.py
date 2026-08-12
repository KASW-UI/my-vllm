"""
Layerwise profiling — correlates CUDA events with model module structure.

Architecture:
  LayerwiseProfileResults extends torch.profiler.profile, building a module
  tree from Kineto CUDA events, then flattening into ModelStatsEntry (per-layer)
  and SummaryStatsEntry (per-op) trees.  This enables:
    - Per-layer GPU time breakdown
    - Op-level latency attribution
    - CSV export for offline analysis (pandas)
    - Hierarchical print tables via TablePrinter
"""

import copy
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar


@dataclass
class SummaryStatsEntry:
    name: str
    cuda_time_us: float
    pct_cuda_time: float
    invocations: int


@dataclass
class ModelStatsEntry:
    name: str
    cpu_time_us: float
    cuda_time_us: float
    pct_cuda_time: float
    trace: str


class LayerwiseProfileResults:
    """
    Layerwise profiling results.

    In real vLLM this extends torch.profiler.profile and parses
    _ProfilerResult → _KinetoEvent → _ModuleTreeNode →
    _StatsTreeNode[ModelStatsEntry] / _StatsTreeNode[SummaryStatsEntry].

    Here we provide the same data model and public API surface.
    """

    num_running_seqs: int | None = None

    def __init__(self):
        self._model_stats_tree: list = []
        self._summary_stats_tree: list = []

    def print_model_table(self, column_widths: dict[str, int] | None = None):
        """Print per-layer model statistics table."""
        pass

    def print_summary_table(self, column_widths: dict[str, int] | None = None):
        """Print op-level summary statistics table."""
        pass

    def export_model_stats_table_csv(self, filename: str):
        """Export per-layer statistics to CSV."""
        pass

    def export_summary_stats_table_csv(self, filename: str):
        """Export summary statistics to CSV."""
        pass

    def convert_stats_to_dict(self) -> dict[str, Any]:
        """Convert stats trees to a flat dictionary."""
        return {}
