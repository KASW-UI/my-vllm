"""Engine process lifecycle management and utilities.

Real vLLM architecture (vllm/v1/engine/utils.py):

  CoreEngineState — tracks engine process lifecycle (NEW → CONNECTED → READY)
  CoreEngine        — one per DP rank, identity + state tracking
  EngineZmqAddresses — socket addresses for front-end ↔ engine ZMQ communication
  EngineHandshakeMetadata — startup metadata (addresses, parallel config)

  Process launch:
    launch_core_engines() — spawns EngineCore processes via multiprocessing
    with ZMQ socket binding, DP coordinator, graceful shutdown via sentinel.

  Health monitoring:
    poll_engine_health() — heartbeat check on engine process, detects dead engines.
    check_health() on AsyncLLM raises EngineDeadError if the engine core process died.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import ParallelConfig


class CoreEngineState(Enum):
    """Engine core process lifecycle states."""
    NEW = auto()
    CONNECTED = auto()
    READY = auto()


class CoreEngine:
    """One per data parallel rank, used to track state during handshaking."""

    def __init__(self, index: int = 0, local: bool = True):
        self.local = local
        self.identity = index.to_bytes(2, "little")
        self.state = CoreEngineState.NEW


@dataclass
class EngineZmqAddresses:
    """ZMQ socket addresses for front-end ↔ engine communication."""
    inputs: list[str]
    outputs: list[str]
    coordinator_input: str | None = None
    coordinator_output: str | None = None
    frontend_stats_publish_address: str | None = None


@dataclass
class EngineHandshakeMetadata:
    """Metadata sent to each engine process during startup handshake."""
    addresses: EngineZmqAddresses
    parallel_config: dict
