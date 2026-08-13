"""Tensor IPC transport — multi-process tensor sharing.

Mirrors vLLM V1 v1/engine/tensor_ipc.py.

vLLM's `tensor_ipc.py` provides zero-copy IPC between the API
server (frontend) and the engine core (worker) using
`torch.multiprocessing.Queue`. The sender moves tensors to
shared memory; the receiver reconstructs a tensor view via
the queue's metadata dict.

C142: minimal subset — `TensorIpcData` dataclass + the
`TensorIpcSender` / `TensorIpcReceiver` classes with the
public API surface (`send(tensor) -> handle_dict`,
`recv(handle_dict) -> tensor`). The full vLLM impl handles
UUID-based sender tracking, `OOBTensorConsumer` integration,
queue routing, and cleanup callbacks — most are out of scope
for my-vllm's DP=1 default.

For DP=1 (my-vllm's default) the sender / receiver short-
circuit: tensors are passed by reference (no copy needed since
the API server and engine core are in the same process).
"""

from __future__ import annotations

import dataclasses
import logging
import uuid
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TensorIpcData:
    """Zero-copy tensor handle for inter-process IPC.

    Mirrors vLLM V1 v1/engine/tensor_ipc.py:TensorIpcData (L30).
    """

    sender_id: str
    message_id: int
    tensor_id: int
    tensor: torch.Tensor

    def to_handle_dict(self) -> dict[str, Any]:
        """Serialize to a flat dict for IPC transport."""
        return {
            "sender_id": self.sender_id,
            "message_id": self.message_id,
            "tensor_id": self.tensor_id,
            "tensor": self.tensor,
        }

    @classmethod
    def from_handle_dict(cls, d: dict[str, Any]) -> "TensorIpcData":
        """Reconstruct from a flat dict."""
        return cls(
            sender_id=d["sender_id"],
            message_id=d["message_id"],
            tensor_id=d["tensor_id"],
            tensor=d["tensor"],
        )


class TensorIpcSender:
    """Send-side IPC helper.

    Mirrors vLLM V1 v1/engine/tensor_ipc.py:TensorIpcSender (L45).

    C142: minimal — for DP=1 the sender doesn't need to move
    tensors to shared memory (same process). Generates a unique
    sender_id and monotonically increasing message_id /
    tensor_id counters.
    """

    def __init__(self) -> None:
        self._tensor_id_counter = 0
        self._message_counter = 0
        self._sender_id = uuid.uuid4().hex[:8]

    def new_message(self) -> None:
        """Reset per-message tensor_id counter for a new message."""
        self._message_counter += 1
        self._tensor_id_counter = 0

    def __call__(self, tensor: torch.Tensor) -> dict[str, Any] | None:
        """Send a tensor and return its handle dict.

        Mirrors vLLM V1 v1/engine/tensor_ipc.py:TensorIpcSender.__call__.

        C142: minimal — no actual transport (returns a handle
        dict directly). A follow-up wires the
        `torch.multiprocessing.Queue` for true cross-process IPC.
        """
        try:
            handle = TensorIpcData(
                sender_id=self._sender_id,
                message_id=self._message_counter,
                tensor_id=self._tensor_id_counter,
                tensor=tensor,
            ).to_handle_dict()
            self._tensor_id_counter += 1
            return handle
        except Exception as exc:
            logger.warning("TensorIpcSender failed: %s", exc)
            return None


class TensorIpcReceiver:
    """Receive-side IPC helper.

    Mirrors vLLM V1 v1/engine/tensor_ipc.py:OOBTensorReceiver.

    C142: minimal — reconstructs the tensor from a handle dict.
    The full vLLM impl maintains a `_handle_map` for late-
    arriving tensors and registers cleanup hooks.
    """

    def __init__(self) -> None:
        self._handle_map: dict[tuple[str, int, int], torch.Tensor] = {}

    def register(self, handle: dict[str, Any]) -> torch.Tensor:
        """Register a handle and return the underlying tensor."""
        data = TensorIpcData.from_handle_dict(handle)
        key = (data.sender_id, data.message_id, data.tensor_id)
        self._handle_map[key] = data.tensor
        return data.tensor

    def __call__(self, handle: dict[str, Any]) -> torch.Tensor:
        """Resolve a handle dict to the original tensor."""
        return self.register(handle)

    def cleanup(self, sender_id: str | None = None) -> int:
        """Drop handles; optionally filter by `sender_id`.

        Returns the number of handles dropped.
        """
        if sender_id is None:
            n = len(self._handle_map)
            self._handle_map.clear()
            return n
        keys = [k for k in self._handle_map if k[0] == sender_id]
        for k in keys:
            self._handle_map.pop(k, None)
        return len(keys)


__all__ = [
    "TensorIpcData",
    "TensorIpcReceiver",
    "TensorIpcSender",
]