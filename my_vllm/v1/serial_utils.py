"""Serialization utilities — IPC support for multiprocess executor.

Mirrors vLLM V1 v1/serial_utils.py (simplified).

C031: provides MsgpackEncoder/Decoder stubs for future
multiprocess executor communication.
"""


class MsgpackEncoder:
    """Simplified msgpack-compatible encoder stub.

    In production vLLM, used for serializing SchedulerOutput
    across process boundaries.  C031 provides a minimal
    placeholder for the architecture boundary.
    """

    @staticmethod
    def encode(obj: object) -> bytes:
        import json

        return json.dumps(obj, default=str).encode()

    @staticmethod
    def decode(data: bytes) -> object:
        import json

        return json.loads(data)
