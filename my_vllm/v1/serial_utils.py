"""Serialization utilities — IPC support for multiprocess executor.

Mirrors vLLM V1 v1/serial_utils.py (simplified — educational subset).

L6: real MsgpackEncoder/Decoder with hook interface and binary format,
replacing previous JSON stub.
"""

from __future__ import annotations

import struct
from collections.abc import Callable, Sequence
from typing import Any

_TYPE_INT = b"\x01"
_TYPE_STR = b"\x02"
_TYPE_LIST = b"\x03"
_TYPE_DICT = b"\x04"
_TYPE_NONE = b"\x05"
_TYPE_BYTES = b"\x06"
_TYPE_CUSTOM = b"\x07"
_INT_NEG = b"\x08"


class MsgpackEncoder:
    """Binary encoder with custom type hook — mirrors vLLM MsgpackEncoder.

    Instance-based (not static) with an enc_hook for custom types.
    Uses a simple binary format: 1-byte type tag + length-prefixed data.
    """

    def __init__(self) -> None:
        self._enc_hook: Callable[[Any], Any] | None = None

    def encode(self, obj: Any) -> bytes:
        return self._encode_value(obj)

    @staticmethod
    def encode_static(obj: Any) -> bytes:
        encoder = MsgpackEncoder()
        return encoder.encode(obj)

    def _encode_value(self, obj: Any) -> bytes:
        t = type(obj)
        if obj is None:
            return _TYPE_NONE
        if t is int:
            return self._encode_int(obj)
        if t is str:
            return self._encode_bytes(obj.encode("utf-8"), _TYPE_STR)
        if t is bytes:
            return self._encode_bytes(obj, _TYPE_BYTES)
        if t is list:
            return self._encode_list(obj)
        if t is dict:
            return self._encode_dict(obj)
        if self._enc_hook is not None:
            encoded = self._enc_hook(obj)
            return _TYPE_CUSTOM + self._encode_bytes(
                encoded if isinstance(encoded, bytes) else str(encoded).encode("utf-8"),
                b"",
            )
        raise TypeError(f"Cannot encode type: {t}")

    def _encode_int(self, val: int) -> bytes:
        if val >= 0:
            return _TYPE_INT + struct.pack(">q", val)
        return _INT_NEG + struct.pack(">q", abs(val))

    @staticmethod
    def _encode_bytes(data: bytes, tag: bytes) -> bytes:
        return tag + struct.pack(">I", len(data)) + data

    def _encode_list(self, obj: list) -> bytes:
        result = _TYPE_LIST + struct.pack(">I", len(obj))
        for item in obj:
            result += self._encode_value(item)
        return result

    def _encode_dict(self, obj: dict) -> bytes:
        items = list(obj.items())
        result = _TYPE_DICT + struct.pack(">I", len(items))
        for key, value in items:
            result += self._encode_value(key)
            result += self._encode_value(value)
        return result


class MsgpackDecoder:
    """Binary decoder with custom type hook — mirrors vLLM MsgpackDecoder.

    Instance-based dec_hook for reconstructing custom types.
    """

    def __init__(self) -> None:
        self._dec_hook: Callable[[type, Any], Any] | None = None
        self._pos: int = 0
        self._data: bytes = b""

    def decode(self, data: bytes) -> Any:
        self._data = data
        self._pos = 0
        return self._decode_value()

    @staticmethod
    def decode_static(data: bytes) -> Any:
        decoder = MsgpackDecoder()
        return decoder.decode(data)

    def _decode_value(self) -> Any:
        tag = self._data[self._pos:self._pos + 1]
        self._pos += 1

        if tag == _TYPE_NONE:
            return None
        if tag in (_TYPE_INT, _INT_NEG):
            val = struct.unpack(">q", self._data[self._pos:self._pos + 8])[0]
            self._pos += 8
            return -val if tag == _INT_NEG else val
        if tag in (_TYPE_STR, _TYPE_BYTES, _TYPE_CUSTOM, b""):
            length = struct.unpack(">I", self._data[self._pos:self._pos + 4])[0]
            self._pos += 4
            data = self._data[self._pos:self._pos + length]
            self._pos += length
            if tag == _TYPE_STR:
                return data.decode("utf-8")
            if tag == _TYPE_CUSTOM and self._dec_hook is not None:
                return self._dec_hook(type(data), data)
            return data
        if tag == _TYPE_LIST:
            length = struct.unpack(">I", self._data[self._pos:self._pos + 4])[0]
            self._pos += 4
            return [self._decode_value() for _ in range(length)]
        if tag == _TYPE_DICT:
            length = struct.unpack(">I", self._data[self._pos:self._pos + 4])[0]
            self._pos += 4
            return {self._decode_value(): self._decode_value() for _ in range(length)}

        raise ValueError(f"Unknown tag: {tag!r}")
