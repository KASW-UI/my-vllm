from collections.abc import Sequence
from typing import Generic, TypeVar, overload

T = TypeVar("T")


def cdiv(a: int, b: int) -> int:
    """Ceiling division (mirrors vllm/utils/math_utils.py:cdiv).

    C257 SimpleCPUOffloadScheduler uses this for block-aligned
    arithmetic (``cdiv(confirmed_tokens, block_size)`` etc.).
    """
    return -(a // -b) if b > 0 else (a // b if b != 0 else 0)


class ConstantList(Generic[T], Sequence):
    """A read-only list wrapper that raises TypeError on mutation.

    Mirrors vLLM V1's v1/utils.py:ConstantList — used to provide
    read-only views of token ID lists in Request objects.
    """

    def __init__(self, x: list[T]) -> None:
        self._x = x

    def append(self, item):
        raise TypeError("Cannot append to a constant list")

    def extend(self, item):
        raise TypeError("Cannot extend a constant list")

    def insert(self, item):
        raise TypeError("Cannot insert into a constant list")

    def pop(self, item):
        raise TypeError("Cannot pop from a constant list")

    def remove(self, item):
        raise TypeError("Cannot remove from a constant list")

    def clear(self):
        raise TypeError("Cannot clear a constant list")

    def index(self, item: T, start: int = 0,
              stop: int | None = None) -> int:
        return self._x.index(item, start,
                             stop if stop is not None else len(self._x))

    @overload
    def __getitem__(self, item: int) -> T: ...
    @overload
    def __getitem__(self, s: slice) -> list[T]: ...

    def __getitem__(self, item: int | slice) -> T | list[T]:
        return self._x[item]

    @overload
    def __setitem__(self, item: int, value: T): ...
    @overload
    def __setitem__(self, s: slice, value: T): ...

    def __setitem__(self, item: int | slice, value: T | list[T]):
        raise TypeError("Cannot set item in a constant list")

    def __delitem__(self, item):
        raise TypeError("Cannot delete item from a constant list")

    def __iter__(self):
        return iter(self._x)

    def __contains__(self, item):
        return item in self._x

    def __len__(self):
        return len(self._x)

    def __eq__(self, other):
        if isinstance(other, ConstantList):
            return self._x == other._x
        return self._x == other

    def __repr__(self):
        return f"ConstantList({self._x!r})"
