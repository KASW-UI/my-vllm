"""KV offload CPU backend — ARC eviction policy (shell).

Mirrors vLLM V1 vllm/v1/kv_offload/cpu/policies/arc.py.

C256 shell: ARC (Adaptive Replacement Cache) requires a ghost list
+ adaptive `target_t1_size` mechanism. The my-vllm JD-aligned
subset ships LRU only — ARC is declared here so the policy
registry has a real name + importable surface, but the body
raises NotImplementedError on actual use.

ARC is a well-known algorithm (Megiddo + Modha, 2003) — its core
trick is maintaining two LRU lists (T1 = recently used, T2 =
frequently used) plus ghost lists (B1 / B2) that track evicted
keys, and adaptively shifting the T1/T2 partition based on hits
in the ghost lists. The math is straightforward but the
implementation is non-trivial (multiple ordered dicts +
probabilistic size balancing). Out of scope for the JD-aligned
subset.
"""

from __future__ import annotations

from collections.abc import Iterable

from my_vllm.v1.kv_offload.base import OffloadKey, ReqContext
from my_vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy


class ARCCachePolicy(CachePolicy):
    """ARC (Adaptive Replacement Cache) — C256 shell.

    Real impl lands when there's a deployment that needs scan-
    resistance (the standard ARC use case). The shell preserves
    the vLLM API so the manager + spec can declare ARC as a
    supported eviction policy without crashing on import.
    """

    def __init__(self, cache_capacity: int):
        del cache_capacity
        self._init_err = (
            "ARCCachePolicy is a C256 shell — the real ARC impl "
            "lands with Phase 5. Use 'lru' for now."
        )
        # State the ARC impl would maintain.
        self.t1: dict = {}
        self.t2: dict = {}
        self.b1: dict = {}  # ghost list (recently evicted from T1)
        self.b2: dict = {}  # ghost list (recently evicted from T2)
        self.p: float = 0.0  # adaptive target size for T1

    def get(self, key: OffloadKey) -> BlockStatus | None:
        raise NotImplementedError(self._init_err)

    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        raise NotImplementedError(self._init_err)

    def remove(self, key: OffloadKey) -> None:
        raise NotImplementedError(self._init_err)

    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        del req_context
        raise NotImplementedError(self._init_err)

    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        del n, protected
        raise NotImplementedError(self._init_err)

    def clear(self) -> None:
        raise NotImplementedError(self._init_err)


__all__ = ["ARCCachePolicy"]