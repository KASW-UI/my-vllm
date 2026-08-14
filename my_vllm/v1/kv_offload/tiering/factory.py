"""KV offload tiering — secondary-tier factory + registry.

Mirrors vLLM V1 vllm/v1/kv_offload/tiering/factory.py.

C256 ships the lazy-loading registry + factory plumbing. No
concrete secondary tier is registered by default (Storage /
FS / P2P / ObjectStore land when a real workload needs them).

The "example" tier is a class shell that returns MISS for every
lookup + raises NotImplementedError on actual transfer — it
serves as a smoke test that the registry + factory plumbing
works end-to-end.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from my_vllm.v1.kv_offload.tiering.base import (
    JobMetadata,
    JobResult,
    ReqContext,
    SecondaryTierManager,
)

if TYPE_CHECKING:
    from my_vllm.v1.kv_offload.base import OffloadingSpec
    from my_vllm.v1.kv_offload.tiering.base import OffloadKey


class SecondaryTierFactory:
    """Lazy-loading registry for secondary-tier subclasses."""

    _registry: dict[str, Callable[[], type[SecondaryTierManager]]] = {}

    @classmethod
    def register_tier(
        cls,
        tier_type: str,
        module_path: str,
        class_name: str,
    ) -> None:
        if tier_type in cls._registry:
            raise ValueError(
                f"Tier '{tier_type}' is already registered."
            )

        def loader() -> type[SecondaryTierManager]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[tier_type] = loader

    @classmethod
    def create_secondary_tier(
        cls,
        tier_config: dict,
        primary_kv_view: "memoryview",
        offloading_spec: "OffloadingSpec",
    ) -> SecondaryTierManager:
        tier_cls = cls.get_tier_class(tier_config)
        config = tier_config.copy()
        tier_type = config.pop("type")
        return tier_cls(
            offloading_spec=offloading_spec,
            primary_kv_view=primary_kv_view,
            tier_type=tier_type,
            **config,
        )

    @classmethod
    def get_tier_class(cls, tier_config: dict) -> type[SecondaryTierManager]:
        tier_type = tier_config.get("type")
        if not tier_type:
            raise ValueError(
                "Secondary tier configuration must include 'type'"
            )
        if tier_type not in cls._registry:
            raise ValueError(
                f"Unknown secondary tier type: {tier_type!r}. "
                f"Supported types: {list(cls._registry)}"
            )
        return cls._registry[tier_type]()

    @classmethod
    def reset(cls) -> None:
        cls._registry.clear()

    @classmethod
    def known_tiers(cls) -> list[str]:
        return list(cls._registry.keys())


# Built-in shell tier — smoke test that the factory wiring works.
class ExampleSecondaryTierManager(SecondaryTierManager):
    """C256 shell — no actual storage; always MISSes lookups.

    Returns MISS for every block; raises NotImplementedError on
    actual store/load. Useful as a smoke test that the registry
    + factory plumbing works end-to-end.
    """

    def lookup(
        self, key: "OffloadKey", req_context: ReqContext
    ) -> str:  # type: ignore[override]
        from my_vllm.v1.kv_offload.base import LookupResult
        return LookupResult.MISS

    def submit_store(self, job_metadata: JobMetadata) -> None:
        raise NotImplementedError(
            "ExampleSecondaryTierManager.submit_store is a C256 shell."
        )

    def submit_load(self, job_metadata: JobMetadata) -> None:
        raise NotImplementedError(
            "ExampleSecondaryTierManager.submit_load is a C256 shell."
        )

    def get_finished_jobs(self) -> Iterable[JobResult]:
        return ()

    def on_schedule_end(self) -> None:
        return None

    def has_pending_work(self) -> bool:
        return False

    def reset_cache(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


# Pre-register the example shell.
SecondaryTierFactory.register_tier(
    "example",
    "my_vllm.v1.kv_offload.tiering.factory",
    "ExampleSecondaryTierManager",
)


__all__ = [
    "ExampleSecondaryTierManager",
    "SecondaryTierFactory",
]