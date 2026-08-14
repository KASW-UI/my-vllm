"""KVCacheSpec registry — pluggable spec ↔ manager mapping.

Mirrors vLLM V1 vllm/v1/kv_cache_spec_registry.py (subset).

This module lets out-of-tree platforms / attention backends register
custom ``KVCacheSpec`` subclasses together with their
``SingleTypeKVCacheManager`` implementation and a uniform-type base
class, without modifying vLLM core.

my-vllm NOTE (C254): we register ``FullAttentionSpec`` →
``FullAttentionManager`` at module import time so the default
(OPT-style dense) path resolves without an explicit
``register_all_kvcache_specs`` call. The other spec types are
**shell registrations** — they're declared as registered specs with
their expected manager class names, but the manager class bodies are
no-op shells (declared in ``single_type_kv_cache_manager.py``). This
keeps the registry lookup functional for routing / dispatch tests
while preserving the scope-reduction policy from docs/alignment-plan.md
(Dense-only my-vllm).

Registration is global and one-shot per process: re-registering the
same spec with the same metadata is idempotent; re-registering with
different metadata is a hard error to surface misconfiguration early.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from my_vllm.v1.core.single_type_kv_cache_manager import (
        SingleTypeKVCacheManager,
    )
    from my_vllm.v1.kv_cache_interface import KVCacheSpec


# ---------------------------------------------------------------------------
# Registry data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KVCacheSpecMetadata:
    """Metadata stored for every registered ``KVCacheSpec`` subclass.

    Mirrors vLLM V1 vllm/v1/kv_cache_spec_registry.py:KVCacheSpecMetadata.
    """

    kvcache_spec_cls: type
    manager_class: type
    uniform_type_base_spec: type
    """Base class for grouping compatibility (specs sharing a base merge
    into one KV cache group)."""


# Module-level registry. Keyed by the *concrete* spec class so the
# ``__mro__`` walk in ``get_manager_class`` / ``get_uniform_type_base_spec``
# finds the closest registered ancestor first.
_REGISTRY_KVCACHESPEC_LIST: dict[type, KVCacheSpecMetadata] = {}

# Track whether the default ``FullAttentionSpec`` → ``FullAttentionManager``
# registration has been installed. Tests can call ``reset_registry()`` to
# wipe + re-bootstrap from a clean slate.
_DEFAULT_REGISTERED = False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class KVCacheSpecRegistry:
    """Global registry for ``KVCacheSpec`` ↔ ``SingleTypeKVCacheManager``."""

    @classmethod
    def register(
        cls,
        kvcache_spec_cls: type,
        manager_class: type | None = None,
        uniform_type_base_spec: type | None = None,
    ) -> None:
        """Register a ``KVCacheSpec`` subclass with its manager + base.

        Args:
            kvcache_spec_cls: The spec class to register.
            manager_class: The ``SingleTypeKVCacheManager`` subclass to use
                for this spec. **Required.**
            uniform_type_base_spec: The base class for grouping
                compatibility. If None, defaults to the spec class itself
                (built-in base specs use this).

        Raises:
            AssertionError: if ``manager_class`` is None, if the spec
                doesn't inherit from the declared base, or if a prior
                registration exists with conflicting metadata.
        """
        assert manager_class is not None, (
            "manager_class is required when registering a KVCacheSpec"
        )
        if uniform_type_base_spec is None:
            uniform_type_base_spec = kvcache_spec_cls
        assert issubclass(kvcache_spec_cls, uniform_type_base_spec), (
            f"{kvcache_spec_cls.__name__} must inherit from its declared "
            f"uniform_type_base_spec {uniform_type_base_spec.__name__}."
        )
        if kvcache_spec_cls in _REGISTRY_KVCACHESPEC_LIST:
            existing = _REGISTRY_KVCACHESPEC_LIST[kvcache_spec_cls]
            same = (
                existing.manager_class == manager_class
                and existing.uniform_type_base_spec == uniform_type_base_spec
            )
            assert same, (
                f"Conflicting registration for KVCacheSpec: "
                f"{kvcache_spec_cls.__name__}"
            )
            return
        _REGISTRY_KVCACHESPEC_LIST[kvcache_spec_cls] = KVCacheSpecMetadata(
            kvcache_spec_cls=kvcache_spec_cls,
            manager_class=manager_class,
            uniform_type_base_spec=uniform_type_base_spec,
        )

    @classmethod
    def get_manager_class(cls, kvcache_spec: "KVCacheSpec") -> type | None:
        """Return the ``SingleTypeKVCacheManager`` class for this spec.

        Walks the spec's ``__mro__`` so subclasses with custom registered
        bases resolve correctly (e.g. an RSWASpec inherits FullAttentionSpec
        but is registered with ``uniform_type_base_spec=FullAttentionSpec``).

        Returns None if no registered ancestor exists.
        """
        cls._ensure_default_registered()
        for base in type(kvcache_spec).__mro__:
            if base in _REGISTRY_KVCACHESPEC_LIST:
                return _REGISTRY_KVCACHESPEC_LIST[base].manager_class
        return None

    @classmethod
    def get_uniform_type_base_spec(
        cls, kvcache_spec: "KVCacheSpec"
    ) -> type | None:
        """Return the base spec class for grouping compatibility.

        Walks the spec's ``__mro__`` to find the closest registered
        ancestor (the one with the most specific ``uniform_type_base_spec``
        declared). Returns None if no registered ancestor exists.
        """
        cls._ensure_default_registered()
        for base in type(kvcache_spec).__mro__:
            if base in _REGISTRY_KVCACHESPEC_LIST:
                return _REGISTRY_KVCACHESPEC_LIST[base].uniform_type_base_spec
        return None

    @classmethod
    def check_kv_cache_spec_registry(
        cls, kv_cache_specs: dict[str, "KVCacheSpec"]
    ) -> None:
        """Validate that every layer's spec is registered.

        Used at model-load time to surface unknown spec types early.
        """
        cls._ensure_default_registered()
        for layer_name, spec in kv_cache_specs.items():
            if cls.get_uniform_type_base_spec(spec) is None:
                raise ValueError(
                    f"Unsupported KV cache spec type for layer {layer_name}: "
                    f"{type(spec).__name__}. Register it using "
                    f"@register_kv_cache_spec decorator."
                )
            if cls.get_manager_class(spec) is None:
                raise ValueError(
                    f"No manager found for KV cache spec type for layer "
                    f"{layer_name}: {type(spec).__name__}. Register it "
                    f"using @register_kv_cache_spec decorator."
                )

    @classmethod
    def reset(cls) -> None:
        """Wipe the registry (test helper).

        After ``reset()`` the default ``FullAttentionSpec`` registration
        is also wiped; the next call to ``get_manager_class`` /
        ``get_uniform_type_base_spec`` re-bootstraps it via
        ``_ensure_default_registered``.
        """
        global _DEFAULT_REGISTERED
        _REGISTRY_KVCACHESPEC_LIST.clear()
        _DEFAULT_REGISTERED = False

    @classmethod
    def _ensure_default_registered(cls) -> None:
        """Lazy-install the default FullAttentionSpec → FullAttentionManager.

        Imported lazily to avoid the circular
        ``kv_cache_interface ↔ single_type_kv_cache_manager ↔ registry``
        cycle at module load.
        """
        global _DEFAULT_REGISTERED
        if _DEFAULT_REGISTERED:
            return
        # Lazy imports break the cycle.
        from my_vllm.v1.core.single_type_kv_cache_manager import (
            FullAttentionManager,
        )
        from my_vllm.v1.kv_cache_interface import FullAttentionSpec

        cls.register(
            FullAttentionSpec,
            FullAttentionManager,
            uniform_type_base_spec=FullAttentionSpec,
        )
        _DEFAULT_REGISTERED = True


# ---------------------------------------------------------------------------
# Decorator API (matches vLLM)
# ---------------------------------------------------------------------------


def register_kv_cache_spec(
    manager_class: type | None = None,
    uniform_type_base_spec: type | None = None,
):
    """Decorator to register a custom ``KVCacheSpec`` class.

    Examples:
        @register_kv_cache_spec(
            manager_class=FullAttentionManager,
            uniform_type_base_spec=FullAttentionSpec,
        )
        @dataclass(frozen=True, kw_only=True)
        class CustomFullAttentionSpec(FullAttentionSpec):
            pass
    """

    def decorator(kvcache_spec_cls: type) -> type:
        KVCacheSpecRegistry.register(
            kvcache_spec_cls=kvcache_spec_cls,
            manager_class=manager_class,
            uniform_type_base_spec=uniform_type_base_spec,
        )
        return kvcache_spec_cls

    return decorator


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------


__all__ = [
    "KVCacheSpecMetadata",
    "KVCacheSpecRegistry",
    "register_kv_cache_spec",
]