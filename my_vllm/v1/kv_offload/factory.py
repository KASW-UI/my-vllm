"""KV cache offloading — spec factory + registry.

Mirrors vLLM V1 vllm/v1/kv_offload/factory.py (subset).

C255: a lazy-loading registry that maps spec names to
`OffloadingSpec` subclasses. Two paths:

1. **Pre-registered** via `register_spec(name, module_path,
   class_name)` — `importlib.import_module` is deferred until the
   first `get_spec_cls` / `create_spec` call, so backend imports
   don't slow down my-vllm startup.
2. **One-shot** via `extra_config["spec_module_path"]` +
   `extra_config["spec_name"]` — for ad-hoc / external specs that
   weren't pre-registered.

Spec lookup precedence:
- `extra_config["spec_name"]` (default `"CPUOffloadingSpec"`).
- If pre-registered → use the registry entry.
- Else if `extra_config["spec_module_path"]` is set → dynamic import.
- Else raise `ValueError`.

C255 ships the registry + factory but not concrete specs (those
land with C256 / C257). The base class registration +
`MyDummyOffloadingSpec` test-only path is enough to exercise the
factory plumbing.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import Any

from my_vllm.v1.kv_offload.base import OffloadingSpec
from my_vllm.v1.kv_offload.config import OffloadingConfig


_DEFAULT_SPEC_NAME = "CPUOffloadingSpec"


class OffloadingSpecFactory:
    """Lazy-loading registry for `OffloadingSpec` subclasses."""

    _registry: dict[str, Callable[[], type[OffloadingSpec]]] = {}

    @classmethod
    def register_spec(
        cls, name: str, module_path: str, class_name: str
    ) -> None:
        """Register a spec by name + module path + class name.

        `module_path` is imported lazily on first lookup so backend
        imports don't run at my-vllm startup.
        """
        if name in cls._registry:
            raise ValueError(f"Spec '{name}' is already registered.")

        def loader() -> type[OffloadingSpec]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def get_spec_cls(
        cls, extra_config: Mapping[str, Any]
    ) -> type[OffloadingSpec]:
        """Resolve the spec class for the given extra_config.

        Resolution order:
        1. `extra_config["spec_name"]` (default `CPUOffloadingSpec`).
        2. If pre-registered → use the registry entry.
        3. Else if `extra_config["spec_module_path"]` is set →
           import that module and look up the class.
        4. Else raise `ValueError`.
        """
        spec_name = extra_config.get("spec_name", _DEFAULT_SPEC_NAME)
        if spec_name in cls._registry:
            return cls._registry[spec_name]()
        spec_module_path = extra_config.get("spec_module_path")
        if spec_module_path is None:
            raise ValueError(f"Unsupported spec type: {spec_name}")
        spec_module = importlib.import_module(spec_module_path)
        spec_cls = getattr(spec_module, spec_name)
        assert issubclass(spec_cls, OffloadingSpec), (
            f"Resolved spec class {spec_name!r} from {spec_module_path} "
            "is not a subclass of OffloadingSpec."
        )
        return spec_cls

    @classmethod
    def create_spec(cls, config: OffloadingConfig) -> OffloadingSpec:
        """Instantiate the spec for the given config.

        Logs the chosen spec name at info level.
        """
        spec_name = config.extra_config.get(
            "spec_name", _DEFAULT_SPEC_NAME
        )
        spec_cls = cls.get_spec_cls(config.extra_config)
        return spec_cls(config)

    @classmethod
    def reset(cls) -> None:
        """Wipe the registry (test helper)."""
        cls._registry.clear()

    @classmethod
    def known_specs(cls) -> list[str]:
        """Return the list of registered spec names."""
        return list(cls._registry.keys())


__all__ = ["OffloadingSpecFactory"]