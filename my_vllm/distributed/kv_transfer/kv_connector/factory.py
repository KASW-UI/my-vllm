"""KVConnectorFactory — connector selection by name.

Mirrors vLLM V1 distributed/kv_transfer/kv_connector/factory.py:1.

C133: minimal factory — only `NixlConnector` is registered, matching
my-vllm's career-relevant subset (PD disaggregation via NIXL
shared-memory local backend; RDMA / NIXL-GPU-direct are out of
scope). vLLM registers ~14 connectors here; my-vllm keeps the
`register_connector` / `create_connector` API so future
connectors can be added without API changes.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from my_vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
    supports_hma,
)

if TYPE_CHECKING:
    from my_vllm.config.vllm import VllmConfig
    from my_vllm.config.kv_transfer import KVTransferConfig


class KVConnectorFactory:
    """Registry mapping connector name → lazily-loaded class."""

    _registry: dict[str, Callable[[], type[KVConnectorBase_V1]]] = {}

    @classmethod
    def register_connector(
        cls, name: str, module_path: str, class_name: str
    ) -> None:
        """Register a connector with a lazy-loading module + class name."""
        if name in cls._registry:
            raise ValueError(
                f"Connector {name!r} is already registered."
            )

        def loader() -> type[KVConnectorBase_V1]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def get_connector_class_by_name(
        cls, connector_name: str
    ) -> type[KVConnectorBase_V1]:
        """Get a registered connector class by name.

        Raises ValueError if the connector is not registered.
        """
        if connector_name not in cls._registry:
            raise ValueError(
                f"Connector {connector_name!r} is not registered."
            )
        return cls._registry[connector_name]()

    @classmethod
    def get_connector_class(
        cls, kv_transfer_config: "KVTransferConfig"
    ) -> type[KVConnectorBase_V1]:
        """Resolve a connector class from KVTransferConfig.kv_connector."""
        connector_name = kv_transfer_config.kv_connector
        if connector_name is None:
            raise ValueError(
                "Connector name is not set in KVTransferConfig"
            )
        return cls.get_connector_class_by_name(connector_name)

    @classmethod
    def create_connector(
        cls,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Any = None,
    ) -> KVConnectorBase_V1:
        """Instantiate the configured connector.

        Args:
            vllm_config: top-level config (must have
                `kv_transfer_config` set).
            role: SCHEDULER or WORKER.
            kv_cache_config: optional KV cache config (passed through
                to the connector constructor).
        """
        kv_transfer_config = vllm_config.kv_transfer_config
        if kv_transfer_config is None:
            raise ValueError(
                "kv_transfer_config must be set to create a connector"
            )
        connector_cls = cls.get_connector_class(kv_transfer_config)

        # HMA gate (mirrors vLLM factory.py:39): if HMA is enabled and
        # the connector does not support it, raise. my-vLLM does not
        # yet enable HMA by default, so this is a no-op in practice.
        from my_vllm.config.scheduler import SchedulerConfig  # noqa: F401

        hma_enabled = False  # my-vLLM: HMA gate deferred to P1-3
        if hma_enabled and not supports_hma(connector_cls):
            raise ValueError(
                f"Connector {connector_cls.__name__} does not support HMA"
            )

        return cast(
            KVConnectorBase_V1,
            connector_cls(vllm_config, role, kv_cache_config),
        )


# Register concrete connectors known to my-vllm.
# Adding a new backend = one extra `register_connector` call here,
# plus the implementation module.

KVConnectorFactory.register_connector(
    "NixlConnector",
    "my_vllm.distributed.kv_transfer.kv_connector.v1.nixl",
    "NixlConnector",
)


__all__ = ["KVConnectorFactory"]