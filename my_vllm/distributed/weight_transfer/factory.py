"""Factory for weight transfer engines with lazy loading.

C243: closes the A-level "跨节点 Weight Transfer" gap from
``phase3-distributed-runtime-plan.md``.

Mirrors ``vllm/distributed/weight_transfer/factory.py`` (230 lines →
~120 in my-vllm).

Two factories:

* :class:`WeightTransferEngineFactory` — worker-side: maps a
  :class:`WeightTransferConfig.backend` string to the concrete
  :class:`WeightTransferEngine` subclass (NCCL / IPC / sparse-NCCL).
* :class:`WeightTransferTrainerFactory` — trainer-side: maps a
  backend string to the concrete
  :class:`TrainerWeightTransferEngine` subclass.

Both use a registry pattern with **lazy loading** — engine modules are
only imported when actually needed.  This keeps the import graph
small (e.g. the IPC engine pulls in CUDA IPC machinery only when
``backend="ipc"`` is requested).

Scope reductions vs vLLM:

* No trainer-side engines registered — my-vllm's
  :class:`TrainerWeightTransferEngine` is an ABC whose subclasses all
  require a real trainer framework (FSDP / Megatron); we keep the
  class shell for interface completeness but don't ship a concrete
  trainer engine.  The factory is wired up so that adding one is a
  one-line ``register_engine`` call.
"""

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from my_vllm.distributed.weight_transfer.base import (
    TrainerWeightTransferEngine,
    WeightTransferEngine,
)

if TYPE_CHECKING:
    import torch

    from my_vllm.config import VllmConfig
    from my_vllm.config.weight_transfer import WeightTransferConfig


class WeightTransferEngineFactory:
    """Factory for creating weight transfer engines with lazy loading."""

    _registry: dict[str, Callable[[], type[WeightTransferEngine]]] = {}

    @classmethod
    def register_engine(
        cls,
        name: str,
        module_path_or_cls: str | type[WeightTransferEngine],
        class_name: str | None = None,
    ) -> None:
        """Register an engine with lazy-loading or direct class reference.

        Supports two calling conventions:

        1. Lazy loading: ``register_engine(name, module_path, class_name)``
        2. Direct class: ``register_engine(name, engine_cls)``

        Args:
            name: The name to register the engine under (e.g. ``"nccl"``).
            module_path_or_cls: Either a module path string for lazy
                loading, or the engine class directly.
            class_name: Name of the engine class (required if
                ``module_path_or_cls`` is a string).

        Raises:
            ValueError: If an engine with the same name is already
                registered, or if a module-path registration omits
                ``class_name``.
        """
        if name in cls._registry:
            raise ValueError(f"Weight transfer engine '{name}' is already registered.")

        if isinstance(module_path_or_cls, str):
            # Lazy loading path
            module_path = module_path_or_cls
            if class_name is None:
                raise ValueError(
                    "class_name is required when registering with module path"
                )

            def loader() -> type[WeightTransferEngine]:
                module = importlib.import_module(module_path)
                return getattr(module, class_name)

            cls._registry[name] = loader
        else:
            # Direct class registration
            engine_cls = module_path_or_cls
            cls._registry[name] = lambda: engine_cls

    @classmethod
    def create_engine(
        cls,
        config: "WeightTransferConfig",
        vllm_config: "VllmConfig",
        device: "torch.device",
        model: "torch.nn.Module",
    ) -> WeightTransferEngine:
        """Create a weight transfer engine instance.

        Args:
            config: Weight transfer configuration containing the
                backend name.
            vllm_config: The full vLLM config (provides parallel/model
                config).
            device: The device this worker's model lives on.
            model: The local model instance which will receive the
                weights.

        Returns:
            An initialized weight transfer engine instance.

        Raises:
            ValueError: If the backend is not registered.
        """
        backend = config.backend
        if backend not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Invalid weight transfer backend: {backend}. "
                f"Available engines: {available}"
            )
        engine_cls = cls._registry[backend]()

        return engine_cls(config, vllm_config, device, model)


class WeightTransferTrainerFactory:
    """Factory for creating trainer-side weight transfer engines.

    Parallel to :class:`WeightTransferEngineFactory`, with its own
    lazy-import registry.  The trainer-side and worker-side registries
    are kept separate: they share backend names by convention, but the
    trainer process never instantiates a worker engine and vice versa,
    so unifying them would only couple the import graphs.

    my-vllm scope note: no concrete trainer engines are registered —
    see ``base.py:TrainerWeightTransferEngine`` for the rationale.
    """

    _registry: dict[str, Callable[[], type[TrainerWeightTransferEngine]]] = {}

    @classmethod
    def register_engine(
        cls,
        name: str,
        module_path_or_cls: "str | type[TrainerWeightTransferEngine]",
        class_name: str | None = None,
    ) -> None:
        """Register a trainer engine.  Same conventions as
        :meth:`WeightTransferEngineFactory.register_engine`."""
        if name in cls._registry:
            raise ValueError(
                f"Weight transfer trainer engine '{name}' is already registered."
            )

        if isinstance(module_path_or_cls, str):
            module_path = module_path_or_cls
            if class_name is None:
                raise ValueError(
                    "class_name is required when registering with module path"
                )

            def loader() -> type[TrainerWeightTransferEngine]:
                module = importlib.import_module(module_path)
                return getattr(module, class_name)

            cls._registry[name] = loader
        else:
            engine_cls = module_path_or_cls
            cls._registry[name] = lambda: engine_cls


# Register built-in weight transfer engines here.
# Registration should be centralized to ensure lazy loading —
# engine modules are only imported when actually used.

WeightTransferEngineFactory.register_engine(
    "nccl",
    "my_vllm.distributed.weight_transfer.nccl_engine",
    "NCCLWeightTransferEngine",
)

WeightTransferEngineFactory.register_engine(
    "ipc",
    "my_vllm.distributed.weight_transfer.ipc_engine",
    "IPCWeightTransferEngine",
)

WeightTransferEngineFactory.register_engine(
    "sparse_nccl",
    "my_vllm.distributed.weight_transfer.sparse_nccl_engine",
    "SparseNCCLWeightTransferEngine",
)
