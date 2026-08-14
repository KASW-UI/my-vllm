"""Base classes for weight transfer engines.

C243: closes the A-level "跨节点 Weight Transfer" gap from
``phase3-distributed-runtime-plan.md``.

Mirrors ``vllm/distributed/weight_transfer/base.py`` (446 lines → ~280 in
my-vllm).  The abstractions here are the *contract* every backend
(NCCL / IPC / sparse-NCCL) implements:

* :class:`ParamMeta` — name / dtype / shape for one parameter, used in
  metadata-only round-trips.
* :class:`WeightSource` — abstract re-iterable stream of ``(name, tensor)``
  pairs that the trainer engine iterates each round.
* :class:`ModuleSource` — concrete :class:`WeightSource` over
  ``module.named_parameters()`` (handles plain tensors; for FSDP/DSets,
  the trainer-side engine is expected to call ``materialize_full_tensor``
  via its own ``post_iter_func``).
* :class:`WeightTransferEngine` — abstract **worker-side** engine.  Each
  backend owns its ``init_transfer_engine`` / ``receive_weights`` /
  ``shutdown`` lifecycle, and the lifecycle hooks
  ``start_weight_update`` / ``finish_weight_update`` that bracket the
  checkpoint load.
* :class:`TrainerWeightTransferEngine` — abstract **trainer-side** engine.
  Constructed via ``trainer_init(...)`` factory classmethod; then calls
  ``send_weights()`` each round to drive the handshake.
* :class:`VLLMWeightSyncClient` — :func:`typing.Protocol` describing the
  four-method control plane a trainer engine talks to the inference
  worker through (mirrors the inference engine's RLHF HTTP routes).

Scope reductions vs vLLM:

* No layerwise reload hooks (``vllm.model_executor.model_loader.reload``
  is not vendored — OPT-125m fits in memory).
* No MTP completeness check (``vllm.model_executor.model_loader.mtp_validation``
  is not vendored — OPT isn't a draft-target model).
* No ``WeightTransferInitRequest`` / ``WeightTransferUpdateRequest``
  wrapper dataclasses (those wrap dicts for HTTP transport; my-vllm
  has no RLHF HTTP routes since the API server was deleted in C151).
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, runtime_checkable

import torch

if TYPE_CHECKING:
    from my_vllm.config.weight_transfer import WeightTransferConfig


TInitInfo = TypeVar("TInitInfo", bound="WeightTransferInitInfo")
TUpdateInfo = TypeVar("TUpdateInfo", bound="WeightTransferUpdateInfo")


def materialize_full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return a full, locally-materialized tensor ready to send.

    FSDP shards (DTensors) expose ``full_tensor()``, a collective
    all-gather; regular tensors do not and are returned unchanged.
    Trainer engines call this at send time so the (potentially
    expensive) gather happens exactly once.

    my-vllm scope note: no FSDP/DTensor vendored yet, so this is the
    identity function on plain tensors; kept for forward-compat with
    a future FSDP path.
    """
    full_tensor = getattr(tensor, "full_tensor", None)
    return full_tensor() if callable(full_tensor) else tensor


@dataclass(frozen=True)
class ParamMeta:
    """Name / wire dtype / full (HF) shape for one output parameter."""

    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]


class WeightSource(ABC):
    """A re-iterable source of the trainer's weights, handed to a trainer engine.

    Two channels:

    * ``metadata()`` — ``(name, wire dtype, full shape)`` for every
      parameter, *without* transferring.  Cheap when shapes are known
      locally (FSDP ``DTensor`` global shape); may be expensive on first
      call for backends that must materialize to learn shapes.
    * iteration — yields fully-materialized ``(name, tensor)`` pairs,
      one at a time.  Materializing is typically a collective (FSDP
      ``full_tensor()``), so every trainer rank must iterate the same
      source in the same order in lockstep, or ranks deadlock.
    """

    @abstractmethod
    def metadata(self) -> list[ParamMeta]:
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[str, torch.Tensor]]:
        raise NotImplementedError


class ModuleSource(WeightSource):
    """:class:`WeightSource` over ``module.named_parameters()`` — the common case.

    Handles both plain dense modules and FSDP-sharded ones with no
    special casing: iteration all-gathers each ``DTensor`` via
    ``full_tensor()`` (a collective) and passes regular tensors through.
    ``metadata()`` reads the *global* ``.shape`` / ``.dtype``, so it
    never triggers a gather.
    """

    def __init__(self, module: torch.nn.Module) -> None:
        self._module = module

    def metadata(self) -> list[ParamMeta]:
        return [
            ParamMeta(name, p.dtype, tuple(p.shape))
            for name, p in self._module.named_parameters()
        ]

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor]]:
        for name, param in self._module.named_parameters():
            yield name, materialize_full_tensor(param)


# Base protocols for backend-specific dataclasses
@dataclass
class WeightTransferInitInfo(ABC):  # noqa: B024
    """Base class for backend-specific initialization info."""

    pass


@dataclass
class TrainerInitInfo(WeightTransferInitInfo):
    """Base trainer-side init info: which trainer rank drives the transfer.

    ``rank`` is this trainer process's rank, provided **explicitly** by
    the caller — the engine does not read it from a global process
    group, which is ambiguous once several groups (FSDP / TP / PP / EP)
    exist.  Rank 0 is always the sender: only it opens the endpoint
    and drives the inference-side RPCs, while every rank still runs
    the trainer-side collectives.  Backend subclasses add their own
    (positional) fields; ``rank`` is keyword-only so that ordering
    never conflicts.
    """

    rank: int = field(kw_only=True)

    @property
    def is_sender(self) -> bool:
        return self.rank == 0


@dataclass
class WeightTransferUpdateInfo(ABC):  # noqa: B024
    """Base class for backend-specific weight update info."""

    pass


class WeightTransferEngine(ABC, Generic[TInitInfo, TUpdateInfo]):
    """
    Base class for weight transfer engines that handle transport of model
    weights from a trainer to inference workers.

    This abstraction separates weight transfer transport logic from the
    worker implementation, allowing different backends (NCCL, CUDA IPC)
    to be plugged in.

    Each engine owns its full weight-update lifecycle:
    ``start_weight_update`` → ``update_weights`` → ``finish_weight_update``.
    Layerwise reloading (used by checkpoint-format engines) is opted into
    per engine by running it inside ``start_weight_update`` /
    ``finish_weight_update``.  Engines that apply weights in place
    (e.g. sparse patches) leave those methods as no-ops.

    Subclasses should define:
        init_info_cls: Type of backend-specific initialization info
        update_info_cls: Type of backend-specific update info
    """

    # Subclasses should override these class attributes
    init_info_cls: type[TInitInfo]
    update_info_cls: type[TUpdateInfo]

    supports_draft_weight_update: bool = True

    def __init__(
        self,
        config: "WeightTransferConfig",
        vllm_config: Any,
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        """
        Initialize the weight transfer engine.

        Args:
            config: The configuration for the weight transfer engine
            vllm_config: The full vLLM config (provides parallel/model config)
            device: The device this worker's model lives on
            model: The local model instance which will receive the weights
        """
        self.config = config
        self.vllm_config = vllm_config
        self.parallel_config = vllm_config.parallel_config
        self.model_config = vllm_config.model_config
        self.device = device
        self.model = model
        self._default_model_config = self.model_config
        self._default_model = model

    def set_weight_update_target(
        self,
        model: torch.nn.Module,
        model_config: Any,
    ) -> None:
        """Set the model that will receive the active weight update."""
        self.model = model
        self.model_config = model_config

    def reset_weight_update_target(self) -> None:
        """Restore weight updates to the engine's default target model."""
        self.model = self._default_model
        self.model_config = self._default_model_config

    def parse_init_info(self, init_dict: dict[str, Any]) -> TInitInfo:
        """Construct typed init info from dict with validation.

        Raises:
            ValueError: If ``init_dict`` is invalid for this backend.
        """
        try:
            return self.init_info_cls(**init_dict)
        except TypeError as e:
            raise ValueError(
                f"Invalid init_info for {self.__class__.__name__}: {e}"
            ) from e

    def parse_update_info(self, update_dict: dict[str, Any]) -> TUpdateInfo:
        """Construct typed update info from dict with validation.

        Raises:
            ValueError: If ``update_dict`` is invalid for this backend.
        """
        try:
            return self.update_info_cls(**update_dict)
        except TypeError as e:
            raise ValueError(
                f"Invalid update_info for {self.__class__.__name__}: {e}"
            ) from e

    @abstractmethod
    def init_transfer_engine(self, init_info: TInitInfo) -> None:
        """Initialize the weight transfer mechanism (called once at startup)."""
        raise NotImplementedError

    @abstractmethod
    def start_weight_update(self) -> None:
        """Prepare the engine for a new weight update.

        my-vllm: no-op (no layerwise reload vendored).  Subclasses may
        override to initialize per-update buffering.
        """
        raise NotImplementedError

    @abstractmethod
    def finish_weight_update(self) -> None:
        """Finalize the current weight update.

        my-vllm: no-op (no layerwise reload vendored).  Subclasses may
        override to finalize per-update buffering.
        """
        raise NotImplementedError

    def update_weights(self, update_info: dict[str, Any]) -> None:
        """Receive one weight update chunk and load it into the model."""
        typed_update_info = self.parse_update_info(update_info)
        self.receive_weights(typed_update_info)
        # NCCL broadcast / IPC paths may be asynchronous.  Synchronize here
        # so the next step uses the new weights.
        torch.accelerator.synchronize()

    @abstractmethod
    def receive_weights(self, update_info: TUpdateInfo) -> None:
        """Receive weights from the trainer and load them into the model."""
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the weight transfer engine (called at worker teardown)."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def trainer_send_weights(
        iterator: Iterator[Any],
        trainer_args: dict[str, Any] | Any,
    ) -> None:
        """Send weights from trainer to inference workers.

        This is a static method that can be called from the trainer
        process to send weights to all inference workers.

        Args:
            iterator: Iterator of backend-specific items to send.
            trainer_args: Dictionary (or backend-specific dataclass)
                containing backend-specific arguments needed to send
                weights.  The structure depends on the backend:

                * NCCL: contains ``group`` (process-group handle),
                  ``src`` (source rank), ``packed`` (enable double-buffered
                  packed transfer), etc.
                * IPC: contains ``send_mode`` (callable taking a
                  ``IPCWeightTransferUpdateInfo``), ``packed``.
        """
        raise NotImplementedError


@runtime_checkable
class VLLMWeightSyncClient(Protocol):
    """Trainer-side stub for the inference engine's weight-sync control plane.

    Mirrors the weight-sync methods that the inference engine exposes
    (``EngineClient`` / the HTTP RLHF routes / Ray actors).  A
    :class:`TrainerWeightTransferEngine` drives the full handshake
    through this protocol so trainer code never has to know the
    transport.

    All methods are synchronous and accept plain dicts (matching what
    the inference side already accepts).  Concurrency that some
    backends need (e.g. NCCL must run ``update_weights`` concurrently
    with the trainer-side broadcast) is the engine's responsibility,
    not the client's, so the protocol stays a flat four-method surface
    that any wrapper can implement.

    The protocol is structural (PEP 544), so user implementations need
    only define these four methods — no import or subclassing required.
    """

    def init_weight_transfer_engine(self, init_info: dict[str, Any]) -> None: ...

    def start_weight_update(self) -> None: ...

    def update_weights(self, update_info: dict[str, Any]) -> None: ...

    def finish_weight_update(self) -> None: ...


class TrainerWeightTransferEngine(ABC, Generic[TInitInfo]):
    """Trainer-side weight transfer engine.

    Symmetric to :class:`WeightTransferEngine` but lives in the training
    process.  Constructed via the ``trainer_init`` factory classmethod;
    carries any backend-specific state (NCCL communicators, IPC device
    info, transfer plans) on ``self``.  The :class:`WeightSource` is
    required at ``trainer_init``, then replayed each round by the
    no-argument :meth:`send_weights`.

    Multi-rank trainers: ``trainer_init`` and ``send_weights`` are
    called on *every* trainer rank.  Rank 0 is the sender, resolved
    once at ``trainer_init`` into ``is_sender``.  Non-sender ranks
    still run every collective (iterating the source, metadata
    export, IPC handle all-gather) so the group stays aligned, but
    each engine explicitly guards the control-plane RPCs and the
    transmit on ``self.is_sender``, so only the sender touches the
    client.

    Subclasses should define:
        init_info_cls: Type of backend-specific trainer init info
    """

    # Subclasses should override these class attributes
    init_info_cls: type[TInitInfo]

    def __init__(
        self,
        *,
        client: "VLLMWeightSyncClient",
        source: "WeightSource",
        is_sender: bool = True,
    ) -> None:
        self.is_sender = is_sender
        # The real client is held on every rank; each engine only
        # *calls* it when ``is_sender``, so non-sender ranks never
        # touch the wire.
        self.client = client
        self.source = source

    @classmethod
    @abstractmethod
    def trainer_init(
        cls,
        init_info: TInitInfo,
        *,
        client: "VLLMWeightSyncClient",
        source: "WeightSource",
    ) -> "TrainerWeightTransferEngine":
        """Rendezvous with the inference side and return a ready instance.

        Called on every trainer rank.  The sender drives the full
        handshake via ``client`` (build the worker-side init info, call
        ``client.init_weight_transfer_engine``, open the trainer-side
        endpoint); non-sender ranks skip the rendezvous and the RPC.
        ``source`` is stored on ``self.source``; after return,
        :meth:`send_weights` is callable.
        """
        raise NotImplementedError

    @abstractmethod
    def send_weights(self) -> None:
        """Push ``self.source``'s weights to inference workers and drive the
        full update round trip: ``start_weight_update``,
        ``update_weights`` (run concurrently with the trainer-side
        broadcast when the backend requires it), then
        ``finish_weight_update``.  Called on every trainer rank.
        """
        raise NotImplementedError

    def shutdown(self) -> None:
        """Tear down communicators / process groups.  Default no-op."""
        pass
