"""NCCL-based (dense) weight transfer engine.

C243: closes the A-level "跨节点 Weight Transfer" gap from
``phase3-distributed-runtime-plan.md``.

Mirrors ``vllm/distributed/weight_transfer/nccl_engine.py`` (272 lines
→ ~190 in my-vllm).

The dense NCCL engine is the canonical cross-process weight transfer
backend.  The trainer (rank 0) iterates :class:`ModuleSource` /
:class:`WeightSource`, calls :func:`packed_nccl_broadcast_producer` (or
the unpacked one-by-one path) on the chosen process group, while each
worker (rank ≥ 1) calls :func:`packed_nccl_broadcast_consumer` /
:meth:`receive_weights` and feeds the resulting ``(name, tensor)``
pairs to ``model.load_weights``.

Scope reductions vs vLLM:

* No layerwise reload — :meth:`start_weight_update` /
  :meth:`finish_weight_update` are no-ops (my-vllm doesn't vendor
  ``vllm/model_executor/model_loader/reload``).
* No MTP completeness check — wrapped call site doesn't need
  ``disable_mtp_completeness_check``.
* Uses :func:`torch.distributed.broadcast` directly (the underlying
  process group exposes ``.broadcast(tensor, src=...)`` whether it's
  vanilla :class:`dist.ProcessGroup` or :class:`PyNcclCommunicator`).
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from my_vllm.config import VllmConfig
    from my_vllm.config.weight_transfer import WeightTransferConfig

from my_vllm.config.weight_transfer import WeightTransferConfig
from my_vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferUpdateInfo,
)
from my_vllm.distributed.weight_transfer.nccl_common import (
    NCCLWeightTransferInitInfo,
    trainer_init,
    worker_init_process_group,
)
from my_vllm.distributed.weight_transfer.packed_tensor import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    DEFAULT_PACKED_NUM_BUFFERS,
    packed_nccl_broadcast_consumer,
)

__all__ = [
    "NCCLWeightTransferInitInfo",
    "NCCLTrainerSendWeightsArgs",
    "NCCLWeightTransferUpdateInfo",
    "NCCLWeightTransferEngine",
]


def _broadcast(group: Any, tensor: torch.Tensor, src: int = 0) -> None:
    """Broadcast a tensor using either a PyNcclCommunicator-style API
    (which accepts ``stream=``) or vanilla :func:`torch.distributed.broadcast`
    (which doesn't).

    my-vllm only ships the vanilla path; the PyNcclCommunicator path
    is kept for forward-compat with a future C extension (mirroring
    vLLM's ``pynccl`` wrapper).
    """
    if hasattr(group, "broadcast") and callable(group.broadcast):
        try:
            # PyNcclCommunicator-style: broadcast(tensor, src=src, stream=...)
            group.broadcast(
                tensor, src=src, stream=torch.cuda.current_stream()
            )
            return
        except TypeError:
            pass
    # Vanilla torch.distributed path.
    import torch.distributed as dist

    dist.broadcast(tensor, src=src, group=group)


@dataclass
class NCCLTrainerSendWeightsArgs:
    """Arguments for :meth:`NCCLWeightTransferEngine.trainer_send_weights`."""

    group: Any
    """Process group (anything with ``.broadcast(tensor, src=...)`` — a
    vanilla ``dist.ProcessGroup`` works the same as vLLM's
    ``PyNcclCommunicator``)."""
    src: int = 0
    """Source rank (default 0, trainer is typically rank 0)."""
    post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor] | None = None
    """Optional function applied to each ``(name, tensor)`` pair before
    broadcasting.  If ``None``, extracts just the tensor."""
    packed: bool = False
    """Whether to use packed tensor broadcasting for efficiency.  When
    ``True``, multiple tensors are batched together before broadcasting
    to reduce per-call NCCL overhead."""
    stream: torch.cuda.Stream | None = None
    """CUDA stream to use for broadcasting if ``packed=False``.  When
    ``packed=True``, new streams are created internally for each
    buffer."""
    packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES
    """Size in bytes for each packed tensor buffer.  Must match the
    value used in :class:`NCCLWeightTransferUpdateInfo`."""
    packed_num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS
    """Number of buffers for double/triple buffering during packed
    transfer.  Must match the value used in
    :class:`NCCLWeightTransferUpdateInfo`."""


@dataclass
class NCCLWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Update info for NCCL weight transfer backend."""

    names: list[str]
    dtype_names: list[str]
    shapes: list[list[int]]
    packed: bool = False
    """Whether to use packed tensor broadcasting for efficiency."""
    packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES
    """Size in bytes for each packed tensor buffer.  Both producer and
    consumer must use the same value."""
    packed_num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS
    """Number of buffers for double/triple buffering during packed
    transfer.  Both producer and consumer must use the same value."""

    def __post_init__(self) -> None:
        """Validate that all lists have the same length."""
        num_params = len(self.names)
        if len(self.dtype_names) != num_params:
            raise ValueError(
                f"`dtype_names` should be of the same size as `names`: "
                f"got {len(self.dtype_names)} and {len(self.names)}"
            )
        if len(self.shapes) != num_params:
            raise ValueError(
                f"`shapes` should be of the same size as `names`: "
                f"got {len(self.shapes)} and {len(self.names)}"
            )


class NCCLWeightTransferEngine(
    WeightTransferEngine[NCCLWeightTransferInitInfo, NCCLWeightTransferUpdateInfo]
):
    """
    Weight transfer engine using NCCL for communication between trainer and
    workers.

    Uses NCCL broadcast operations (via the underlying
    :class:`torch.distributed.ProcessGroup`) to transfer dense
    checkpoint-format weights from the trainer (rank 0) to all inference
    workers in a process group.  Received weights are loaded via
    ``model.load_weights``.

    Supports both unpacked (one tensor per broadcast) and packed
    (double-buffered 1 GB chunks) modes via the ``packed`` field of
    :class:`NCCLWeightTransferUpdateInfo`.
    """

    # Define backend-specific dataclass types
    init_info_cls = NCCLWeightTransferInitInfo
    update_info_cls = NCCLWeightTransferUpdateInfo

    def __init__(
        self,
        config: WeightTransferConfig,
        vllm_config: "VllmConfig",
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)
        self.model_update_group: Any = None

    def init_transfer_engine(self, init_info: NCCLWeightTransferInitInfo) -> None:
        """Initialize the trainer↔worker process group."""
        self.model_update_group = worker_init_process_group(
            init_info, self.parallel_config
        )

    def start_weight_update(self) -> None:
        """No-op (no layerwise reload in my-vllm)."""
        pass

    def finish_weight_update(self) -> None:
        """No-op (no layerwise reload in my-vllm)."""
        pass

    def receive_weights(self, update_info: NCCLWeightTransferUpdateInfo) -> None:
        """Receive weights from trainer via NCCL broadcast.

        If ``update_info.packed`` is True, uses packed tensor
        broadcasting (efficient batched transfer).  Otherwise, uses
        simple one-by-one broadcasting.
        """
        if self.model_update_group is None:
            raise RuntimeError(
                "NCCL weight transfer not initialized. "
                "Call init_transfer_engine() first."
            )

        if update_info.packed:
            # Build iterator of (name, (shape, dtype)) from update_info.
            def state_dict_info_iterator():
                for name, dtype_name, shape in zip(
                    update_info.names, update_info.dtype_names, update_info.shapes
                ):
                    dtype = getattr(torch, dtype_name)
                    yield (name, (shape, dtype))

            packed_nccl_broadcast_consumer(
                iterator=state_dict_info_iterator(),
                group=self.model_update_group,
                src=0,
                post_unpack_func=self.model.load_weights,
                buffer_size_bytes=update_info.packed_buffer_size_bytes,
                num_buffers=update_info.packed_num_buffers,
                device=self.device,
            )
        else:
            # Use simple one-by-one broadcasting.
            for name, dtype_name, shape in zip(
                update_info.names, update_info.dtype_names, update_info.shapes
            ):
                dtype = getattr(torch, dtype_name)
                weight = torch.empty(shape, dtype=dtype, device=self.device)
                _broadcast(self.model_update_group, weight, src=0)
                self.model.load_weights([(name, weight)])
                del weight

    def shutdown(self) -> None:
        """Drop the process group reference (does not call
        ``destroy_process_group`` — the worker is responsible for the
        global group lifecycle)."""
        self.model_update_group = None

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[tuple[str, torch.Tensor]],
        trainer_args: dict[str, Any] | NCCLTrainerSendWeightsArgs,
    ) -> None:
        """Broadcast dense weights from trainer to vLLM workers.

        Args:
            iterator: Iterator of model parameters.  Returns
                ``(name, tensor)`` tuples.
            trainer_args: Dictionary or :class:`NCCLTrainerSendWeightsArgs`
                instance containing NCCL-specific arguments.  If a dict,
                should contain keys from :class:`NCCLTrainerSendWeightsArgs`.

        Example:
            >>> from my_vllm.distributed.weight_transfer.nccl_engine import (
            ...     NCCLWeightTransferEngine,
            ...     NCCLTrainerSendWeightsArgs,
            ... )
            >>> param_iter = ((n, p) for n, p in model.named_parameters())
            >>> args = NCCLTrainerSendWeightsArgs(group=group, packed=True)
            >>> NCCLWeightTransferEngine.trainer_send_weights(param_iter, args)
        """
        # Parse trainer args - accept either dict or dataclass instance
        if isinstance(trainer_args, dict):
            args = NCCLTrainerSendWeightsArgs(**trainer_args)
        else:
            args = trainer_args

        if args.post_iter_func is None:
            # Default: extract just the tensor from (name, tensor) tuple
            post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor] = (
                lambda x: x[1]
            )
        else:
            post_iter_func = args.post_iter_func

        if args.packed:
            # Use packed tensor broadcasting for efficiency.
            from my_vllm.distributed.weight_transfer.packed_tensor import (
                packed_nccl_broadcast_producer,
            )

            packed_nccl_broadcast_producer(
                iterator=iterator,
                group=args.group,
                src=args.src,
                post_iter_func=post_iter_func,
                buffer_size_bytes=args.packed_buffer_size_bytes,
                num_buffers=args.packed_num_buffers,
            )
        else:
            # Use simple one-by-one broadcasting.
            for item in iterator:
                tensor = post_iter_func(item)
                _broadcast(
                    args.group,
                    tensor,
                    src=args.src,
                )

    # Trainer-side process-group setup. Delegates to the shared helper so
    # the sparse engine can reuse the exact same rendezvous without
    # subclassing.
    trainer_init = staticmethod(trainer_init)
