"""IPC-based weight transfer engine using CUDA IPC for communication.

C243: closes the A-level "跨节点 Weight Transfer" gap from
``phase3-distributed-runtime-plan.md``.

Mirrors ``vllm/distributed/weight_transfer/ipc_engine.py`` (501 lines
→ ~370 in my-vllm).

The IPC backend transfers weights via CUDA IPC handles — the trainer
serializes each weight tensor with
:func:`torch.multiprocessing.reductions.reduce_tensor` and shares the
handle with the worker over the inference engine's RPC layer; the
worker reconstructs the tensor with
:func:`torch.multiprocessing.reductions.rebuild_cuda_tensor` and feeds
it into ``model.load_weights``.

This is the natural backend when trainer and worker share the same
node (no cross-node NCCL available) but live in separate processes
(so the in-process ``model.state_dict()`` shortcut can't be used).

Scope reductions vs vLLM:

* **No Ray backend.**  vLLM's ``send_mode="ray"`` calls
  ``ray.get([handle.update_weights.remote(...) for handle in
  handles])`` against Ray actor handles.  my-vllm has no Ray — deleted
  in C151.
* **No HTTP backend.**  vLLM's ``send_mode="http"`` POSTs pickled IPC
  handles to ``/update_weights`` and requires
  ``VLLM_ALLOW_INSECURE_SERIALIZATION=1``.  my-vllm has no RLHF HTTP
  routes — OpenAI API server deleted in C151.
* **Only callable ``send_mode``** remains, accepting a function that
  takes an :class:`IPCWeightTransferUpdateInfo` and dispatches it to
  the worker.  This is what the unit tests use; in production a real
  vLLM deployment would subclass this with a Ray or HTTP transport.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor

if TYPE_CHECKING:
    from my_vllm.config import VllmConfig

from my_vllm.config.weight_transfer import WeightTransferConfig
from my_vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)
from my_vllm.distributed.weight_transfer.packed_tensor import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    packed_ipc_consumer,
    packed_ipc_producer,
)

__all__ = [
    "IPCTrainerSendWeightsArgs",
    "IPCWeightTransferInitInfo",
    "IPCWeightTransferUpdateInfo",
    "IPCWeightTransferEngine",
]


@dataclass
class IPCTrainerSendWeightsArgs:
    """Arguments for :meth:`IPCWeightTransferEngine.trainer_send_weights`."""

    send_mode: str | Callable[["IPCWeightTransferUpdateInfo"], None]
    """How to send updates to vLLM.  Only the ``callable`` form is
    supported in my-vllm: a function that receives an
    :class:`IPCWeightTransferUpdateInfo` and performs the send (e.g.
    dispatches it over the inference engine's RPC)."""
    packed: bool = False
    """Whether to use packed tensor transfer for bounded-memory chunking."""
    packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES
    """Size in bytes for each packed tensor buffer when ``packed=True``."""

    def __post_init__(self) -> None:
        """Validate that ``send_mode`` is a callable."""
        if not callable(self.send_mode):
            raise ValueError(
                "my-vllm only supports `callable` send_mode for IPC weight "
                f"transfer (got {self.send_mode!r}).  Ray / HTTP modes are "
                "not vendored."
            )


@dataclass
class IPCWeightTransferInitInfo(WeightTransferInitInfo):
    """Initialization info for IPC weight transfer backend.

    No init needed for IPC — the constructor takes no args beyond the
    default ``WeightTransferInitInfo`` base.
    """

    pass


@dataclass
class IPCWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Update info for IPC weight transfer backend."""

    names: list[str]
    dtype_names: list[str]
    shapes: list[list[int]]
    ipc_handles: list[dict[str, tuple]] | dict[str, tuple]
    """IPC handles mapping physical GPU UUID to
    :func:`rebuild_cuda_tensor` args tuple.  For non-packed mode: list
    of per-parameter handle dicts.  For packed mode: single handle dict
    for the packed buffer."""
    tensor_sizes: list[int] | None = None
    """Per-parameter sizes in bytes within the packed buffer.  Required
    when ``packed=True``, unused otherwise."""
    packed: bool = False
    """Whether this update uses packed tensor format."""

    def __post_init__(self) -> None:
        if self.ipc_handles is None:
            raise ValueError("`ipc_handles` must be provided")
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
        if (
            not self.packed
            and isinstance(self.ipc_handles, list)
            and len(self.ipc_handles) != num_params
        ):
            raise ValueError(
                f"`ipc_handles` should be of the same size as `names`: "
                f"got {len(self.ipc_handles)} and {len(self.names)}"
            )
        if self.packed and self.tensor_sizes is None:
            raise ValueError("`tensor_sizes` is required when packed=True")


class IPCWeightTransferEngine(
    WeightTransferEngine[IPCWeightTransferInitInfo, IPCWeightTransferUpdateInfo]
):
    """Weight transfer engine using CUDA IPC for communication between
    trainer and workers.

    Uses CUDA IPC to transfer weights from the trainer (rank 0) to all
    inference workers in a process group.  IPC handles are used to
    share memory between processes on the same node.
    """

    # Define backend-specific dataclass types
    init_info_cls = IPCWeightTransferInitInfo
    update_info_cls = IPCWeightTransferUpdateInfo

    def __init__(
        self,
        config: WeightTransferConfig,
        vllm_config: "VllmConfig",
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)

    def init_transfer_engine(self, init_info: IPCWeightTransferInitInfo) -> None:
        """No-op (IPC backend needs no rendezvous — IPC handles are
        exchanged on the inference RPC layer)."""
        pass

    def start_weight_update(self) -> None:
        """No-op (no layerwise reload in my-vllm)."""
        pass

    def finish_weight_update(self) -> None:
        """No-op (no layerwise reload in my-vllm)."""
        pass

    def receive_weights(self, update_info: IPCWeightTransferUpdateInfo) -> None:
        """Receive weights from the trainer via CUDA IPC handles and load them.

        The receive path is no longer wrapped in
        ``with torch.device(self.device)`` by the caller, so the
        current device is not guaranteed to match ``self.device``.  The
        IPC tensors must be rebuilt on the device the model lives on.
        """
        device_index = self.device.index

        if update_info.packed:
            assert update_info.tensor_sizes is not None
            assert isinstance(update_info.ipc_handles, dict)
            weights = packed_ipc_consumer(
                ipc_handle=update_info.ipc_handles,
                names=update_info.names,
                shapes=update_info.shapes,
                dtype_names=update_info.dtype_names,
                tensor_sizes=update_info.tensor_sizes,
                device_index=device_index,
            )
        else:
            assert isinstance(update_info.ipc_handles, list)
            weights = []
            for name, ipc_handle in zip(
                update_info.names,
                update_info.ipc_handles,
            ):
                props = torch.cuda.get_device_properties(device_index)
                physical_gpu_id = str(props.uuid)

                if physical_gpu_id not in ipc_handle:
                    raise ValueError(
                        f"IPC handle not found for GPU UUID "
                        f"{physical_gpu_id}. "
                        f"Available UUIDs: {list(ipc_handle.keys())}"
                    )

                args = ipc_handle[physical_gpu_id]
                list_args = list(args)
                # Index 6 of the args from reduce_tensor is the device_index.
                # We need to overwrite it with the receiver's device index.
                list_args[6] = device_index
                weight = rebuild_cuda_tensor(*list_args)
                weights.append((name, weight))

        self.model.load_weights(weights)

    def shutdown(self) -> None:
        pass

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[tuple[str, torch.Tensor]],
        trainer_args: dict[str, Any] | IPCTrainerSendWeightsArgs,
    ) -> None:
        """Send weights from trainer to inference workers via CUDA IPC.

        Supports two transfer strategies:

        * **Non-packed** (default): all weights in a single call.
        * **Packed** (``packed=True``): chunked transfer with bounded
          GPU memory.

        For multi-GPU training, all ranks must call this method in
        parallel.  IPC handles are all-gathered across ranks and
        merged so that each vLLM worker can find its own GPU UUID.
        Only rank 0 sends the payload to vLLM.

        .. note::
            This method calls ``update_weights`` internally.  The
            caller must call ``start_weight_update`` before and
            ``finish_weight_update`` after this method.

        Args:
            iterator: Iterator of ``(name, tensor)`` pairs.  For
                multi-GPU, each rank should yield the full tensor on
                its own GPU (e.g. via FSDP ``full_tensor()``).
            trainer_args: :class:`IPCTrainerSendWeightsArgs` or
                equivalent dict.
        """
        args = (
            IPCTrainerSendWeightsArgs(**trainer_args)
            if isinstance(trainer_args, dict)
            else trainer_args
        )
        device_index = torch.accelerator.current_device_index()
        gpu_uuid = str(torch.cuda.get_device_properties(device_index).uuid)
        if args.packed:
            IPCWeightTransferEngine._send_packed(iterator, args, gpu_uuid)
        else:
            IPCWeightTransferEngine._send_unpacked(iterator, args, gpu_uuid)

    @staticmethod
    def _is_rank_zero() -> bool:
        """Return True if this is rank 0 or no distributed group exists."""
        if not torch.distributed.is_initialized():
            return True
        return torch.distributed.get_rank() == 0

    @staticmethod
    def _all_gather_and_merge_handles(
        handles: list[dict[str, tuple]],
    ) -> list[dict[str, tuple]]:
        """All-gather and merge IPC handle dicts across ranks in one call.

        Each rank contributes a list of ``{gpu_uuid: ipc_args}`` dicts
        (one per parameter or one per chunk).  A single
        :func:`all_gather_object` collects every rank's full list, then
        rank 0 merges per-index so each dict maps every GPU UUID to
        its args.

        Non-rank-0 returns a list of empty dicts.  No-op (returns
        ``handles`` unchanged) when no distributed group exists.
        """
        if (
            not torch.distributed.is_initialized()
            or torch.distributed.get_world_size() == 1
        ):
            return handles

        world_size = torch.distributed.get_world_size()
        gathered: list[list[dict[str, tuple]] | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered, handles)
        torch.distributed.barrier()
        torch.cuda.synchronize()

        if torch.distributed.get_rank() == 0:
            merged: list[dict[str, tuple]] = []
            for param_idx in range(len(handles)):
                m: dict[str, tuple] = {}
                for rank_handles in gathered:
                    if rank_handles is not None:
                        m.update(rank_handles[param_idx])
                merged.append(m)
            return merged
        return [{} for _ in handles]

    @staticmethod
    def _post_send_sync() -> None:
        """Barrier + ``ipc_collect`` after a send; no-op if single-GPU."""
        if (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            torch.distributed.barrier()
        torch.cuda.ipc_collect()

    @staticmethod
    def _send_unpacked(
        iterator: Iterator[tuple[str, torch.Tensor]],
        args: IPCTrainerSendWeightsArgs,
        gpu_uuid: str,
    ) -> None:
        """Send all weights in a single call (non-packed mode)."""
        names: list[str] = []
        dtype_names: list[str] = []
        shapes: list[list[int]] = []
        ipc_handles: list[dict[str, tuple]] = []
        # Hold strong refs to every contiguous copy until the send +
        # post-send sync completes.  ``reduce_tensor``'s returned args
        # do NOT keep storage alive, and non-contiguous inputs
        # allocate fresh storage in ``.contiguous()`` that would
        # otherwise be GC'd before the consumer opens the IPC handle.
        weight_refs: list[torch.Tensor] = []

        for name, tensor in iterator:
            names.append(name)
            dtype_names.append(str(tensor.dtype).split(".")[-1])
            shapes.append(list(tensor.shape))

            weight = tensor.detach().contiguous()
            weight_refs.append(weight)
            _, ipc_args = reduce_tensor(weight)
            ipc_handles.append({gpu_uuid: ipc_args})

        ipc_handles = IPCWeightTransferEngine._all_gather_and_merge_handles(ipc_handles)

        if IPCWeightTransferEngine._is_rank_zero():
            update_info = IPCWeightTransferUpdateInfo(
                names=names,
                dtype_names=dtype_names,
                shapes=shapes,
                ipc_handles=ipc_handles,
                packed=False,
            )
            assert callable(args.send_mode)
            args.send_mode(update_info)

        IPCWeightTransferEngine._post_send_sync()

    @staticmethod
    def _send_packed(
        iterator: Iterator[tuple[str, torch.Tensor]],
        args: IPCTrainerSendWeightsArgs,
        gpu_uuid: str,
    ) -> None:
        """Send weights in bounded-memory chunks (packed mode)."""
        post_iter_func: Callable = lambda item: item[1]

        for chunk in packed_ipc_producer(
            iterator=iterator,
            gpu_uuid=gpu_uuid,
            post_iter_func=post_iter_func,
            buffer_size_bytes=args.packed_buffer_size_bytes,
        ):
            ipc_handle = IPCWeightTransferEngine._all_gather_and_merge_handles(
                [chunk.ipc_handle]
            )[0]

            if IPCWeightTransferEngine._is_rank_zero():
                update_info = IPCWeightTransferUpdateInfo(
                    names=chunk.names,
                    dtype_names=chunk.dtype_names,
                    shapes=chunk.shapes,
                    ipc_handles=ipc_handle,
                    tensor_sizes=chunk.tensor_sizes,
                    packed=True,
                )
                assert callable(args.send_mode)
                args.send_mode(update_info)

            IPCWeightTransferEngine._post_send_sync()
