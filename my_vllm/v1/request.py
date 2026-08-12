import enum
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from my_vllm.v1.engine import FinishReason
from my_vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from my_vllm.lora.request import LoRARequest


class RequestStatus(enum.IntEnum):
    """Status of a request — mirrors vLLM V1 v1/request.py:343."""

    WAITING = enum.auto()
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    WAITING_FOR_STREAMING_REQ = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()
    FINISHED_ERROR = enum.auto()
    FINISHED_REPETITION = enum.auto()

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> FinishReason:
        return _FINISHED_REASON_MAP.get(status, FinishReason.STOP)


_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ERROR: FinishReason.ERROR,
    RequestStatus.FINISHED_REPETITION: FinishReason.REPETITION,
}


@dataclass
class SamplingParams:
    """Minimal sampling parameters — placeholder for C015."""

    max_tokens: int = 50
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1


class Request:
    """Request object tracking the full lifecycle of a generation.

    Mirrors vLLM V1's v1/request.py:59 — Request is a regular class
    (not a dataclass) to support ConstantList read-only views and
    controlled mutations via append_output_token_ids().
    """

    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int] | None = None,
        sampling_params: SamplingParams | None = None,
        arrival_time: float | None = None,
        priority: int = 0,
        block_hasher: Callable[["Request"], list[bytes]] | None = None,
        lora_request: "LoRARequest | None" = None,
    ) -> None:
        self.request_id = request_id
        self.prompt_token_ids: list[int] = prompt_token_ids or []
        self.status = RequestStatus.WAITING

        self.sampling_params = sampling_params
        self.pooling_params = None
        self.lora_request = lora_request
        self.tenant_id: str | None = None
        self.arrival_time = (
            arrival_time if arrival_time is not None else time.time()
        )
        self.max_tokens: int = (
            sampling_params.max_tokens if sampling_params is not None else 0
        )

        self.priority: int = priority
        self.num_computed_tokens: int = 0
        self.num_preemptions: int = 0
        self.block_hashes: list[bytes] = []
        self._block_hasher: Callable[[Request], list[bytes]] | None = block_hasher

        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = list(self.prompt_token_ids)

        self.output_token_ids = ConstantList(self._output_token_ids)
        self.all_token_ids = ConstantList(self._all_token_ids)

        self.spec_token_ids: list[int] = []
        self.is_prefill_chunk: bool = False

        self.events: list = []
        self.stop_reason: int | str | None = None

    def compute_block_hashes(self, block_size: int = 16) -> list[bytes]:
        """Compute and cache hash for each block of prompt token IDs."""
        if self.block_hashes:
            return self.block_hashes
        if not self.prompt_token_ids:
            return []
        tokens = self.prompt_token_ids
        num_blocks = (len(tokens) + block_size - 1) // block_size
        self.block_hashes = []
        for i in range(num_blocks):
            start = i * block_size
            end = min(start + block_size, len(tokens))
            block_tokens = tokens[start:end]
            h = hash(tuple(block_tokens))
            self.block_hashes.append(
                h.to_bytes(8, "big", signed=True)
            )
        return self.block_hashes

    def append_output_token_ids(self, token_ids: int | list[int]) -> None:
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def __lt__(self, other: "Request") -> bool:
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.arrival_time != other.arrival_time:
            return self.arrival_time < other.arrival_time
        if self.request_id != other.request_id:
            return self.request_id < other.request_id
        return id(self) < id(other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Request):
            return NotImplemented
        return self.request_id == other.request_id

    def __hash__(self) -> int:
        return hash(self.request_id)

    def __repr__(self) -> str:
        return (
            f"Request(id={self.request_id!r}, "
            f"status={self.status.name}, "
            f"prompt_tokens={len(self.prompt_token_ids)}, "
            f"output_tokens={self.num_output_tokens})"
        )
