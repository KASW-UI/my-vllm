from dataclasses import dataclass, field
from enum import IntEnum


class RequestStatus(IntEnum):
    WAITING = 0
    RUNNING = 1
    FINISHED_STOPPED = 2
    FINISHED_LENGTH_CAPPED = 3


@dataclass
class Request:
    request_id: str
    prompt_token_ids: list[int] = field(default_factory=list)
    output_token_ids: list[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING

    def __repr__(self) -> str:
        return (
            f"Request(id={self.request_id!r}, "
            f"status={self.status.name}, "
            f"prompt_tokens={len(self.prompt_token_ids)}, "
            f"output_tokens={len(self.output_token_ids)})"
        )
