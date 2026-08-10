"""SchedulerInterface — abstract base class for scheduler implementations.

Mirrors vLLM V1 v1/core/sched/interface.py.
C037: standardizes the scheduler contract for future priority/preemption/async variants.

Subclasses must implement:
    schedule() → SchedulerOutput
    update_from_output()
    add_request()
    finish_requests()
    get_num_unfinished_requests()
    has_unfinished_requests()
"""

from abc import ABC, abstractmethod

from my_vllm.v1.core.sched.output import SchedulerOutput
from my_vllm.v1.outputs import EngineCoreOutputs, ModelRunnerOutput
from my_vllm.v1.request import Request, RequestStatus


class SchedulerInterface(ABC):
    """Abstract scheduler contract — mirrors vLLM V1 SchedulerInterface.

    Every scheduler variant (FCFS, Priority, Async) implements this
    interface so EngineCore can swap them without changes.
    """

    @abstractmethod
    def schedule(self) -> SchedulerOutput:
        """Produce a batch of work for the next execution step."""

    @abstractmethod
    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> EngineCoreOutputs:
        """Process model output, detect finished requests."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Enqueue a new request for scheduling."""

    @abstractmethod
    def finish_requests(
        self,
        request_ids: list[str],
        finished_status: RequestStatus,
    ) -> list[Request]:
        """Mark requests as finished and release scheduler state."""

    @abstractmethod
    def get_num_unfinished_requests(self) -> int:
        """Return the count of requests not yet finished (waiting + running)."""

    @abstractmethod
    def has_unfinished_requests(self) -> bool:
        """Return True if there are unfinished requests."""
