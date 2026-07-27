"""Executor ABC — mirrors vLLM V1 v1/executor/abstract.py.

C017: adds execute_model(SchedulerOutput) → ModelRunnerOutput
as the primary execution interface.
"""

from abc import ABC, abstractmethod

from my_vllm.v1.outputs import ModelRunnerOutput


class Executor(ABC):
    """Abstract executor managing model execution on one or more devices."""

    @abstractmethod
    def execute_model(
        self, scheduler_output
    ) -> ModelRunnerOutput:
        """Execute the model on the given scheduler output.

        Args:
            scheduler_output: structured scheduling metadata.

        Returns:
            ModelRunnerOutput with per-request sampled token IDs.
        """
        ...

    @abstractmethod
    def get_worker(self):
        ...
