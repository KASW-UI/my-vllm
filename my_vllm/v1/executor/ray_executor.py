"""RayDistributedExecutor — Ray-based distributed executor. Aligns with vLLM.

C091: Ray actor lifecycle + PlacementGroup for cross-node GPU inference.
"""

from my_vllm.v1.executor.abstract import Executor
from my_vllm.v1.outputs import ModelRunnerOutput


class RayDistributedExecutor(Executor):
    """Ray-based executor — aligns with vLLM RayDistributedExecutor.

    Uses Ray actors to wrap Worker processes across nodes.
    Ray handles cross-node scheduling, fault recovery, GPU placement.
    """

    uses_ray: bool = True
    supports_pp: bool = True

    def __init__(
        self,
        model_name: str = "facebook/opt-125m",
        dtype_str: str = "bfloat16",
        world_size: int = 1,
        placement_group=None,
    ):
        super().__init__()
        self.model_name = model_name
        self.dtype_str = dtype_str
        self.world_size = world_size
        self.placement_group = placement_group
        self._workers: list = []

    def _init_executor(self) -> None:
        """Initialize Ray cluster and create worker actors.

        Real vLLM calls initialize_ray_cluster() then _init_workers_ray().
        Education version: stubs for Ray API calls.
        """
        pg = self.placement_group
        if pg is None:
            try:
                import ray
                if not ray.is_initialized():
                    pg = initialize_ray_cluster()
                self.placement_group = pg
            except ImportError:
                pass

    def execute_model(self, scheduler_output) -> ModelRunnerOutput | None:
        """Dispatch via Ray actors, collect results."""
        if not self._workers:
            return None
        try:
            import ray
            futures = [w.execute_model.remote(scheduler_output)
                        for w in self._workers]
            outputs = ray.get(futures)
            return outputs[0] if outputs else None
        except ImportError:
            return None

    def get_worker(self):
        raise NotImplementedError("RayDistributedExecutor: use execute_model()")

    def shutdown(self) -> None:
        for w in self._workers:
            try:
                import ray
                ray.kill(w)
            except Exception:
                pass
        self._workers.clear()


def initialize_ray_cluster(
    placement_strategy: str = "STRICT_SPREAD",
) -> object:
    """Initialize Ray cluster with PlacementGroup.

    Aligns with vLLM initialize_ray_cluster.

    PlacementGroup strategies:
      - STRICT_SPREAD: max 1 actor per node → cross-node
      - STRICT_PACK: all actors on same node → low latency
      - PACK: same node preferred, spill if needed → default
    """
    try:
        import ray
        if not ray.is_initialized():
            ray.init()
        # Placeholder: in real vLLM creates ray.util.placement_group
        return None
    except ImportError:
        return None
