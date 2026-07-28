"""C028: tests for Executor ABC and UniProcExecutor."""

from my_vllm.v1.executor.abstract import Executor
from my_vllm.v1.executor.uniproc_executor import UniProcExecutor


class DummyWorker:
    def execute_model(self, scheduler_output):
        from my_vllm.v1.outputs import ModelRunnerOutput
        return ModelRunnerOutput(req_ids=[], sampled_token_ids={})

    def get_step(self, rid):
        return 0

    def cleanup(self, rid):
        pass


class TestExecutorABC:
    def test_executor_is_abstract(self):
        import pytest
        with pytest.raises(TypeError):
            Executor()

    def test_get_class_uni(self):
        cls = Executor.get_class("uni")
        assert cls is UniProcExecutor

    def test_get_class_default(self):
        cls = Executor.get_class()
        assert cls is UniProcExecutor

    def test_get_class_mp_fallback(self):
        cls = Executor.get_class("mp")
        assert cls is UniProcExecutor  # stub: falls back to uni

    def test_uses_ray_false(self):
        assert Executor.uses_ray is False


class TestUniProcExecutor:
    def test_init(self):
        worker = DummyWorker()
        ex = UniProcExecutor(worker)
        assert ex.worker is worker

    def test_execute_model(self):
        worker = DummyWorker()
        ex = UniProcExecutor(worker)
        result = ex.execute_model(None)
        assert result is not None

    def test_get_worker(self):
        worker = DummyWorker()
        ex = UniProcExecutor(worker)
        assert ex.get_worker() is worker

    def test_initialize_cache_noop(self):
        worker = DummyWorker()
        ex = UniProcExecutor(worker)
        ex.initialize_cache(num_kv_blocks=64, block_size=16)
