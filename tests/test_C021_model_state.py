"""C021: tests for ModelState interface + DefaultModelState."""

import torch

from my_vllm.v1.worker.gpu.model_states.interface import ModelState
from my_vllm.v1.worker.gpu.model_states.default import DefaultModelState
from my_vllm.v1.worker.gpu_worker import Worker, _DirectModelState

MODEL = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m"


class TestModelStateInterface:
    def test_interface_is_abstract(self):
        import pytest
        with pytest.raises(TypeError):
            ModelState()

    def test_direct_model_state(self):
        from my_vllm.runtime.model_loader import load_model
        model = load_model(MODEL)
        device = next(model.parameters()).device
        ms = _DirectModelState(model, device)
        assert ms.load_model() is model
        assert ms.get_device() == device

    def test_worker_with_model_state(self):
        from my_vllm.runtime.model_loader import load_model
        model = load_model(MODEL)
        device = next(model.parameters()).device
        ms = _DirectModelState(model, device)
        worker = Worker(model_state=ms)
        worker.init_device()
        assert worker.model_state is ms
        assert worker._runner() is not None
        assert worker.get_step("nonexistent") == 0

    def test_worker_legacy_ctor(self):
        from my_vllm.runtime.model_loader import load_model
        model = load_model(MODEL)
        device = next(model.parameters()).device
        worker = Worker(model, device)
        assert isinstance(worker.model_state, _DirectModelState)
        worker.init_device()
        assert worker.get_step("nonexistent") == 0


class TestDefaultModelState:
    def test_init(self):
        ms = DefaultModelState(MODEL)
        assert ms.model_name == MODEL

    def test_load_model(self):
        ms = DefaultModelState(MODEL)
        model = ms.load_model()
        assert model is not None
        assert ms._model is model  # cached

    def test_get_device(self):
        ms = DefaultModelState(MODEL)
        ms.load_model()
        assert isinstance(ms.get_device(), torch.device)

    def test_get_dtype(self):
        ms = DefaultModelState(MODEL)
        assert ms.get_dtype() == torch.bfloat16

    def test_tokenizer(self):
        ms = DefaultModelState(MODEL)
        tok = ms.tokenizer
        assert tok is not None
        ids = tok.encode("hello")
        assert len(ids) > 0


class TestWorkerWithDefaultModelState:
    def test_full_pipeline(self):
        ms = DefaultModelState(MODEL)
        worker = Worker(model_state=ms)
        worker.init_device()

        from my_vllm.v1.executor.uniproc_executor import UniProcExecutor
        from my_vllm.v1.engine.core import EngineCore
        from my_vllm.v1.engine.llm_engine import LLMEngine

        executor = UniProcExecutor(worker)
        ec = EngineCore(executor, ms.get_device(), max_new_tokens=10)
        engine = LLMEngine(ec, ms.tokenizer)

        result = engine.generate("Hello world", max_new_tokens=10)
        assert result["output_tokens"] >= 1
        assert len(result["text"]) > len("Hello world")
