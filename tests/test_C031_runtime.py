"""C031: final architecture completeness verification.

Validates that all C031 target files exist and control-flow /
data-flow / component-boundary constraints are satisfied.
"""

import os

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


class TestC031FileCompleteness:
    """Verify every file in the C031 target tree exists."""

    C031_FILES = [
        "my_vllm/__init__.py",
        "my_vllm/v1/__init__.py",
        "my_vllm/v1/request.py",
        "my_vllm/v1/outputs.py",
        "my_vllm/v1/utils.py",
        "my_vllm/v1/serial_utils.py",
        "my_vllm/v1/engine/__init__.py",
        "my_vllm/v1/engine/core.py",
        "my_vllm/v1/engine/llm_engine.py",
        "my_vllm/v1/engine/async_llm.py",
        "my_vllm/v1/engine/output_processor.py",
        "my_vllm/v1/engine/input_processor.py",
        "my_vllm/v1/engine/detokenizer.py",
        "my_vllm/v1/engine/exceptions.py",
        "my_vllm/v1/core/__init__.py",
        "my_vllm/v1/core/block_pool.py",
        "my_vllm/v1/core/kv_cache_manager.py",
        "my_vllm/v1/core/kv_cache_utils.py",
        "my_vllm/v1/core/sched/__init__.py",
        "my_vllm/v1/core/sched/scheduler.py",
        "my_vllm/v1/core/sched/output.py",
        "my_vllm/v1/core/sched/request_queue.py",
        "my_vllm/v1/kv_cache_interface.py",
        "my_vllm/v1/worker/__init__.py",
        "my_vllm/v1/worker/worker_base.py",
        "my_vllm/v1/worker/gpu_worker.py",
        "my_vllm/v1/worker/gpu_model_runner.py",
        "my_vllm/v1/worker/gpu/__init__.py",
        "my_vllm/v1/worker/gpu/input_batch.py",
        "my_vllm/v1/worker/gpu/block_table.py",
        "my_vllm/v1/worker/gpu/states.py",
        "my_vllm/v1/worker/gpu/model_states/__init__.py",
        "my_vllm/v1/worker/gpu/model_states/interface.py",
        "my_vllm/v1/worker/gpu/model_states/default.py",
        "my_vllm/v1/executor/__init__.py",
        "my_vllm/v1/executor/abstract.py",
        "my_vllm/v1/executor/uniproc_executor.py",
        "my_vllm/v1/attention/__init__.py",
        "my_vllm/v1/attention/backend.py",
        "my_vllm/v1/attention/selector.py",
        "my_vllm/v1/attention/backends/__init__.py",
        "my_vllm/v1/attention/backends/flash_attn.py",
        "my_vllm/v1/attention/ops/__init__.py",
        "my_vllm/v1/attention/ops/paged_attn.py",
        "my_vllm/v1/sample/__init__.py",
        "my_vllm/v1/sample/sampler.py",
        "my_vllm/v1/sample/metadata.py",
        "my_vllm/v1/sample/logits_processor/__init__.py",
        "my_vllm/v1/sample/logits_processor/interface.py",
        "my_vllm/v1/sample/logits_processor/builtin.py",
        "my_vllm/v1/sample/logits_processor/state.py",
        "my_vllm/v1/metrics/__init__.py",
        "my_vllm/model_executor/__init__.py",
        "my_vllm/model_executor/model_loader/__init__.py",
        "my_vllm/model_executor/model_loader/base_loader.py",
        "my_vllm/model_executor/model_loader/default_loader.py",
        "my_vllm/model_executor/model_loader/utils.py",
        "my_vllm/model_executor/model_loader/weight_utils.py",
        "my_vllm/model_executor/models/opt.py",
        "my_vllm/entrypoints/__init__.py",
        "my_vllm/entrypoints/llm.py",
        "my_vllm/entrypoints/openai/__init__.py",
        "my_vllm/entrypoints/openai/api_server.py",
    ]

    def test_c031_files_exist(self):
        missing = []
        for f in self.C031_FILES:
            path = os.path.join(PROJECT_ROOT, f)
            if not os.path.isfile(path):
                missing.append(f)
        assert not missing, (
            f"Missing C031 files: {missing}"
        )


class TestC031ControlFlowBoundaries:
    """Verify component boundaries match C031 architecture."""

    def test_llm_engine_does_not_import_scheduler(self):
        import ast
        path = os.path.join(
            PROJECT_ROOT, "my_vllm/v1/engine/llm_engine.py"
        )
        with open(path) as f:
            tree = ast.parse(f.read())
        imports = [
            node.names[0].name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
        ] + [
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        ]
        assert not any("scheduler" in i for i in imports), (
            "LLMEngine should not import Scheduler directly"
        )

    def test_engine_core_exposes_step(self):
        from my_vllm.v1.engine.core import EngineCore
        assert hasattr(EngineCore, "step")
        assert hasattr(EngineCore, "add_request")

    def test_executor_has_execute_model(self):
        from my_vllm.v1.executor.abstract import Executor
        assert hasattr(Executor, "execute_model")

    def test_scheduler_returns_structured_output(self):
        from my_vllm.v1.core.sched.output import SchedulerOutput
        so = SchedulerOutput()
        assert hasattr(so, "scheduled_new_reqs")
        assert hasattr(so, "scheduled_cached_reqs")


class TestC031DataFlow:
    """Verify structured dataclass types exist for inter-component communication."""

    def test_model_runner_output(self):
        from my_vllm.v1.outputs import ModelRunnerOutput
        out = ModelRunnerOutput(
            req_ids=["r1"], req_id_to_index={"r1": 0},
            sampled_token_ids=[5],
        )
        assert out.req_ids == ["r1"]

    def test_sampler_output_field(self):
        from my_vllm.v1.outputs import SamplerOutput
        out = SamplerOutput(
            sampled_token_ids=[5],
        )
        assert out.sampled_token_ids is not None

    def test_request_output(self):
        from my_vllm.v1.engine.output_processor import RequestOutput
        out = RequestOutput(
            request_id="r1", prompt_text="Hello",
            output_text=" world", finished=True,
            finish_reason="FINISHED_STOPPED",
            token_ids=[5, 6],
        )
        assert out.is_finished()
        assert out.output_token_ids == [5, 6]


class TestC031Config:
    def test_engine_config(self):
        from my_vllm.config.vllm import VllmConfig
        cfg = VllmConfig()
        assert cfg.cache_config.block_size == 16
        assert cfg.scheduler_config.max_num_batched_tokens == 2048

    def test_serial_utils(self):
        from my_vllm.v1.serial_utils import MsgpackEncoder, MsgpackDecoder
        encoder = MsgpackEncoder()
        data = encoder.encode({"a": 1, "b": 2})
        decoder = MsgpackDecoder()
        decoded = decoder.decode(data)
        assert decoded == {"a": 1, "b": 2}

    def test_metrics_module(self):
        import my_vllm.v1.metrics
