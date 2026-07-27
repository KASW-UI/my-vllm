from my_vllm.runtime.tokenizer import load_tokenizer
from my_vllm.runtime.model_loader import load_model
from my_vllm.engine.state_manager import StateManager
from my_vllm.engine.llm_engine import LLMEngine
from my_vllm.worker.gpu_worker import Worker
from my_vllm.executor.uniproc_executor import UniProcExecutor


def test_llm_engine_single_request():
    model_name = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m"
    tokenizer = load_tokenizer(model_name)
    hf_model = load_model(model_name)

    worker = Worker(hf_model, hf_model.device)
    worker.init_device()
    executor = UniProcExecutor(worker)
    sm = StateManager(executor, hf_model.device)
    engine = LLMEngine(sm, tokenizer)

    result = engine.generate("Once upon a time", max_new_tokens=20)
    assert "text" in result
    assert len(result["text"]) > len("Once upon a time")
    assert result["output_tokens"] >= 1
    assert result["output_tokens"] <= 20
    print(f"LLMEngine: {result['finish_reason']} "
          f"({result['output_tokens']} tokens)")
    print(f"  {result['text'][:80]}...")


def test_llm_engine_multiple_requests():
    model_name = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m"
    tokenizer = load_tokenizer(model_name)
    hf_model = load_model(model_name)

    worker = Worker(hf_model, hf_model.device)
    worker.init_device()
    executor = UniProcExecutor(worker)
    sm = StateManager(executor, hf_model.device)
    engine = LLMEngine(sm, tokenizer)

    engine.add_request("Hello world", max_new_tokens=10)
    engine.add_request("The meaning of life is", max_new_tokens=10)

    results = []
    while engine.has_work():
        results.extend(engine.step())

    assert len(results) == 2
    for r in results:
        assert r["output_tokens"] >= 1
        print(f"[{r['request_id']}] {r['finish_reason']} "
              f"({r['output_tokens']} tokens)")
