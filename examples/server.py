"""Start the API server."""

from my_vllm.runtime.tokenizer import load_tokenizer
from my_vllm.runtime.model_loader import load_model
from my_vllm.engine.state_manager import StateManager
from my_vllm.engine.llm_engine import LLMEngine
from my_vllm.worker.gpu_worker import Worker
from my_vllm.executor.uniproc_executor import UniProcExecutor
from my_vllm.server.api_server import run_server


def main():
    model_name = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m"
    print(f"Loading model: {model_name}")
    tokenizer = load_tokenizer(model_name)
    hf_model = load_model(model_name)

    worker = Worker(hf_model, hf_model.device)
    worker.init_device()
    executor = UniProcExecutor(worker)
    sm = StateManager(executor, hf_model.device)
    engine = LLMEngine(sm, tokenizer)

    run_server(engine, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
