"""Start the API server."""

from my_vllm.entrypoints.llm import LLM, load_tokenizer
from my_vllm.config.vllm import EngineConfig
from my_vllm.v1.engine.llm_engine import LLMEngine
from my_vllm.v1.executor.uniproc_executor import UniProcExecutor
from my_vllm.entrypoints.openai.api_server import run_server


def main():
    model_name = "facebook/opt-125m"
    print(f"Loading model: {model_name}")
    tokenizer = load_tokenizer(model_name)

    cfg = EngineConfig()
    cfg.model_config.model_name = model_name

    engine = LLMEngine(
        vllm_config=cfg,
        executor_class=UniProcExecutor,
        tokenizer=tokenizer,
        busy_loop=True,
    )

    run_server(engine, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()