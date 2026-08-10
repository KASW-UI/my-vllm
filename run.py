"""CLI entry point — offline inference via V1 engine pipeline.

C032: model loading moved deep into Executor → Worker → GPUModelRunner
chain.  run.py no longer loads the model or creates Worker directly;
it assembles VllmConfig + LLMEngine.
"""

from my_vllm.runtime.tokenizer import load_tokenizer
from my_vllm.config.vllm import EngineConfig  # VllmConfig alias
from my_vllm.v1.engine.llm_engine import LLMEngine
from my_vllm.v1.executor.uniproc_executor import UniProcExecutor


def run(prompts: list[str]):
    cfg = EngineConfig()
    tokenizer = load_tokenizer(cfg.model_config.model_name)

    engine = LLMEngine(
        vllm_config=cfg,
        executor_class=UniProcExecutor,
        tokenizer=tokenizer,
    )

    for prompt in prompts:
        result = engine.generate(prompt, max_new_tokens=50)
        print(f"Prompt: {prompt}")
        print(f"Output: {result['text']}")
        print(f"Finish: {result['finish_reason']}")
        print()


if __name__ == "__main__":
    run(["Hello, my name is"])