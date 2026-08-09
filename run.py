"""CLI entry point — offline inference via V1 engine pipeline.

C032: model loading moved deep into Executor → Worker → GPUModelRunner
chain.  run.py no longer loads the model or creates Worker directly;
it just assembles the EngineConfig + EngineCore + LLMEngine.
"""

from my_vllm.runtime.tokenizer import load_tokenizer
from my_vllm.v1.config import EngineConfig
from my_vllm.v1.executor.uniproc_executor import UniProcExecutor
from my_vllm.v1.engine.core import EngineCore
from my_vllm.v1.engine.llm_engine import LLMEngine
from my_vllm.v1.request import Request


def run(prompts: list[str]):
    cfg = EngineConfig()
    tokenizer = load_tokenizer(cfg.model_name)

    executor = UniProcExecutor(
        model_name=cfg.model_name,
        dtype_str=cfg.dtype,
        eos_token_id=cfg.eos_token_id,
        block_size=cfg.block_size,
        max_num_blocks=cfg.num_kv_blocks,
    )
    executor._init_executor()

    engine_core = EngineCore(
        executor=executor,
        device=executor.worker._runner().device,
        max_new_tokens=cfg.max_new_tokens,
        eos_token_id=cfg.eos_token_id,
        max_running=cfg.max_running,
        num_kv_blocks=cfg.num_kv_blocks,
        block_size=cfg.block_size,
        tokenizer=tokenizer,
    )
    engine = LLMEngine(engine_core, tokenizer)

    for prompt in prompts:
        result = engine.generate(prompt, max_new_tokens=cfg.max_new_tokens)
        print(f"Prompt: {prompt}")
        print(f"Output: {result['text']}")
        print(f"Finish: {result['finish_reason']}")
        print()


if __name__ == "__main__":
    run(["Hello, my name is"])
