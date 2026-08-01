"""CLI entry point — offline inference via V1 engine pipeline.

C032: uses V1 LLMEngine instead of legacy runtime.
"""

from my_vllm.runtime.tokenizer import load_tokenizer
from my_vllm.runtime.model_loader import load_model
from my_vllm.config import EngineConfig
from my_vllm.v1.worker.gpu_worker import Worker
from my_vllm.v1.executor.uniproc_executor import UniProcExecutor
from my_vllm.v1.engine.core import EngineCore
from my_vllm.v1.engine.llm_engine import LLMEngine
from my_vllm.v1.request import Request


def run(prompts: list[str]):
    cfg = EngineConfig()
    tokenizer = load_tokenizer(cfg.model_name)
    model = load_model(cfg.model_name, dtype=cfg.dtype)
    device = next(model.parameters()).device

    worker = Worker(model=model, device=device, eos_token_id=cfg.eos_token_id)
    executor = UniProcExecutor(worker)
    engine_core = EngineCore(
        executor=executor, device=device,
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
