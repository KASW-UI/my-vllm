"""Legacy CLI entry point — now delegates to LLMEngine pattern."""

from my_vllm.runtime.tokenizer import load_tokenizer
from my_vllm.config.vllm import EngineConfig
from my_vllm.v1.engine.llm_engine import LLMEngine
from my_vllm.v1.executor.uniproc_executor import UniProcExecutor


def run(
    prompts: list[str] | None = None,
    model_name: str = "facebook/opt-125m",
    max_new_tokens: int = 50,
):
    if prompts is None:
        prompts = [
            "The capital of France is",
            "Once upon a time",
            "In the year 2024",
        ]

    print(f"Loading model: {model_name}")
    tokenizer = load_tokenizer(model_name)

    cfg = EngineConfig()
    cfg.model_config.model_name = model_name

    engine = LLMEngine(
        vllm_config=cfg,
        executor_class=UniProcExecutor,
        tokenizer=tokenizer,
    )

    for prompt in prompts:
        result = engine.generate(prompt, max_new_tokens=max_new_tokens)
        print(f"[{result['request_id']}] {prompt!r} → {result['text']!r}")


if __name__ == "__main__":
    run()