from transformers import AutoTokenizer

from my_vllm.config.vllm import EngineConfig
from my_vllm.v1.engine.llm_engine import LLMEngine
from my_vllm.v1.executor.uniproc_executor import UniProcExecutor


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class LLM:
    """An LLM for generating texts from given prompts and sampling parameters."""

    def __init__(
        self,
        model: str = "facebook/opt-125m",
        max_new_tokens: int = 50,
        executor_class=UniProcExecutor,
    ):
        self.model_name = model
        self.max_new_tokens = max_new_tokens

        cfg = EngineConfig()
        cfg.model_config.model_name = model

        self.tokenizer = load_tokenizer(model)
        self.engine = LLMEngine(
            vllm_config=cfg,
            executor_class=executor_class,
            tokenizer=self.tokenizer,
        )

    def generate(
        self,
        prompts: str | list[str],
        max_new_tokens: int | None = None,
    ) -> list[dict]:
        if isinstance(prompts, str):
            prompts = [prompts]
        max_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        results = []
        for prompt in prompts:
            result = self.engine.generate(prompt, max_new_tokens=max_tokens)
            results.append(result)
        return results

    def get_tokenizer(self):
        return self.tokenizer
