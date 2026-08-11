"""CLI entry point — offline inference via V1 engine pipeline.

C032: model loading moved deep into Executor → Worker → GPUModelRunner
chain.  run.py no longer loads the model or creates Worker directly;
it assembles VllmConfig + LLMEngine.
"""

from my_vllm.entrypoints.llm import LLM


def run(prompts: list[str]):
    llm = LLM(model="facebook/opt-125m")
    for prompt in prompts:
        result = llm.generate(prompt, max_new_tokens=50)[0]
        print(f"Prompt: {prompt}")
        print(f"Output: {result['text']}")
        print(f"Finish: {result['finish_reason']}")
        print()


if __name__ == "__main__":
    run(["Hello, my name is"])