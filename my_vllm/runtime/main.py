from .tokenizer import load_tokenizer
from .model_loader import load_model
from .model import LLM
from my_vllm.engine.request import Request


def run(
    prompt: str,
    model_name: str = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m",
    max_new_tokens: int = 50,
):
    print(f"Loading model: {model_name}")
    tokenizer = load_tokenizer(model_name)

    print("Loading weights ...")
    hf_model = load_model(model_name)
    llm = LLM(hf_model)

    request = Request(request_id="req-0")
    request.prompt_token_ids = tokenizer.encode(prompt)
    print(f"Request: {request}")
    print()

    print("Generating ...")
    llm.generate(request, max_new_tokens=max_new_tokens)

    generated_text = tokenizer.decode(request.output_token_ids)
    full_text = prompt + generated_text
    print(f"Output: {full_text!r}")
    print(f"Status: {request.status.name}")
    print(f"Generated {len(request.output_token_ids)} tokens")


if __name__ == "__main__":
    run("Hello, my name is")
