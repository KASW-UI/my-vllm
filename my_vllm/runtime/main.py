from .tokenizer import load_tokenizer
from .model_loader import load_model
from .model import LLM
from my_vllm.engine.request import Request


def run_single(prompt: str, model_name: str, max_new_tokens: int):
    tokenizer = load_tokenizer(model_name)
    hf_model = load_model(model_name)
    llm = LLM(hf_model)

    request = Request(request_id="req-0")
    request.prompt_token_ids = tokenizer.encode(prompt)
    print(f"Request: {request}")

    llm.generate(request, max_new_tokens=max_new_tokens)

    text = tokenizer.decode(request.output_token_ids)
    print(f"Output: {prompt + text!r}")
    print(f"Status: {request.status.name} ({len(request.output_token_ids)} tokens)")
    print()


def run_batch(prompts: list[str], model_name: str, max_new_tokens: int):
    tokenizer = load_tokenizer(model_name)
    hf_model = load_model(model_name)
    llm = LLM(hf_model)

    requests = []
    for i, prompt in enumerate(prompts):
        req = Request(request_id=f"req-{i}")
        req.prompt_token_ids = tokenizer.encode(prompt)
        requests.append(req)

    print(f"Batch of {len(requests)} requests:")
    for req in requests:
        print(f"  {req}")
    print()

    llm.generate_batch(requests, max_new_tokens=max_new_tokens)

    for req in requests:
        text = tokenizer.decode(req.output_token_ids)
        prompt = tokenizer.decode(req.prompt_token_ids)
        print(f"[{req.request_id}] status={req.status.name} "
              f"tokens={len(req.output_token_ids)}")
        print(f"  output: {prompt + text!r}")
        print()


def run(
    prompt: str = "Hello, my name is",
    model_name: str = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m",
    max_new_tokens: int = 50,
    batch: bool = False,
):
    print(f"Loading model: {model_name}")
    if batch:
        prompts = [
            "The capital of France is",
            "Once upon a time",
            "In the year 2024",
        ]
        run_batch(prompts, model_name, max_new_tokens)
    else:
        run_single(prompt, model_name, max_new_tokens)


if __name__ == "__main__":
    run(batch=True)
