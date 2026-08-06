from .tokenizer import load_tokenizer
from my_vllm.v1.model_executor.model_loader import load_model
from my_vllm.v1.request import Request
from my_vllm.v1.engine.core import EngineCore as StateManager
from my_vllm.v1.worker.gpu_worker import Worker
from my_vllm.v1.executor.uniproc_executor import UniProcExecutor


def run(
    prompts: list[str] | None = None,
    model_name: str = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m",
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
    hf_model = load_model(model_name)

    worker = Worker(hf_model, hf_model.device)
    worker.init_device()
    executor = UniProcExecutor(worker)
    sm = StateManager(
        executor, hf_model.device, max_new_tokens=max_new_tokens
    )

    for i, prompt in enumerate(prompts):
        req = Request(request_id=f"req-{i}")
        req.prompt_token_ids = tokenizer.encode(prompt)
        sm.add_request(req)
        print(f"Added: {req}")

    print()

    while sm.has_work():
        finished = sm.step()
        for req in finished:
            text = tokenizer.decode(req.output_token_ids)
            prompt = tokenizer.decode(req.prompt_token_ids)
            print(f"[{req.request_id}] DONE ({req.status.name}, "
                  f"{len(req.output_token_ids)} tokens)")
            print(f"  {prompt + text!r}")
            print()


if __name__ == "__main__":
    run()
