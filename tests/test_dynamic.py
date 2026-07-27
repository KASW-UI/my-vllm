from my_vllm.runtime.tokenizer import load_tokenizer
from my_vllm.runtime.model_loader import load_model
from my_vllm.engine.request import Request, RequestStatus
from my_vllm.engine.state_manager import StateManager


def test_dynamic_lifecycle():
    prompts = [
        "The capital of France is",
        "Artificial intelligence",
        "In the beginning",
    ]
    model_name = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m"
    max_new_tokens = 15

    tokenizer = load_tokenizer(model_name)
    hf_model = load_model(model_name)

    sm = StateManager(
        hf_model, hf_model.device, max_new_tokens=max_new_tokens
    )

    # Add requests — all start WAITING
    for i, prompt in enumerate(prompts):
        req = Request(request_id=f"d-{i}")
        req.prompt_token_ids = tokenizer.encode(prompt)
        sm.add_request(req)
        assert req.status == RequestStatus.WAITING

    step_count = 0
    while sm.has_work():
        finished = sm.step()
        step_count += 1
        for req in finished:
            assert req.status.value >= 2  # FINISHED
            assert 1 <= len(req.output_token_ids) <= max_new_tokens

    assert step_count > 0
    assert len(sm.finished) == len(prompts)

    for req in sm.finished:
        text = tokenizer.decode(req.output_token_ids)
        assert len(text) > 0
        print(f"[{req.request_id}] {req.status.name} "
              f"({len(req.output_token_ids)} tokens)")


def test_partial_join():
    """New requests can join while others are running."""
    model_name = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m"
    tokenizer = load_tokenizer(model_name)
    hf_model = load_model(model_name)

    sm = StateManager(
        hf_model, hf_model.device, max_new_tokens=10, max_running=2
    )

    req_a = Request(request_id="A")
    req_a.prompt_token_ids = tokenizer.encode("Once upon a time")
    sm.add_request(req_a)

    # Run a few steps with only one request
    for _ in range(3):
        sm.step()
        assert len(sm.scheduler.running) == 1
    assert req_a.status == RequestStatus.RUNNING

    # Add a second request mid-execution
    req_b = Request(request_id="B")
    req_b.prompt_token_ids = tokenizer.encode("The future of AI")
    sm.add_request(req_b)

    assert req_b.status == RequestStatus.WAITING
    sm.step()
    # After one more step, B should be promoted to running
    assert req_b.status == RequestStatus.RUNNING
    assert len(sm.scheduler.running) <= 2

    # Run to completion
    while sm.has_work():
        sm.step()

    assert req_a.status.value >= 2
    assert req_b.status.value >= 2
    print(f"A: {tokenizer.decode(req_a.output_token_ids)!r}")
    print(f"B: {tokenizer.decode(req_b.output_token_ids)!r}")
