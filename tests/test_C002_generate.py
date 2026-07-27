from my_vllm.runtime.tokenizer import load_tokenizer
from my_vllm.runtime.model_loader import load_model
from my_vllm.runtime.model import LLM
from my_vllm.engine.request import Request, RequestStatus


def test_generate():
    prompt = "Once upon a time"
    model_name = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m"
    max_new_tokens = 20

    tokenizer = load_tokenizer(model_name)
    hf_model = load_model(model_name)
    llm = LLM(hf_model)

    request = Request(request_id="test-0")
    request.prompt_token_ids = tokenizer.encode(prompt)
    assert request.status == RequestStatus.WAITING
    assert len(request.output_token_ids) == 0

    llm.generate(request, max_new_tokens=max_new_tokens)

    assert request.status in (
        RequestStatus.FINISHED_STOPPED,
        RequestStatus.FINISHED_LENGTH_CAPPED,
    )
    assert 1 <= len(request.output_token_ids) <= max_new_tokens

    generated_text = tokenizer.decode(request.output_token_ids)
    assert len(generated_text) > 0

    print(f"Prompt: {prompt!r}")
    print(f"Status: {request.status.name}")
    print(f"Generated ({len(request.output_token_ids)} tokens): {prompt + generated_text!r}")
