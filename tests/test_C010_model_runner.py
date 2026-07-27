from my_vllm.worker.model_runner import ModelRunner
from my_vllm.cache.kv_cache import KVCache
from my_vllm.runtime.tokenizer import load_tokenizer
from my_vllm.runtime.model_loader import load_model


def test_model_runner_prefill_decode():
    model_name = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m"
    tokenizer = load_tokenizer(model_name)
    hf_model = load_model(model_name)
    device = next(hf_model.parameters()).device

    runner = ModelRunner(hf_model, device, eos_token_id=2)
    cache = KVCache()

    import torch
    input_ids = torch.tensor(
        [tokenizer.encode("Hello world")], device=device
    )

    first = runner.prefill("test", input_ids, cache)
    assert first != 2
    assert runner.get_step("test") == 1

    for _ in range(5):
        token = runner.decode("test", cache)
        assert token >= 0
    assert runner.get_step("test") == 6

    runner.cleanup("test")
    assert runner.get_step("test") == 0
    print("ModelRunner prefill/decode/cleanup OK")
