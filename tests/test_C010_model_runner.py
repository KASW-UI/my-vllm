from my_vllm.v1.worker.gpu_model_runner import GPUModelRunner
from my_vllm.cache.kv_cache import KVCache
from my_vllm.runtime.tokenizer import load_tokenizer
from my_vllm.runtime.model_loader import load_model


def test_model_runner_prefill_decode():
    model_name = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m"
    tokenizer = load_tokenizer(model_name)
    hf_model = load_model(model_name)
    device = next(hf_model.parameters()).device

    runner = GPUModelRunner(hf_model, device, eos_token_id=2)
    cache = KVCache()

    import torch
    tokens = tokenizer.encode("Hello world")
    input_ids = torch.tensor([tokens], device=device)

    # C017 legacy API: _prefill_one / _decode_one_legacy
    first = runner._prefill_one("test", input_ids, cache)
    assert first != 2
    assert runner.get_step("test") == 1

    for _ in range(5):
        token = runner._decode_one_legacy("test", cache)
        assert token >= 0
    assert runner.get_step("test") == 6

    runner.cleanup("test")
    assert runner.get_step("test") == 0
    print("ModelRunner prefill/decode/cleanup OK")
