from my_vllm.engine.request import Request, RequestStatus
from my_vllm.cache.allocator import KVCacheAllocator
from my_vllm.cache.kv_cache_manager import KVCacheManager
from my_vllm.scheduler.scheduler import Scheduler


def test_allocator_budget():
    alloc = KVCacheAllocator(max_total_tokens=1000)
    assert alloc.can_allocate(500)
    alloc.reserve("a", 500)
    assert alloc.used() == 500
    assert alloc.free() == 500
    assert alloc.can_allocate(500)
    assert not alloc.can_allocate(501)
    alloc.reserve("b", 300)
    assert alloc.used() == 800
    assert not alloc.can_allocate(300)
    alloc.release("a")
    assert alloc.used() == 300
    assert alloc.can_allocate(700)
    print("Allocator budget tracking OK")


def test_kv_cache_manager_lifecycle():
    alloc = KVCacheAllocator(max_total_tokens=2000)
    mgr = KVCacheManager(allocator=alloc)

    req = Request(request_id="kv-test")
    req.prompt_token_ids = list(range(10))

    assert mgr.can_allocate(10 + 50)

    ok = mgr.allocate(req)
    assert ok
    assert mgr.get_cache(req.request_id) is not None

    mgr.set_step(req.request_id, 5)
    assert mgr.get_step(req.request_id) == 5

    mgr.set_next_token(req.request_id, 42)
    assert mgr.get_next_token(req.request_id) == 42

    mgr.free(req.request_id)
    assert mgr.get_cache(req.request_id) is None
    assert mgr.get_step(req.request_id) == 0
    print("KVCacheManager lifecycle OK")


def test_scheduler_blocks_when_no_kv_memory():
    """Scheduler should not promote if KV cache is full."""
    alloc = KVCacheAllocator(max_total_tokens=200)
    mgr = KVCacheManager(allocator=alloc)
    sched = Scheduler(max_running=8, kv_cache_manager=mgr)

    req_a = Request(request_id="A")
    req_a.prompt_token_ids = list(range(100))
    req_b = Request(request_id="B")
    req_b.prompt_token_ids = list(range(100))
    req_c = Request(request_id="C")
    req_c.prompt_token_ids = list(range(100))

    sched.add_request(req_a)
    sched.add_request(req_b)
    sched.add_request(req_c)

    new = sched.schedule()
    assert len(new) == 1
    assert new[0].request_id == "A"
    assert len(sched.waiting) == 2
    assert len(sched.running) == 1

    # A uses ~150 tokens (100 prompt + 50), only ~50 left
    # B needs ~150, won't be scheduled
    assert req_b.status == RequestStatus.WAITING

    mgr.free("A")
    sched.finish(req_a)

    new2 = sched.schedule()
    assert len(new2) == 1
    assert new2[0].request_id == "B"
    print("Scheduler respects KV cache budget")


def test_kv_cache_manager_e2e():
    """Full pipeline: Scheduler + KVCacheManager + StateManager."""
    import torch
    from my_vllm.runtime.tokenizer import load_tokenizer
    from my_vllm.runtime.model_loader import load_model
    from my_vllm.engine.state_manager import StateManager

    model_name = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m"
    tokenizer = load_tokenizer(model_name)
    hf_model = load_model(model_name)

    sm = StateManager(
        hf_model,
        hf_model.device,
        max_new_tokens=10,
        max_running=2,
        max_kv_tokens=4096,
    )

    prompts = [
        "Artificial intelligence",
        "Machine learning",
        "Deep neural networks",
    ]
    for i, p in enumerate(prompts):
        req = Request(request_id=f"kv-e2e-{i}")
        req.prompt_token_ids = tokenizer.encode(p)
        sm.add_request(req)

    while sm.has_work():
        sm.step()

    assert len(sm.finished) == 3
    for req in sm.finished:
        text = tokenizer.decode(req.output_token_ids)
        assert len(text) > 0
        print(f"[{req.request_id}] {req.status.name} "
              f"({len(req.output_token_ids)} tokens)")
        # Verify KV cache was freed
        assert sm.kv_cache_manager.get_cache(req.request_id) is None
