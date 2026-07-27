from my_vllm.engine.request import Request, RequestStatus
from my_vllm.cache.block import KVCacheBlock
from my_vllm.cache.block_pool import BlockPool
from my_vllm.cache.kv_cache_manager import KVCacheManager
from my_vllm.scheduler.scheduler import Scheduler


def test_block_pool_allocate_free():
    pool = BlockPool(num_blocks=10, block_size=16)
    assert pool.num_free == 10
    assert pool.num_used == 0

    blocks = pool.get_new_blocks(3)
    assert blocks is not None
    assert len(blocks) == 3
    assert pool.num_free == 7
    assert pool.num_used == 3
    assert all(not b.is_free for b in blocks)

    pool.free_blocks(blocks)
    assert pool.num_free == 10
    assert pool.num_used == 0
    assert all(b.is_free for b in blocks)

    # Too many blocks → None
    assert pool.get_new_blocks(11) is None
    print("BlockPool allocate/free OK")


def test_block_pool_fragmentation_resistant():
    """BlockPool can reuse freed blocks — no fragmentation."""
    pool = BlockPool(num_blocks=5, block_size=16)

    a = pool.get_new_blocks(3)  # blocks 0,1,2
    b = pool.get_new_blocks(2)  # blocks 3,4
    assert pool.num_free == 0

    pool.free_blocks(a)  # free 0,1,2
    assert pool.num_free == 3

    c = pool.get_new_blocks(3)  # reuses 0,1,2
    assert c is not None
    assert [blk.block_id for blk in c] == [0, 1, 2]
    print("BlockPool fragmentation OK")


def test_kv_cache_manager_blocks():
    pool = BlockPool(num_blocks=50, block_size=16)
    mgr = KVCacheManager(block_pool=pool)

    req = Request(request_id="kv-test")
    req.prompt_token_ids = list(range(32))  # 32 tokens → 2 blocks

    assert mgr.allocate(req)
    assert mgr.get_cache(req.request_id) is not None

    block_ids = mgr.get_block_ids(req.request_id)
    assert len(block_ids) == 2
    assert block_ids == [0, 1]

    assert mgr.num_free_blocks == 48

    mgr.free(req.request_id)
    assert mgr.get_cache(req.request_id) is None
    assert mgr.get_block_ids(req.request_id) == []
    assert pool.num_free == 50
    print("KVCacheManager blocks OK")


def test_scheduler_blocks_when_no_free_blocks():
    """Scheduler should not promote if BlockPool is full."""
    pool = BlockPool(num_blocks=5, block_size=16)
    mgr = KVCacheManager(block_pool=pool)
    sched = Scheduler(max_running=8, kv_cache_manager=mgr)

    # 3 requests, each needs 7+ blocks (100 tokens)
    for letter in "ABC":
        req = Request(request_id=letter)
        req.prompt_token_ids = list(range(100))
        sched.add_request(req)

    # First request: 100 tokens → ceil(100/16) = 7 blocks
    # But only 5 blocks total! So nothing should schedule.
    new = sched.schedule()
    assert len(new) == 0
    assert len(sched.waiting) == 3
    assert len(sched.running) == 0
    assert req.status == RequestStatus.WAITING
    print("Scheduler blocks when BlockPool full")


def test_scheduler_schedules_with_enough_blocks():
    pool = BlockPool(num_blocks=20, block_size=16)
    mgr = KVCacheManager(block_pool=pool)
    sched = Scheduler(max_running=2, kv_cache_manager=mgr)

    req_a = Request(request_id="A")
    req_a.prompt_token_ids = list(range(32))  # 2 blocks
    req_b = Request(request_id="B")
    req_b.prompt_token_ids = list(range(48))  # 3 blocks
    req_c = Request(request_id="C")
    req_c.prompt_token_ids = list(range(64))  # 4 blocks

    sched.add_request(req_a)
    sched.add_request(req_b)
    sched.add_request(req_c)

    new = sched.schedule()
    assert len(new) == 2  # max_running=2
    assert new[0].request_id == "A"
    assert new[1].request_id == "B"
    assert list(pool._free_ids) == [5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                    15, 16, 17, 18, 19]  # 15 free
    print("Scheduler schedules with enough blocks")


def test_kv_cache_manager_e2e():
    """Full pipeline with paged blocks."""
    import torch
    from my_vllm.runtime.tokenizer import load_tokenizer
    from my_vllm.runtime.model_loader import load_model
    from my_vllm.engine.state_manager import StateManager
    from my_vllm.worker.gpu_worker import Worker
    from my_vllm.executor.uniproc_executor import UniProcExecutor

    model_name = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m"
    tokenizer = load_tokenizer(model_name)
    hf_model = load_model(model_name)

    worker = Worker(hf_model, hf_model.device)
    worker.init_device()
    executor = UniProcExecutor(worker)
    sm = StateManager(
        executor, hf_model.device,
        max_new_tokens=10, max_running=2,
        num_kv_blocks=256, block_size=16,
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
        assert sm.kv_cache_manager.get_cache(req.request_id) is None
        print(f"[{req.request_id}] {req.status.name} "
              f"({len(req.output_token_ids)} tokens)")
