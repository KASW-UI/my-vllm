from my_vllm.v1.request import Request, RequestStatus
from my_vllm.v1.core.sched.scheduler import Scheduler


def test_scheduler_fcfs():
    sched = Scheduler(max_running=2)

    req_a = Request(request_id="A")
    req_b = Request(request_id="B")
    req_c = Request(request_id="C")

    sched.add_request(req_a)
    sched.add_request(req_b)
    sched.add_request(req_c)

    assert len(sched.waiting) == 3
    assert len(sched.running) == 0
    assert req_a.status == RequestStatus.WAITING

    # Schedule — should promote 2 (max_running)
    new = sched.schedule()
    assert len(new.scheduled_new_reqs) == 2
    assert new.scheduled_new_reqs[0].req_id == "A"
    assert new.scheduled_new_reqs[1].req_id == "B"
    assert len(sched.waiting) == 1
    assert len(sched.running) == 2
    assert req_a.status == RequestStatus.RUNNING
    assert req_b.status == RequestStatus.RUNNING
    assert req_c.status == RequestStatus.WAITING

    # Finish A — frees a slot
    sched.finish(req_a)
    assert len(sched.running) == 1

    # Schedule again — C should be promoted
    new = sched.schedule()
    assert len(new.scheduled_new_reqs) == 1
    assert new.scheduled_new_reqs[0].req_id == "C"
    assert len(sched.waiting) == 0
    assert len(sched.running) == 2
    assert req_c.status == RequestStatus.RUNNING

    print("FCFS order: A → B → C (after A freed)")


def test_scheduler_has_work():
    sched = Scheduler(max_running=2)
    assert not sched.has_work()

    req = Request(request_id="X")
    sched.add_request(req)
    assert sched.has_work()

    sched.schedule()
    assert sched.has_work()  # still running

    sched.finish(req)
    assert not sched.has_work()
    print("has_work transitions correct")


def test_scheduler_full_e2e():
    """Full end-to-end: scheduler + state_manager work together."""
    import torch
    from my_vllm.runtime.tokenizer import load_tokenizer
    from my_vllm.model_executor.model_loader import load_model
    from my_vllm.v1.engine.core import EngineCore as StateManager
    from my_vllm.v1.worker.gpu_worker import Worker
    from my_vllm.v1.executor.uniproc_executor import UniProcExecutor

    model_name = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m"
    tokenizer = load_tokenizer(model_name)
    hf_model = load_model(model_name)

    worker = Worker(hf_model, hf_model.device)
    worker.init_device()
    executor = UniProcExecutor(worker)
    sm = StateManager(
        executor, hf_model.device,
        max_new_tokens=10, max_running=2
    )

    prompts = ["Hello world", "Machine learning is", "The quick brown"]
    for i, p in enumerate(prompts):
        req = Request(request_id=f"e2e-{i}")
        req.prompt_token_ids = tokenizer.encode(p)
        sm.add_request(req)

    while sm.has_work():
        sm.step()

    assert len(sm.finished) == 3
    for req in sm.finished:
        text = tokenizer.decode(req.output_token_ids)
        assert len(text) > 0
        assert req.is_finished()
        print(f"[{req.request_id}] {req.status.name} "
              f"({len(req.output_token_ids)} tokens): ...{text[-30:]!r}")
