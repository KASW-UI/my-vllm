"""C030: tests for chunked prefill scheduler."""

from my_vllm.v1.core.sched.scheduler import Scheduler
from my_vllm.v1.request import Request, RequestStatus


def _make_request(rid, tokens):
    req = Request(request_id=rid)
    if tokens:
        req.prompt_token_ids = tokens
    return req


class TestChunkedPrefillScheduler:
    def test_short_prompt_not_chunked(self):
        sched = Scheduler(max_running=4, max_num_batched_tokens=2048)
        req = _make_request("a", [1, 2, 3, 4, 5])
        sched.add_request(req)
        output = sched.schedule()
        assert req.is_prefill_chunk is False
        assert len(output.scheduled_new_reqs) == 1

    def test_long_prompt_chunked(self):
        """Prompt longer than budget should be chunked."""
        sched = Scheduler(max_running=4, max_num_batched_tokens=10)
        req = _make_request("a", list(range(25)))
        sched.add_request(req)

        # Step 1: first chunk (10 tokens)
        output = sched.schedule()
        assert req.is_prefill_chunk is True
        assert len(output.scheduled_new_reqs) == 1
        d = output.scheduled_new_reqs[0]
        assert d.num_computed_tokens == 0
        assert d.req_id == "a"

        # Step 2: second chunk (10 tokens)
        output = sched.schedule()
        assert req.is_prefill_chunk is True
        assert len(output.scheduled_new_reqs) == 1

        # Step 3: third chunk (5 tokens, completes)
        output = sched.schedule()
        assert req.is_prefill_chunk is False
        assert len(output.scheduled_new_reqs) == 1

    def test_chunk_progress_tracking(self):
        """num_computed_tokens should increase across chunks."""
        sched = Scheduler(max_running=4, max_num_batched_tokens=5)
        req = _make_request("a", list(range(12)))
        sched.add_request(req)

        sched.schedule()  # first 5
        assert req.num_computed_tokens == 5

        sched.schedule()  # next 5
        assert req.num_computed_tokens == 10

        sched.schedule()  # final 2
        assert req.num_computed_tokens == 12
        assert req.is_prefill_chunk is False

    def test_multiple_requests_chunked(self):
        """Multiple requests share token budget."""
        sched = Scheduler(max_running=4, max_num_batched_tokens=15)
        req_a = _make_request("a", list(range(10)))
        req_b = _make_request("b", list(range(10)))
        sched.add_request(req_a)
        sched.add_request(req_b)

        output = sched.schedule()
        # Both should fit: 10+10=20, budget=15, so first gets 10, second gets 5
        assert len(output.scheduled_new_reqs) == 2
        assert req_a.is_prefill_chunk is False  # 10 <= budget leftover after a
        assert req_b.is_prefill_chunk is True   # only 5 of 10 fit

    def test_chunk_continue_with_running(self):
        """Running request with is_prefill_chunk should get next chunk."""
        sched = Scheduler(max_running=4, max_num_batched_tokens=5)
        req = _make_request("a", list(range(12)))
        sched.add_request(req)

        # First schedule: new request gets 5 tokens chunk
        sched.schedule()
        assert len(sched.running) == 1
        assert req.is_prefill_chunk is True

        # Second schedule: chunk continue
        sched.schedule()
        assert req.num_computed_tokens == 10

    def test_budget_exhausted_stops_scheduling(self):
        """When budget hits 0, no more new requests."""
        sched = Scheduler(max_running=4, max_num_batched_tokens=5)
        req_a = _make_request("a", list(range(10)))
        req_b = _make_request("b", list(range(10)))
        sched.add_request(req_a)
        sched.add_request(req_b)

        output = sched.schedule()
        # Only one request gets scheduled (budget=5 used for 5 of 10 tokens)
        assert len(output.scheduled_new_reqs) == 1
        assert req_a.status == RequestStatus.RUNNING
        assert req_b.status == RequestStatus.WAITING

    def test_finish_request(self):
        sched = Scheduler(max_running=4)
        req = _make_request("a", [1, 2, 3])
        sched.add_request(req)
        sched.schedule()
        assert len(sched.running) == 1
        sched.finish(req)
        assert len(sched.running) == 0

    def test_has_work(self):
        sched = Scheduler(max_running=4)
        assert not sched.has_work()
        req = _make_request("a", [1, 2])
        sched.add_request(req)
        assert sched.has_work()

    def test_is_prefill_chunk_initialization(self):
        req = _make_request("a", [1, 2, 3])
        assert req.is_prefill_chunk is False
        assert req.num_computed_tokens == 0
