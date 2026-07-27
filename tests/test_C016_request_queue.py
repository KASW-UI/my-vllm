"""C016: tests for FCFSRequestQueue (v1/core/sched/request_queue.py)."""

from my_vllm.v1.core.sched.request_queue import FCFSRequestQueue
from my_vllm.v1.request import Request


class TestFCFSRequestQueue:
    def test_add_and_pop(self):
        q = FCFSRequestQueue()
        r1 = Request("a")
        r2 = Request("b")
        q.add_request(r1)
        q.add_request(r2)
        assert len(q) == 2
        assert bool(q)

        assert q.pop_request().request_id == "a"
        assert q.pop_request().request_id == "b"
        assert len(q) == 0
        assert not bool(q)

    def test_peek(self):
        q = FCFSRequestQueue()
        r1 = Request("x")
        q.add_request(r1)
        assert q.peek_request().request_id == "x"
        assert len(q) == 1  # peek doesn't remove

    def test_peek_empty_raises(self):
        import pytest
        q = FCFSRequestQueue()
        with pytest.raises(IndexError):
            q.peek_request()

    def test_prepend(self):
        q = FCFSRequestQueue()
        r1 = Request("first")
        r2 = Request("second")
        q.add_request(r1)
        q.prepend_request(r2)
        # r2 should be at front
        assert q.pop_request().request_id == "second"
        assert q.pop_request().request_id == "first"

    def test_remove(self):
        q = FCFSRequestQueue()
        r1 = Request("x")
        r2 = Request("y")
        r3 = Request("z")
        q.add_request(r1)
        q.add_request(r2)
        q.add_request(r3)

        q.remove_request(r2)
        assert len(q) == 2
        assert q.pop_request().request_id == "x"
        assert q.pop_request().request_id == "z"

    def test_iter(self):
        q = FCFSRequestQueue()
        r1 = Request("a")
        r2 = Request("b")
        q.add_request(r1)
        q.add_request(r2)
        # FCFS iter should preserve order
        ids = [r.request_id for r in q]
        assert ids == ["a", "b"]

    def test_isinstance_deque(self):
        from collections import deque
        q = FCFSRequestQueue()
        assert isinstance(q, deque)
