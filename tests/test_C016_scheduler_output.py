"""C016: tests for SchedulerOutput (v1/core/sched/output.py)."""

from my_vllm.v1.core.sched.output import (
    NewRequestData,
    CachedRequestData,
    SchedulerOutput,
)


class TestNewRequestData:
    def test_default(self):
        d = NewRequestData(req_id="req-1")
        assert d.req_id == "req-1"
        assert d.prompt_token_ids == []
        assert d.block_ids == []
        assert d.num_computed_tokens == 0

    def test_full(self):
        d = NewRequestData(
            req_id="a",
            prompt_token_ids=[1, 2, 3],
            block_ids=[4, 5],
            num_computed_tokens=6,
        )
        assert d.req_id == "a"
        assert d.prompt_token_ids == [1, 2, 3]
        assert d.block_ids == [4, 5]
        assert d.num_computed_tokens == 6


class TestCachedRequestData:
    def test_default(self):
        d = CachedRequestData()
        assert d.req_ids == []
        assert d.num_reqs == 0

    def test_make_empty(self):
        d = CachedRequestData.make_empty()
        assert d.req_ids == []
        assert d.num_reqs == 0
        assert d.new_block_ids == []
        assert d.num_computed_tokens == []
        assert d.num_output_tokens == []

    def test_full(self):
        d = CachedRequestData(
            req_ids=["r1", "r2"],
            new_block_ids=[[10], None],
            num_computed_tokens=[5, 3],
            num_output_tokens=[2, 1],
        )
        assert d.req_ids == ["r1", "r2"]
        assert d.num_reqs == 2
        assert d.new_block_ids[0] == [10]
        assert d.new_block_ids[1] is None
        assert d.num_computed_tokens[1] == 3


class TestSchedulerOutput:
    def test_default(self):
        o = SchedulerOutput()
        assert o.scheduled_new_reqs == []
        assert o.total_num_scheduled_tokens == 0

    def test_make_empty(self):
        o = SchedulerOutput.make_empty()
        assert o.scheduled_new_reqs == []
        assert o.scheduled_cached_reqs.req_ids == []
        assert o.num_scheduled_tokens == {}
        assert o.total_num_scheduled_tokens == 0
        assert o.finished_req_ids == set()

    def test_full(self):
        nd = NewRequestData(req_id="n1", prompt_token_ids=[1])
        cd = CachedRequestData(
            req_ids=["r1"],
            num_computed_tokens=[0],
            num_output_tokens=[0],
        )
        o = SchedulerOutput(
            scheduled_new_reqs=[nd],
            scheduled_cached_reqs=cd,
            num_scheduled_tokens={"n1": 1, "r1": 1},
            total_num_scheduled_tokens=2,
            finished_req_ids={"done"},
        )
        assert len(o.scheduled_new_reqs) == 1
        assert o.scheduled_new_reqs[0].req_id == "n1"
        assert o.total_num_scheduled_tokens == 2
        assert "done" in o.finished_req_ids
