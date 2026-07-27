"""C015: tests for expanded Request (v1/request.py) — 11 statuses,
is_finished(), append_output_token_ids(), ConstantList views."""

from my_vllm.v1.request import Request, RequestStatus


class TestRequestStatus:
    def test_all_statuses_exist(self):
        names = {s.name for s in RequestStatus}
        expected = {
            "WAITING", "WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR",
            "WAITING_FOR_REMOTE_KVS", "WAITING_FOR_STREAMING_REQ",
            "RUNNING", "PREEMPTED",
            "FINISHED_STOPPED", "FINISHED_LENGTH_CAPPED",
            "FINISHED_ABORTED", "FINISHED_IGNORED",
            "FINISHED_ERROR", "FINISHED_REPETITION",
        }
        assert names == expected

    def test_is_finished_on_running(self):
        assert not RequestStatus.is_finished(RequestStatus.RUNNING)

    def test_is_finished_on_waiting(self):
        assert not RequestStatus.is_finished(RequestStatus.WAITING)

    def test_is_finished_on_stopped(self):
        assert RequestStatus.is_finished(RequestStatus.FINISHED_STOPPED)

    def test_is_finished_on_length_capped(self):
        assert RequestStatus.is_finished(RequestStatus.FINISHED_LENGTH_CAPPED)

    def test_is_finished_on_preempted(self):
        assert not RequestStatus.is_finished(RequestStatus.PREEMPTED)


class TestRequest:
    def test_default_init(self):
        r = Request("r1")
        assert r.request_id == "r1"
        assert r.status == RequestStatus.WAITING
        assert r.prompt_token_ids == []
        assert r.num_output_tokens == 0
        assert r.num_tokens == 0
        assert r.num_computed_tokens == 0
        assert r.num_preemptions == 0
        assert r.block_hashes == []
        assert r.max_tokens == 0  # no sampling_params

    def test_init_with_prompt(self):
        r = Request("r2", prompt_token_ids=[1, 2, 3])
        assert r.prompt_token_ids == [1, 2, 3]

    def test_arrival_time_auto(self):
        r = Request("r3")
        assert r.arrival_time > 0

    def test_is_finished_false_initially(self):
        r = Request("r4")
        assert not r.is_finished()

    def test_is_finished_true_after_finish(self):
        r = Request("r5")
        r.status = RequestStatus.FINISHED_STOPPED
        assert r.is_finished()

    def test_append_output_token_ids_single(self):
        r = Request("r6", prompt_token_ids=[5, 6])
        r.append_output_token_ids(100)
        r.append_output_token_ids(200)
        assert r.num_output_tokens == 2
        assert list(r.output_token_ids) == [100, 200]
        assert r.num_tokens == 4  # prompt + output

    def test_append_output_token_ids_list(self):
        r = Request("r7", prompt_token_ids=[1])
        r.append_output_token_ids([10, 20, 30])
        assert r.num_output_tokens == 3
        assert list(r.output_token_ids) == [10, 20, 30]
        assert r.num_tokens == 4

    def test_output_token_ids_is_readonly(self):
        r = Request("r8")
        r.append_output_token_ids(42)
        import pytest
        with pytest.raises(TypeError):
            r.output_token_ids.append(99)

    def test_all_token_ids_is_readonly(self):
        r = Request("r9", prompt_token_ids=[1, 2])
        r.append_output_token_ids(3)
        assert list(r.all_token_ids) == [1, 2, 3]
        import pytest
        with pytest.raises(TypeError):
            r.all_token_ids.append(4)

    def test_spec_token_ids(self):
        r = Request("r10")
        assert r.spec_token_ids == []
        assert r.num_tokens_with_spec == 0

    def test_is_prefill_chunk_default(self):
        r = Request("r11")
        assert not r.is_prefill_chunk

    def test_max_tokens_from_sampling_params(self):
        from my_vllm.v1.request import SamplingParams
        sp = SamplingParams(max_tokens=30)
        r = Request("r12", sampling_params=sp)
        assert r.max_tokens == 30
        assert r.sampling_params is sp
