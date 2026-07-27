"""C020: tests for RequestState (v1/worker/gpu/states.py)."""

import torch

from my_vllm.v1.worker.gpu.states import RequestState


class TestRequestState:
    @staticmethod
    def _make_state(max_num_reqs=4, max_model_len=256):
        return RequestState(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            device=torch.device("cpu"),
        )

    def test_init(self):
        rs = self._make_state(max_num_reqs=8)
        assert rs.max_num_reqs == 8
        assert rs.num_reqs == 0
        assert len(rs.free_indices) == 8
        assert rs.all_token_ids.shape == (8, 256)
        assert rs.last_sampled_tokens.shape == (8, 1)

    def test_add_request(self):
        rs = self._make_state(max_num_reqs=4)
        idx = rs.add_request("r1", [1, 2, 3])
        assert rs.num_reqs == 1
        assert rs.req_id_to_index["r1"] == idx
        assert rs.index_to_req_id[idx] == "r1"
        assert rs.num_computed_tokens_cpu[idx] == 3
        assert rs.all_token_ids[idx, 0].item() == 1
        assert rs.all_token_ids[idx, 2].item() == 3

    def test_add_multiple_requests(self):
        rs = self._make_state(max_num_reqs=4)
        a = rs.add_request("a", [10, 20])
        b = rs.add_request("b", [30])
        assert rs.num_reqs == 2
        assert rs.get_index("a") == a
        assert rs.get_index("b") == b
        assert a != b

    def test_append_output_token(self):
        rs = self._make_state(max_num_reqs=4)
        idx = rs.add_request("r1", [1, 2])  # 2 prompt tokens → count=2
        rs.append_output_token("r1", 100)
        assert rs.all_token_ids[idx, 2].item() == 100
        assert rs.num_computed_tokens_cpu[idx] == 3

    def test_last_sampled_token(self):
        rs = self._make_state()
        rs.add_request("r1", [5])
        rs.set_last_sampled_token("r1", 42)
        assert rs.get_last_token("r1") == 42

    def test_get_step_with_append(self):
        rs = self._make_state()
        rs.add_request("r1", [1, 2, 3])
        assert rs.get_step("r1") == 3
        rs.append_output_token("r1", 100)
        assert rs.get_step("r1") == 4

    def test_get_step_nonexistent(self):
        rs = self._make_state()
        assert rs.get_step("nope") == 0

    def test_get_last_token_nonexistent(self):
        rs = self._make_state()
        assert rs.get_last_token("nope") == 0

    def test_remove_request(self):
        rs = self._make_state(max_num_reqs=4)
        rs.add_request("r1", [1])
        idx = rs.get_index("r1")
        freed = rs.remove_request("r1")
        assert freed == idx
        assert rs.num_reqs == 0
        assert "r1" not in rs.req_id_to_index
        assert idx in rs.free_indices
        assert rs.last_sampled_tokens[idx, 0].item() == 0

    def test_remove_nonexistent(self):
        rs = self._make_state()
        assert rs.remove_request("ghost") is None

    def test_index_reuse(self):
        rs = self._make_state(max_num_reqs=4)
        idx1 = rs.add_request("a", [1])
        rs.remove_request("a")
        idx2 = rs.add_request("b", [2])
        assert idx1 == idx2  # slot reused

    def test_all_token_ids_persistence(self):
        rs = self._make_state(max_num_reqs=4, max_model_len=16)
        idx = rs.add_request("r1", [5, 10, 15])
        rs.append_output_token("r1", 20)
        rs.append_output_token("r1", 25)
        assert [rs.all_token_ids[idx, i].item() for i in range(5)] == [
            5, 10, 15, 20, 25
        ]
