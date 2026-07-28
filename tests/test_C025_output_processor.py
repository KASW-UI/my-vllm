"""C025: tests for OutputProcessor and IncrementalDetokenizer."""

from my_vllm.v1.engine.detokenizer import IncrementalDetokenizer
from my_vllm.v1.engine.output_processor import OutputProcessor, RequestOutput
from my_vllm.v1.request import Request


class DummyTokenizer:
    def decode(self, ids):
        return " ".join(f"<{i}>" for i in ids)

    def encode(self, text):
        return [1, 2, 3]


class TestIncrementalDetokenizer:
    def test_empty(self):
        d = IncrementalDetokenizer(DummyTokenizer())
        assert d.num_output_tokens() == 0
        assert d.full_text() == ""

    def test_update_delta(self):
        d = IncrementalDetokenizer(DummyTokenizer())
        delta1 = d.update([5, 6, 7])
        assert delta1 == "<5> <6> <7>"
        assert d.num_output_tokens() == 3

        delta2 = d.update([8, 9])
        assert delta2 == "<8> <9>"
        assert d.num_output_tokens() == 5
        assert d.full_text() == "<5> <6> <7> <8> <9>"

    def test_update_empty(self):
        d = IncrementalDetokenizer(DummyTokenizer())
        delta = d.update([])
        assert delta == ""

    def test_token_ids_property(self):
        d = IncrementalDetokenizer(DummyTokenizer())
        d.update([1, 2])
        assert d.token_ids == [1, 2]

    def test_none_tokenizer(self):
        d = IncrementalDetokenizer(None)
        d.update([1, 2, 3])
        assert d.full_text() == ""
        assert d.token_ids == [1, 2, 3]


class TestOutputProcessor:
    def test_request_output_finished(self):
        ro = RequestOutput(
            request_id="r1",
            prompt_text="Hello",
            output_text=" world",
            finished=True,
            finish_reason="FINISHED_STOPPED",
            token_ids=[5, 6],
        )
        assert ro.is_finished()
        assert ro.output_token_ids == [5, 6]

    def test_request_output_running(self):
        ro = RequestOutput(
            request_id="r2",
            prompt_text="Hi",
            output_text=" there",
            finished=False,
        )
        assert not ro.is_finished()

    def test_processor_with_tokenizer(self):
        from my_vllm.v1.outputs import ModelRunnerOutput

        tok = DummyTokenizer()
        op = OutputProcessor(tok)

        req = Request(request_id="r1", prompt_token_ids=[1, 2, 3])
        req.max_tokens = 10
        req_map = {"r1": req}

        mo = ModelRunnerOutput(
            req_ids=["r1"],
            sampled_token_ids={"r1": [5, 6]},
        )
        outputs = op.process_outputs(mo, req_map, eos_token_id=99, max_tokens=50)
        assert len(outputs) == 1
        out = outputs[0]
        assert out.request_id == "r1"
        assert out.output_tokens == 2
        assert out.finished is False
        assert len(out.output_token_ids) == 2

    def test_processor_eos_detection(self):
        from my_vllm.v1.outputs import ModelRunnerOutput

        op = OutputProcessor(None)
        req = Request(request_id="r1", prompt_token_ids=[1])
        req.max_tokens = 10
        req_map = {"r1": req}

        mo = ModelRunnerOutput(
            req_ids=["r1"],
            sampled_token_ids={"r1": [5, 99, 7]},  # 99 is EOS
        )
        outputs = op.process_outputs(mo, req_map, eos_token_id=99, max_tokens=50)
        assert outputs[0].finished is True
        assert outputs[0].finish_reason != ""

    def test_processor_max_tokens(self):
        from my_vllm.v1.outputs import ModelRunnerOutput

        op = OutputProcessor(None)
        req = Request(request_id="r1", prompt_token_ids=[1])
        req.max_tokens = 2
        req.append_output_token_ids(10)  # already has 1 token
        req_map = {"r1": req}

        mo = ModelRunnerOutput(
            req_ids=["r1"],
            sampled_token_ids={"r1": [20, 30]},
        )
        outputs = op.process_outputs(mo, req_map, eos_token_id=99, max_tokens=50)
        assert outputs[0].finished is True

    def test_cleanup(self):
        op = OutputProcessor(None)
        op._detokenizers["x"] = IncrementalDetokenizer(None)
        op.cleanup("x")
        assert "x" not in op._detokenizers
