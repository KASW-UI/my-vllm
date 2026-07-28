"""C026: tests for InputProcessor."""

from my_vllm.v1.engine.input_processor import InputProcessor
from my_vllm.v1.request import Request


class DummyTokenizer:
    def encode(self, text):
        if isinstance(text, list):
            return text
        return [ord(c) % 100 + 1 for c in text]


class TestInputProcessor:
    def test_process_string_prompt(self):
        tokenizer = DummyTokenizer()
        ip = InputProcessor(tokenizer, default_max_tokens=20)
        req = ip.process("hello")
        assert isinstance(req, Request)
        assert req.max_tokens == 20
        assert len(req.prompt_token_ids) == 5  # h,e,l,l,o → mapped ids
        assert req.request_id.startswith("req-")

    def test_process_token_id_list(self):
        tokenizer = DummyTokenizer()
        ip = InputProcessor(tokenizer)
        req = ip.process([1, 2, 3])
        assert req.prompt_token_ids == [1, 2, 3]

    def test_custom_max_tokens(self):
        tokenizer = DummyTokenizer()
        ip = InputProcessor(tokenizer, default_max_tokens=50)
        req = ip.process("test", max_new_tokens=10)
        assert req.max_tokens == 10

    def test_custom_request_id(self):
        tokenizer = DummyTokenizer()
        ip = InputProcessor(tokenizer)
        req = ip.process("x", request_id="my-id")
        assert req.request_id == "my-id"

    def test_auto_increment_ids(self):
        tokenizer = DummyTokenizer()
        ip = InputProcessor(tokenizer)
        r1 = ip.process("a")
        r2 = ip.process("b")
        assert r1.request_id == "req-0"
        assert r2.request_id == "req-1"
