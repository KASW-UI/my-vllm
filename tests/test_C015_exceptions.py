"""C015: tests for Engine exceptions (v1/engine/exceptions.py)."""

from my_vllm.v1.engine.exceptions import EngineGenerateError, EngineDeadError


class TestExceptions:
    def test_engine_generate_error_is_recoverable(self):
        try:
            raise EngineGenerateError("test failure")
        except EngineGenerateError as e:
            assert str(e) == "test failure"

    def test_engine_dead_error_has_default_message(self):
        with TestExceptions._captured():
            try:
                raise EngineDeadError()
            except EngineDeadError as e:
                assert "EngineCore encountered an issue" in str(e)

    def test_engine_dead_error_suppress_context(self):
        err = EngineDeadError(suppress_context=True)
        assert err.__suppress_context__ is True

        err2 = EngineDeadError()
        assert err2.__suppress_context__ is False

    @staticmethod
    def _captured():
        import contextlib
        return contextlib.nullcontext()
