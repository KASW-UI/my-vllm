"""C015: tests for ConstantList (v1/utils.py) — read-only list wrapper."""

import pytest

from my_vllm.v1.utils import ConstantList


class TestConstantList:
    def test_len_and_iter(self):
        cl = ConstantList([1, 2, 3])
        assert len(cl) == 3
        assert list(cl) == [1, 2, 3]

    def test_getitem(self):
        cl = ConstantList([10, 20, 30])
        assert cl[0] == 10
        assert cl[2] == 30
        assert cl[1:3] == [20, 30]
        assert cl[-1] == 30

    def test_contains(self):
        cl = ConstantList(["a", "b", "c"])
        assert "a" in cl
        assert "x" not in cl

    def test_index(self):
        cl = ConstantList([100, 200, 300, 200])
        assert cl.index(200) == 1
        assert cl.index(200, 2) == 3

    def test_eq(self):
        a = ConstantList([1, 2])
        b = ConstantList([1, 2])
        assert a == b
        assert a == [1, 2]

    def test_append_raises(self):
        cl = ConstantList([1])
        with pytest.raises(TypeError):
            cl.append(2)

    def test_extend_raises(self):
        cl = ConstantList([1])
        with pytest.raises(TypeError):
            cl.extend([2])

    def test_insert_raises(self):
        cl = ConstantList([1])
        with pytest.raises(TypeError):
            cl.insert(3)

    def test_setitem_raises(self):
        cl = ConstantList([1, 2])
        with pytest.raises(TypeError):
            cl[0] = 99

    def test_delitem_raises(self):
        cl = ConstantList([1, 2])
        with pytest.raises(TypeError):
            del cl[0]

    def test_pop_raises(self):
        cl = ConstantList([1])
        with pytest.raises(TypeError):
            cl.pop(1)

    def test_remove_raises(self):
        cl = ConstantList([1])
        with pytest.raises(TypeError):
            cl.remove(1)

    def test_clear_raises(self):
        cl = ConstantList([1])
        with pytest.raises(TypeError):
            cl.clear()

    def test_repr(self):
        cl = ConstantList([5, 6])
        assert repr(cl) == "ConstantList([5, 6])"

    def test_empty(self):
        cl = ConstantList([])
        assert len(cl) == 0
        assert list(cl) == []
