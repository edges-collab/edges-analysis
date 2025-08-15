"""Tests of the tools module."""

import numpy as np
import pytest

from edges import tools


def test_dct_to_list():
    """Ensure simple dictionary is dealt with correctly."""
    dct_of_lists = {"a": [1, 2], "b": [3, 4]}

    list_of_dicts = tools.dct_of_list_to_list_of_dct(dct_of_lists)

    assert list_of_dicts == [
        {"a": 1, "b": 3},
        {"a": 1, "b": 4},
        {"a": 2, "b": 3},
        {"a": 2, "b": 4},
    ]


def test_tuplify():
    assert tools._tuplify((3, 4, 5, 3)) == (3, 4, 5, 3)
    assert tools._tuplify((3.0, 4.0)) == (3, 4)
    assert tools._tuplify(3) == (3, 3, 3) == tools._tuplify(3.0)

    with pytest.raises(ValueError):
        tools._tuplify("hey")


class TestJoinStructArrays:
    def test_join(self):
        ab = np.array([(1, 2), (3, 4), (5, 6)], dtype=[("a", float), ("b", float)])
        cd = np.array([(7, 8), (9, 10), (11, 12)], dtype=[("c", float), ("d", float)])

        new = tools.join_struct_arrays([ab, cd])
        assert "a" in new.dtype.names
        assert "b" in new.dtype.names
        assert "c" in new.dtype.names
        assert "d" in new.dtype.names
