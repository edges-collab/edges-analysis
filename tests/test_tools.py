"""Tests of the tools module."""

from hashlib import md5

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


class TestStableHash:
    def test_large_arrays_differing_in_middle(self):
        a = np.zeros(5000)
        b = a.copy()
        b[2500] = 1.0
        assert tools.stable_hash(a) != tools.stable_hash(b)
        assert tools.stable_hash((a, 1)) != tools.stable_hash((b, 1))

    def test_arrays_differing_beyond_print_precision(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0 + 1e-12])
        assert tools.stable_hash(a) != tools.stable_hash(b)

    def test_deterministic(self):
        a = np.linspace(0, 1, 5000)
        assert tools.stable_hash((a, "x", 3)) == tools.stable_hash((a.copy(), "x", 3))

    def test_unchanged_for_simple_values(self):
        # Hashes of plain values (e.g. existing cache keys) are unchanged.
        x = (50.0, "abc", None, 3, (1, 2), np.array([1.0, 2.5]))
        assert tools.stable_hash(x) == md5(str(x).encode()).hexdigest()

    def test_print_options_restored(self):
        before = np.get_printoptions()
        tools.stable_hash(np.zeros(5000))
        assert np.get_printoptions() == before
