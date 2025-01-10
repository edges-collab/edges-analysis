"""Test the tools module."""

import numpy as np

from edges_analysis import tools


class TestJoinStructArrays:
    def test_join(self):
        ab = np.array([(1, 2), (3, 4), (5, 6)], dtype=[("a", float), ("b", float)])
        cd = np.array([(7, 8), (9, 10), (11, 12)], dtype=[("c", float), ("d", float)])

        new = tools.join_struct_arrays([ab, cd])
        assert "a" in new.dtype.names
        assert "b" in new.dtype.names
        assert "c" in new.dtype.names
        assert "d" in new.dtype.names
