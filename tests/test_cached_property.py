"""Tests of the cached_property module."""

import pytest

from edges import cached_property as cp


class Derp:
    @cp.safe_property
    def lard(self):
        _ = self.non_existent
        print("did it")
        return 1

    @cp.cached_property
    def cheese(self):
        _ = self.macaroni
        print("did it")
        return 2

    def __getattr__(self, item):
        if item == "nugget":
            return "spam"

        raise AttributeError


def test_safe_property():
    d = Derp()
    with pytest.raises(RuntimeError, match="Wrapped AttributeError"):
        _ = d.lard


def test_cached_property():
    d = Derp()
    with pytest.raises(RuntimeError, match="failed with an AttributeError"):
        _ = d.cheese
