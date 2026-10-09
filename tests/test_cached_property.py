"""Tests of the cached_property module."""

import attrs
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


class Counter:
    def __init__(self):
        self.ncalls = 0

    @cp.cached_property
    def value(self):
        self.ncalls += 1
        return 42


def test_cached_property_caches():
    c = Counter()
    assert c.value == 42
    assert c.value == 42
    assert c.ncalls == 1
    # Only the cached value is stored on the instance (no extra bookkeeping).
    assert set(c.__dict__) == {"ncalls", "value"}


def test_cached_property_on_slotted_attrs_class():
    # attrs replaces cached properties on slotted classes with its own, so the
    # value is still cached, but the AttributeError is not converted (as
    # documented in the class docstring).
    @attrs.define
    class Slotted:
        ncalls: int = 0

        @cp.cached_property
        def value(self):
            self.ncalls += 1
            return 42

        @cp.cached_property
        def broken(self):
            return self.nonexistent

    s = Slotted()
    assert s.value == 42
    assert s.value == 42
    assert s.ncalls == 1
    with pytest.raises(AttributeError):
        _ = s.broken
