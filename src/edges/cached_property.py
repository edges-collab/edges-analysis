"""A wrapper of cached_property that does not let AttributeErrors escape."""

from functools import cached_property as cp


class cached_property(cp):  # ruff: ignore[invalid-class-name]
    """A :func:`functools.cached_property` that re-raises AttributeError.

    An ``AttributeError`` raised while computing the property is re-raised as a
    ``RuntimeError``. Otherwise, Python would take it to mean that the property
    itself does not exist, and fall back to ``__getattr__`` (if defined), hiding
    the real error.

    Notes
    -----
    Like :func:`functools.cached_property`, this needs an instance ``__dict__``.
    On a slotted class (e.g. one made with ``attrs.define``, which uses slots by
    default), ``attrs`` replaces every ``functools.cached_property`` (including
    this subclass) with its own slot-based implementation. The value is then still
    cached, but an ``AttributeError`` is no longer converted to a ``RuntimeError``.
    Use ``attrs.define(slots=False)`` to keep the conversion.
    """

    def __get__(self, obj, cls):
        """Get the stored value of the attribute, computing it if necessary."""
        try:
            return super().__get__(obj, cls)
        except AttributeError as e:
            raise RuntimeError(
                f"{self.func.__name__} failed with an AttributeError"
            ) from e


def safe_property(f):
    """An alternative property decorator that substitates AttributeError for RunTime."""

    def getter(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except AttributeError as e:
            raise RuntimeError(f"Wrapped AttributeError in {f.__name__}") from e

    return property(getter)
