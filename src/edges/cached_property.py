"""A wrapper of cached_property that also saves which things are cached."""

from functools import cached_property as cp


class cached_property(cp):  # noqa
    def __get__(self, obj, cls):
        """Get the store value of the attribute."""
        try:
            value = super().__get__(obj, cls)
        except AttributeError as e:
            raise RuntimeError(
                f"{self.func.__name__} failed with an AttributeError"
            ) from e

        # Add the name of the decorated func to the _cached_ item of the dict
        if obj is not None:
            if "_cached_" not in obj.__dict__:
                obj.__dict__["_cached_"] = set()

            obj.__dict__["_cached_"].add(self.func.__name__)

        return value


def safe_property(f):
    """An alternative property decorator that substitates AttributeError for RunTime."""

    def getter(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except AttributeError as e:
            raise RuntimeError(f"Wrapped AttributeError in {f.__name__}") from e

    return property(getter)
