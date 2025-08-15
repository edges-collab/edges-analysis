"""Various utility functions."""

import functools
import logging
import operator
from collections.abc import Sequence
from hashlib import md5
from itertools import product

import numpy as np
from numpy import typing as npt
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

logger = logging.getLogger(__name__)


def stable_hash(x) -> str:
    """A simple hash function to string."""
    return md5(str(x).encode()).hexdigest()


def linear_to_decibels(x: npt.NDArray) -> npt.NDArray[float]:
    """Convert a linear number to decibels."""
    return 20 * np.log10(np.abs(x))


def decibels_to_linear(x: npt.NDArray) -> npt.NDArray[float]:
    """Convert a number in decibels to linear."""
    return 10 ** (x / 20)


def as_readonly(x: np.ndarray) -> np.ndarray:
    """Get a read-only view into an array without copying."""
    result = x.view()
    result.flags.writeable = False
    return result


def dct_of_list_to_list_of_dct(dct: dict[str, Sequence]) -> list[dict]:
    """Take a dict of key: list pairs and turn it into a list of all combos of dicts.

    Parameters
    ----------
    dct
        A dictionary for which each value is an iterable.

    Returns
    -------
    list
        A list of dictionaries, each having the same keys as the input ``dct``, but
        in which the values are the elements of the original iterables.

    Examples
    --------
    >>> dct_of_list_to_list_of_dct(
    >>>    { 'a': [1, 2], 'b': [3, 4]}
    [
        {"a": 1, "b": 3},
        {"a": 1, "b": 4},
        {"a": 2, "b": 3},
        {"a": 2, "b": 4},
    ]
    """
    lists = dct.values()

    prod = product(*lists)

    return [dict(zip(dct.keys(), p, strict=False)) for p in prod]


class ComplexSpline:
    """Return a complex spline object."""

    def __init__(self, x, y, **kwargs):
        self.real = Spline(x, y.real, **kwargs)
        self.imag = Spline(x, y.imag, **kwargs)

    def __call__(self, x):
        """Compute the interpolation at x."""
        return self.real(x) + 1j * self.imag(x)


def join_struct_arrays(arrays):
    """Join a list of structured numpy arrays (make new columns)."""
    dtype = functools.reduce(operator.iadd, (a.dtype.descr for a in arrays), [])
    out = np.empty(len(arrays[0]), dtype=dtype)

    for a in arrays:
        for name in a.dtype.names:
            out[name] = a[name]
    return out


def slice_along_axis(x: np.ndarray, idx: np.ndarray | slice, axis: int = -1):
    """Get a view of x at indices idx on a given axis."""
    from_end = False
    if axis < 0:  # choosing axis at the end
        from_end = True
        axis = -1 - axis
    explicit_inds_slice = axis * (slice(None),)
    if from_end:
        return x[Ellipsis, idx, *explicit_inds_slice]
    return x[*explicit_inds_slice, idx]


def _tuplify(x):
    if not hasattr(x, "__len__"):
        return (int(x), int(x), int(x))
    return tuple(int(xx) for xx in x)
