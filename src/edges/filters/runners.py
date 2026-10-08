"""A module that provides high-level functions for filtering using multiple threads."""

import logging
import warnings
from collections import defaultdict
from multiprocessing import Pool, cpu_count, current_process
from multiprocessing.sharedctypes import RawArray

import numpy as np

from . import xrfi

logger = logging.getLogger(__name__)

_globals = {}


def _init_worker(spectrum, weights, shape, method=None, freqs=None, kwargs=None):
    # This just shoves things into _globals so that each worker in a pool hass access
    # to them. If they are in shared memory space (such as a RawArray), then they are
    # not copied to each process, just accessed therefrom.
    _globals["spectrum"] = spectrum
    _globals["weights"] = weights
    _globals["shape"] = shape
    _globals["method"] = method
    _globals["freqs"] = freqs
    _globals["kwargs"] = kwargs


def _get_flags(out) -> np.ndarray:
    """Extract the flags from the output of an xRFI function.

    Most xRFI functions return a tuple of ``(flags, info)``, but some (e.g.
    :func:`~edges.filters.xrfi.xrfi_explicit`) return just the flags.
    """
    return out[0] if isinstance(out, tuple) else out


def _run_single(
    method: str,
    spectrum: np.ndarray,
    weights: np.ndarray,
    freqs: np.ndarray,
    kwargs: dict,
) -> tuple[np.ndarray, list[str]]:
    """Run an xRFI method on a single spectrum, recording any warnings raised.

    Returns the flags and the list of warning messages emitted (every warning is
    recorded, not just the first from each code location).
    """
    if not np.any(weights > 0):
        return np.ones_like(spectrum, dtype=bool), []

    rfi = getattr(xrfi, f"xrfi_{method}")
    with warnings.catch_warnings(record=True) as wrns:
        warnings.simplefilter("always")
        flags = _get_flags(rfi(spectrum, freqs=freqs, weights=weights, **kwargs))

    return flags, [str(w.message) for w in wrns]


def _pool_worker(i: int) -> tuple[np.ndarray, list[str]]:
    # Gets the spectrum/weights from the global var dict, which was
    # initialized by the pool.
    # See https://research.wmz.ninja/articles/2018/03/on-sharing-large-
    # arrays-when-using-pythons-multiprocessing.html
    shape = _globals["shape"]
    spec = np.frombuffer(_globals["spectrum"]).reshape(shape)[i]
    wght = np.frombuffer(_globals["weights"]).reshape(shape)[i]
    return _run_single(
        _globals["method"], spec, wght, _globals["freqs"], _globals["kwargs"]
    )


def run_xrfi(
    *,
    method: str,
    spectrum: np.ndarray,
    freqs: np.ndarray,
    weights: np.ndarray | None = None,
    flags: np.ndarray | None = None,
    n_threads: int = cpu_count(),
    fl_id=None,
    **kwargs,
) -> np.ndarray:
    """Run an xrfi method on given spectrum and weights.

    Parameters
    ----------
    method
        The name of the xRFI method to use, i.e. the name of a function in
        :mod:`edges.filters.xrfi` without the ``xrfi_`` prefix (e.g. ``"iterative"``).
    spectrum
        The spectrum to flag. The last axis must be frequency. If the method does not
        support the dimensionality of the spectrum, it is run on each 1D/2D sub-array
        separately (possibly in parallel).
    freqs
        The frequencies of the spectrum, in MHz.
    weights
        The weights of the spectrum. Zero weights are treated as flagged.
    flags
        Existing flags of the spectrum (combined with the weights).
    n_threads
        The number of processes to use when running the method over sub-arrays of the
        spectrum. If greater than one, a multiprocessing pool is used.
    fl_id
        An identifier used to prefix the summary of warnings in the log.
    **kwargs
        Passed through to the xRFI method.

    Returns
    -------
    flags
        Boolean flags with the same shape as ``spectrum``.
    """
    rfi = getattr(xrfi, f"xrfi_{method}")

    if weights is None:
        weights = np.ones_like(spectrum) if flags is None else (~flags).astype(float)

    if flags is not None:
        weights = np.where(flags, 0, weights)

    if spectrum.ndim in rfi.ndim:
        flags = _get_flags(rfi(spectrum, weights=weights, freqs=freqs, **kwargs))
    elif spectrum.ndim > max(rfi.ndim) + 1:
        # say we have a 3-dimensional spectrum but can only do 1D in the method.
        # then we collapse to 2D and recursively run xrfi_pipe. That will trigger
        # the *next* clause, which will do parallel mapping over the first axis.
        orig_shape = spectrum.shape
        new_shape = (-1, *orig_shape[-max(rfi.ndim) :])
        flags = run_xrfi(
            spectrum=spectrum.reshape(new_shape),
            weights=weights.reshape(new_shape),
            freqs=freqs,
            method=method,
            n_threads=n_threads,
            fl_id=fl_id,
            **kwargs,
        )
        return flags.reshape(orig_shape)
    else:
        n_threads = min(n_threads, len(spectrum))

        # Use a parallel map unless this function itself is being called by a
        # parallel map.
        if current_process().name == "MainProcess" and n_threads > 1:
            shared_spectrum = RawArray("d", spectrum.size)
            shared_weights = RawArray("d", spectrum.size)

            # Wrap X as an numpy array so we can easily manipulate its data.
            shared_spectrum_np = np.frombuffer(shared_spectrum).reshape(spectrum.shape)
            shared_weights_np = np.frombuffer(shared_weights).reshape(spectrum.shape)

            # Copy data to our shared array.
            np.copyto(shared_spectrum_np, spectrum)
            np.copyto(shared_weights_np, weights)

            with Pool(
                n_threads,
                initializer=_init_worker,
                initargs=(
                    shared_spectrum,
                    shared_weights,
                    spectrum.shape,
                    method,
                    freqs,
                    kwargs,
                ),
            ) as p:
                results = p.map(_pool_worker, range(len(spectrum)))
        else:
            results = [
                _run_single(method, spectrum[i], weights[i], freqs, kwargs)
                for i in range(len(spectrum))
            ]

        flags = np.array([r[0] for r in results])

        wrns = defaultdict(int)
        for _, msgs in results:
            for msg in msgs:
                wrns[msg] += 1

        fl_id = f"{fl_id}: " if fl_id else ""

        for msg, count in wrns.items():
            msg = msg.replace("\n", " ")
            logger.warning(
                f"{fl_id}Received warning '{msg}' {count}/{len(flags)} times."
            )

    return flags
