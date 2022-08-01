"""Various utility functions."""
from typing import Optional
from multiprocess import Pool, current_process, cpu_count
from multiprocessing.sharedctypes import RawArray

import numpy as np
from edges_cal import xrfi
import warnings
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

_globals = {}


def _init_worker(spectrum, weights, shape):
    # This just shoves things into _globals so that each worker in a pool hass access
    # to them. If they are in shared memory space (such as a RawArray), then they are
    # not copied to each process, just accessed therefrom.
    _globals["spectrum"] = spectrum
    _globals["weights"] = weights
    _globals["shape"] = shape


def join_struct_arrays(arrays):
    """Join a list of structured numpy arrays (make new columns)."""
    dtype = sum((a.dtype.descr for a in arrays), [])
    out = np.empty(len(arrays[0]), dtype=dtype)
    for a in arrays:
        for name in a.dtype.names:
            out[name] = a[name]
    return out


def run_xrfi(
    *,
    method: str,
    spectrum: np.ndarray,
    freq: np.ndarray,
    weights: Optional[np.ndarray] = None,
    flags: Optional[np.ndarray] = None,
    n_threads: int = cpu_count(),
    fl_id=None,
    **kwargs,
) -> np.ndarray:
    """Run an xrfi method on given spectrum and weights."""
    rfi = getattr(xrfi, f"xrfi_{method}")

    if weights is None:
        if flags is None:
            weights = np.ones_like(spectrum)
        else:
            weights = (~flags).astype(float)

    if flags is not None:
        weights = np.where(flags, 0, weights)

    if spectrum.ndim in rfi.ndim:
        flags = rfi(spectrum, weights=weights, **kwargs)[0]
    elif spectrum.ndim > max(rfi.ndim) + 1:
        # say we have a 3-dimensional spectrum but can only do 1D in the method.
        # then we collapse to 2D and recursively run xrfi_pipe. That will trigger
        # the *next* clause, which will do parallel mapping over the first axis.
        orig_shape = spectrum.shape
        new_shape = (-1,) + orig_shape[2:]
        flags = run_xrfi(
            spectrum=spectrum.reshape(new_shape),
            weights=weights.reshape(new_shape),
            freq=freq,
            method=method,
            n_threads=n_threads,
            **kwargs,
        )
        return flags.reshape(orig_shape)
    else:
        n_threads = min(n_threads, len(spectrum))

        # Use a parallel map unless this function itself is being called by a
        # parallel map.
        wrns = defaultdict(lambda: 0)

        def count_warnings(message, *args, **kwargs):
            wrns[str(message)] += 1

        old = warnings.showwarning
        warnings.showwarning = count_warnings

        if current_process().name == "MainProcess" and n_threads > 1:

            def fnc(i):
                # Gets the spectrum/weights from the global var dict, which was
                # initialized by the pool.
                # See https://research.wmz.ninja/articles/2018/03/on-sharing-large-
                # arrays-when-using-pythons-multiprocessing.html
                spec = np.frombuffer(_globals["spectrum"]).reshape(_globals["shape"])[i]
                wght = np.frombuffer(_globals["weights"]).reshape(_globals["shape"])[i]

                if np.any(wght > 0):
                    return rfi(spec, freq=freq, weights=wght, **kwargs)[0]
                else:
                    return np.ones_like(spec, dtype=bool)

            shared_spectrum = RawArray("d", spectrum.size)
            shared_weights = RawArray("d", spectrum.size)

            # Wrap X as an numpy array so we can easily manipulates its data.
            shared_spectrum_np = np.frombuffer(shared_spectrum).reshape(spectrum.shape)
            shared_weights_np = np.frombuffer(shared_weights).reshape(spectrum.shape)

            # Copy data to our shared array.
            np.copyto(shared_spectrum_np, spectrum)
            np.copyto(shared_weights_np, weights)

            p = Pool(
                n_threads,
                initializer=_init_worker,
                initargs=(shared_spectrum, shared_weights, spectrum.shape),
            )
            m = p.map
        else:

            def fnc(i):
                if np.any(weights[i] > 0):
                    return rfi(spectrum[i], freq=freq, weights=weights[i], **kwargs)[0]
                else:
                    return np.ones_like(spectrum[i], dtype=bool)

            m = map

        results = m(fnc, range(len(spectrum)))
        flags = np.array(list(results))

        warnings.showwarning = old

        # clear global memory (not sure if it still exists)
        _init_worker(0, 0, 0)

        fl_id = f"{fl_id}: " if fl_id else ""

        if wrns:
            for msg, count in wrns.items():
                msg = msg.replace("\n", " ")
                logger.warning(
                    f"{fl_id}Received warning '{msg}' {count}/{len(flags)} times."
                )

    return flags
