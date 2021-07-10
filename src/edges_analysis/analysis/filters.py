from __future__ import annotations

import logging
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path
from typing import Tuple, List, Sequence, Dict, Union, Optional, Type, Callable
from .levels import FilteredData, CalibratedData, FrequencyRange
import dill as pickle

import h5py
import numpy as np
import yaml
from multiprocess.pool import Pool
from edges_cal.modelling import Model, Polynomial, _get_mad, flagged_filter, robust_divide
from edges_cal import FrequencyRange

from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


@dataclass
class RMSInfo:
    gha: np.ndarray
    model_params: dict
    meta: dict
    rms: dict
    flags: dict
    model_eval: dict
    abs_resids: dict
    model_stds: dict
    poly_params: dict
    poly_params_std: dict

    @classmethod
    def from_file(cls, fname: [str, Path]) -> RMSInfo:
        fname = Path(fname)

        def get_key(fl, key):
            return {
                model: {
                    (float(band[1:].split(",")[0]), float(band[:-1].split(",")[1])): v[...]
                    for band, v in model_dct.items()
                }
                for model, model_dct in fl[key].items()
            }

        with h5py.File(fname, "r") as fl:
            data = {
                key: get_key(fl, key)
                for key in [
                    "rms",
                    "flags",
                    "model_eval",
                    "abs_resids",
                    "model_stds",
                    "poly_params",
                    "poly_params_std",
                ]
            }

            model_params = {}
            for mdl_name, mdl_data in fl["model_params"].items():
                model_params[mdl_name] = dict(mdl_data.attrs)

            out = cls(
                gha=fl["gha"][...],
                meta={k: v for k, v in fl.attrs.items() if not k.startswith("model_")},
                model_params=model_params,
                **data,
            )
        return out

    def write(self, fname: [str, Path]):
        with h5py.File(fname, "w") as fl:
            fl["gha"] = self.gha

            for key in [
                "rms",
                "flags",
                "model_eval",
                "abs_resids",
                "model_stds",
                "poly_params",
                "poly_params_std",
            ]:
                grp = fl.create_group(key)

                for model in self.model_names:
                    mdl_grp = grp.create_group(model)

                    for band in self.bands[model]:
                        mdl_grp[str(band)] = getattr(self, key)[model][band]

            for key, val in self.meta.items():
                fl.attrs[key] = val

            grp = fl.create_group("model_params")
            for model in self.model_names:
                mdl_grp = grp.create_group(model)

                for k, v in self.model_params[model].items():
                    mdl_grp.attrs[k] = v

    @property
    def model_names(self):
        """The names of the models used to generate RMS"""
        return tuple(self.model_eval.keys())

    @property
    def bands(self) -> Dict[str, List[Tuple[float, float]]]:
        """The bands."""
        return {m: list(self.model_eval[m].keys()) for m in self.model_names}

    def plot(
        self,
        n_sigma: int = 3,
        data_gha: Optional[np.ndarray] = None,
        data_rms: Optional[np.ndarray] = None,
    ):
        isort = np.argsort(self.gha)
        gha = self.gha[isort]

        i = 0
        for name, data in self.rms.items():
            for band, rms in data.items():
                model = self.model_eval[name][band][isort]
                std = self.model_stds[name][band][isort]

                plt.plot(gha, rms[isort], label=f"{name}: {band}", color=f"C{i}")
                plt.plot(gha, model, ls="--", color=f"C{i}")
                plt.fill_between(
                    gha, model - n_sigma * std, model + n_sigma * std, color=f"C{i}", alpha=0.4
                )
                if data_gha is not None:
                    plt.scatter(data_gha, data_rms[name][band], color=f"C{i}")
                i += 1

        plt.legend()


def get_rms_info(
    steps: Sequence[Union[FilteredData, CalibratedData]],
    models: Dict[str, Dict],
    n_poly: int = 3,
    n_sigma: float = 3,
    n_terms: int = 16,
    n_std: int = 6,
    n_threads: int = cpu_count(),
) -> RMSInfo:
    """Computation of RMS info on a subset of files.

    This function computes the RMS, and polynomial models of the RMS over GHA. It
    also computes an estimated model of the standard deviation of the model.

    Parameters
    ----------
    steps
        A list of :class:`~FilteredData` or :class:`~CalibratedData` objects on which
        to compute the RMS statistics.
    models
        Dictionary of models to fit in order to obtain the RMS. Each should be a dictionary,
        with keys "model", "params", "resolution", and "bands". The "model" should be a
        string matching an edges-cal model, the "params" should be a dictionary of
        parameters to pass to this :class:`Model`. The "resolution" should be an int or
        float specifying the frequency bin size (in no. of channels or bandwidth in MHz).
        The "bands" should be a list of tuples of frequency ranges (in MHz), or the
        strings "full", "low" or "high" (corresponding to the full band, lower half,
        or upper half). These are used to compute the RMS (the model itself is fit over
        the full range every time).
    n_poly
        Number of polynomial terms with which to fit the the RMS as a function of GHA,
        purely for flagging bad RMS values.
    n_sigma
        Number of sigma at which to threshold individual RMS values from the model
        fit over GHA.
    n_terms
        The number of polynomial terms to fit to the final model of RMS vs GHA.
    n_std
        The number of polynomial terms to fit to the absolute residual of RMS
        as a function of GHA.
    n_threads
        The number of threads to use to compute the RMS.

    Returns
    -------
    rms_info
        A special data structure representing the RMS calculation.

    See Also
    --------
    rms_filter
        The output of this function is supposed to be used in ``rms_filter`` to actually
        filter files (those files may not be the same as went into this function).
    """
    # Make a big vector of all GHA's in all files.
    gha = np.hstack([l1.ancillary["gha"] for l1 in steps])

    n_threads = min(n_threads, len(steps))
    if n_threads > 1:
        pool = Pool(n_threads)
        m = pool.map
    else:
        m = map

    bands = _get_band_specs(models, steps[0].freq)

    out = {name: {b: {} for b in bands} for name, bands in bands.items()}
    params = {}
    for name, model in models.items():
        # Get the RMS values for each of the files, for each of the bands.
        mdl = model.get("model")
        prms = model.get("params", {})

        # Put all the RMS values for all files into one long vector.
        rms = list(
            m(
                lambda i: steps[i].get_model_rms(
                    freq_ranges=list(bands[name]),
                    model=mdl,
                    resolution=model.get("resolution", 0.0488),
                    weights=steps[i].weights,
                    **prms,
                ),
                list(range(len(steps))),
            )
        )
        params[name] = prms.copy()
        params[name].update(**{k: v for k, v in model.items() if k not in ["params", "bands"]})

        for band in bands[name]:
            this = out[name][band]
            this["rms"] = np.hstack([r[band] for r in rms])
            this.update(get_rms_model_over_time(this["rms"], gha, n_poly, n_sigma, n_terms, n_std))

    # Transform the out dict to have the measurements as top-level keys
    new_out = _bring_measurements_to_top(out)

    return RMSInfo(
        gha=gha,
        meta={"n_poly": n_poly, "n_sigma": n_sigma, "n_terms": n_terms, "n_std": n_std},
        model_params=params,
        **new_out,
    )


def _bring_measurements_to_top(out: Dict) -> Dict[str, Dict]:
    for name, value_ in out.items():
        for band in value_:
            keys = value_[band].keys()

    new_out = {}
    for key in keys:
        new_out[key] = {}
        for name, value in out.items():
            new_out[key][name] = {}
            for band in value:
                new_out[key][name][band] = out[name][band][key]

    return new_out


def _get_band_specs(
    models: Dict[str, Dict], freq: FrequencyRange
) -> Dict[str, List[Tuple[float, float]]]:
    bands = {name: [] for name in models}
    for name, model in models.items():
        bands_ = model.get("bands", ["full"])
        if not bands_:
            raise ValueError("'bands' must be a list of strings/tuples")

        for b in bands_:
            bands[name].append(_get_bands(b, freq))

    return bands


def _get_bands(b, freq) -> Tuple[float, float]:
    freq = FrequencyRange(freq)
    if b == "full" or b is None:
        return (np.floor(freq.min), np.ceil(freq.max))
    elif b == "low":
        return (np.floor(freq.min), np.floor(freq.center))
    elif b == "high":
        return (np.floor(freq.center), np.ceil(freq.max))
    elif isinstance(b, tuple) and len(b) == 2:
        return (float(b[0]), float(b[1]))
    else:
        raise ValueError(f"'bands' must be a list of strings/tuples, got {b}")


def get_rms_model_over_time(
    rms: np.ndarray, gha: np.ndarray, n_poly: int, n_sigma: float, n_terms: int, n_std: int
):
    flags = np.isnan(rms)

    for i in range(24):  # Go through each hour
        mask = (gha >= i) & (gha < (i + 1))
        extern_flg = flags[mask]
        this_flg = flags[mask].copy()

        n_flg = np.sum(this_flg) - 1

        while np.sum(this_flg) > n_flg and np.sum(~this_flg) > 2 * n_poly:
            n_flg = np.sum(this_flg)
            res, std = _get_polyfit_res_std(n_poly, gha[mask], rms[mask], flags=this_flg)
            this_flg = extern_flg | (np.abs(res) > n_sigma * std)

        extern_flg[:] = this_flg

    par = np.polyfit(gha[~flags], rms[~flags], n_terms - 1)
    model = np.polyval(par, gha)
    abs_res = np.abs(rms - model)
    par_std = np.polyfit(gha[~flags], abs_res[~flags], n_std - 1)
    model_std = np.polyval(par_std, gha)

    return {
        "flags": flags,
        "model_eval": model,
        "abs_resids": abs_res,
        "model_stds": model_std,
        "poly_params": par,
        "poly_params_std": par_std,
    }


def rms_filter(
    rms_info: [str, Path, RMSInfo],
    gha: np.ndarray,
    rms: Dict[str, Dict[Tuple[float, float], np.ndarray]],
    n_sigma: float = 3,
    fl_id=None,
):
    """Filter RMS data based on a summary set of data.

    Parameters
    ----------
    rms_info
        The output of ``get_rms_info``, or a file containing it.
    gha
        1D array of the GHA's of the current data.
    rms
        The RMS of the current data. Keys are different models (must match those in
        ``rms_info``, and values are dicts with keys that are the bands.
    n_sigma
        The threshold at which individual RMS values are flagged.

    Returns
    -------
    flags
        1D array of len(gha) specifying which integrations to flag.
    """
    if not isinstance(rms_info, RMSInfo):
        rms_info = RMSInfo.from_file(rms_info)

    flags = np.zeros(len(gha), dtype=bool)
    fl_id = f"{fl_id}: " if fl_id else ""

    for model, rms_model in rms.items():
        for band, rms_band in rms_model.items():
            pp = rms_info.poly_params[model][band]
            pstd = rms_info.poly_params_std[model][band]

            m = np.polyval(pp, gha)
            ms = np.polyval(pstd, gha)
            flags |= np.abs(rms_band - m) > n_sigma * ms

            logger.info(
                f"{fl_id}{np.sum(flags)}/{len(flags)} GHA's flagged after RMS filter for model={model}, band={band}."
            )

    return flags


def total_power_filter(
    gha: np.ndarray,
    spectra: np.ndarray,
    frequencies: np.ndarray,
    flags: [None, np.ndarray] = None,
    n_poly: int = 3,
    n_sigma: float = 3.0,
    bands: [None, List[Tuple[float, float]]] = None,
    std_thresholds=None,
    width: int = 100,
):
    """
    Filter on total power.

    The algorithm of the filter is:

    Get the sum of each spectrum (for a night/day) in three different bands (low half,
    upper half, total) (yielding, for each band, a "total power" as a function of GHA).
    For each band, split the total power into smallish chunks of GHA (nominally 1hr bins),
    and for each bin, fit a low-order polynomial to the total power.
    GHA's whose total power is outside some threshold are flagged, and the previous step
    is repeated, until no new GHAs are flagged. This threshold is itself thresholded:
    if the RMS of the residuals of the fit are very large, even 1-sigma outliers are
    flagged. Otherwise, only 3-4-sigma outliers are flagged.


    Parameters
    ----------
    gha : array_like
        1D array of galactic hour angles
    spectra
        2D array where first dimension is length GHA and corresponds to different
        integrations, and second dimension corresponds to frequency.
    frequencies
        The frequencies at which the spectra are defined.
    flags
        Any flagged entries in the spectra (2D array, shape (GHA, freq)).
    n_poly
        The number of polynomial terms to fit to each 1-hour GHA bin.
    n_sigma
        The number of sigma at which to threshold individual integrations.
    bands
        A list of tuples, each with two floats: min and max of a frequency band over
        which to take the total power. By default, use the whole band only.
    std_thresholds
        The absolute threshold for each band in terms of the standard deviation of
        residuals. If very high, then individual integrations will be flagged at
        one sigma deviation rather than ``n_sigma``. Must be the same length as ``bands``
        if given.

    """
    # Set the relevant frequency bands over which to take the total powers.
    if bands is None:
        bands = [None]

    for i, band in bands:
        bands[i] = _get_bands(band, frequencies)

    assert spectra.shape == (len(gha), len(frequencies)), "total_power has wrong shape"

    if flags is None:
        flags = np.zeros(spectra.shape, dtype=bool)

    flags_1d = np.prod(flags, axis=1).astype(bool)

    # Now sum over the frequency bands.
    total_power = np.zeros((len(bands), len(gha)))
    for i, band in enumerate(bands):
        freq_mask = (frequencies >= band[0]) & (frequencies < band[1])
        total_power[i] = np.nanmean(np.where((~flags) | freq_mask, spectra, np.nan), axis=1)

    if std_thresholds is None:
        std_thresholds = [np.inf] * len(bands)

    res_collect = {}
    std_collect = {}
    tp_collect = {}
    mdl_collect = {}
    init_flags = {}
    gha_collect = {}
    flg_collect = {}

    for j, (tpi, std_threshold, band) in enumerate(zip(total_power, std_thresholds, bands)):
        # First flag really bad stuff
        expected_mean = mean_power_model(gha, bands[j][0], bands[j][1])
        flags_1d |= (total_power[j] < expected_mean / 3) | (total_power[j] > expected_mean * 3)

        (
            flg_collect[band],
            res_collect[band],
            std_collect[band],
            mdl_collect[band],
            tp_collect[band],
            gha_collect[band],
            init_flags[band],
        ) = gha_based_poly_filter_in_chunks(
            tpi,
            gha,
            flags=flags_1d,
            std_threshold=std_threshold,
            gha_bin_size=gha_bin_size,
            n_poly=n_poly,
            std_estimator=std_estimator,
        )

        flags_1d |= flg_collect[-1]

    return (
        flags_1d,
        flg_collect,
        res_collect,
        std_collect,
        tp_collect,
        mdl_collect,
        init_flags,
        gha_collect,
        total_power,
    )


def gha_based_filter(
    *,
    spectra: np.ndarray,
    gha: np.ndarray,
    freq: np.ndarray,
    aggregator: Callable,
    bands: [None, List[Tuple[float, float]]] = None,
    flags: Optional[np.ndarray] = None,
    filter_fnc: Callable = model_filter_1d_chunks,
    **kwargs,
):
    """
    Filter data that has been aggregated over the frequency dimension.

    This essentially performs an aggregate over frequency for a given band.
    """
    if flags is not None:
        flags_1d = np.prod(flags, axis=1).astype(bool)
    else:
        flags_1d = np.zeros(len(gha), dtype=bool)

    assert spectra.shape == (len(gha), len(freq)), "spectra has wrong shape"

    # Set the relevant frequency bands over which to take the aggregates
    if bands is None:
        bands = [None]

    for i, band in enumerate(bands):
        bands[i] = _get_bands(band, freq)

    gha_data = aggregator(spectra, bands)

    out = filter_fnc(gha_data, gha, flags=flags_1d, **kwargs)
    flags_1d |= out[0]
    meta = out[1:]

    return flags_1d, meta


# @dataclass
# class MultiDayGHAFilter:

#     @classmethod
#     def from_data(cls, objs: Sequence[Union[FilteredData, CalibratedData]], n_threads: int=1):
#         # Make a big vector of all GHA's in all files.
#         gha = np.hstack([obj.gha for obj in objs])

#         n_threads = min(n_threads, len(objs))
#         if n_threads > 1:
#             pool = Pool(n_threads)
#             m = pool.map
#         else:
#             m = map

#         bands = _get_band_specs(models, objs[0].freq)

#         out = {name: {b: {} for b in bands} for name, bands in bands.items()}
#         params = {}
#         for name, model in models.items():
#             # Get the RMS values for each of the files, for each of the bands.
#             mdl = model.get("model")
#             prms = model.get("params", {})

#             # Put all the RMS values for all files into one long vector.
#             rms = list(
#                 m(
#                     lambda i: steps[i].get_model_rms(
#                         freq_ranges=list(bands[name]),
#                         model=mdl,
#                         resolution=model.get("resolution", 0.0488),
#                         weights=steps[i].weights,
#                         **prms,
#                     ),
#                     list(range(len(steps))),
#                 )
#             )
#             params[name] = prms.copy()
#             params[name].update(**{k: v for k, v in model.items() if k not in ["params", "bands"]})

#             for band in bands[name]:
#                 this = out[name][band]
#                 this["rms"] = np.hstack([r[band] for r in rms])
#                 this.update(get_rms_model_over_time(this["rms"], gha, n_poly, n_sigma, n_terms, n_std))

#         # Transform the out dict to have the measurements as top-level keys
#         new_out = _bring_measurements_to_top(out)

#         return RMSInfo(
#             gha=gha,
#             meta={"n_poly": n_poly, "n_sigma": n_sigma, "n_terms": n_terms, "n_std": n_std},
#             model_params=params,
#             **new_out,
#         )


def model_filter_1d_chunks(
    data, x, bin_size, flags=None, **kwargs
) -> Tuple[
    np.ndarray,
    List[float],
    List[List[np.ndarray]],
    List[List[np.ndarray]],
    List[List[np.ndarray]],
    List[List[np.ndarray]],
    List[np.ndarray],
    List[np.ndarray],
]:
    """
    Filter data non-parametrically in smaller chunks.

    Essentially, this just calls :func:`model_filter_1d` on small independent chunks
    of data. See that function for details.


    Parameters
    ----------
    data
        The data to be filtered. 1D.
    x
        The coordinates of the data.
    bin_size
        The size of each independent chunk to be filtered
        (in units of x)
    flags
        The initial flags to be applied.

    Other Parameters
    ----------------
    kwargs
        Passed through to :func:`model_filter_1d`.

    Returns
    -------
    flags
        The final flags
    bins
        The coordinates bins in which the flags were estimated.
    flg_list, res_list, std_list, mdl_lst
        The flags/residuals/standard deviation/model estimated for
        each chunk and each iteration of the flagger.
    data_list, x_list
        The data and coordinates in their bins.
    """
    if flags is None:
        flags = np.zeros(len(data), dtype=bool)

    res_collect = []
    std_collect = []
    data_collect = []
    mdl_collect = []
    x_collect = []
    flg_collect = []

    bins = []

    for i, xmin in enumerate(range(x.min(), x.max(), bin_size)):
        mask = (x >= xmin) & (x < xmin + bin_size)
        bins.append(xmin)

        out_flags, res, std, mdl, flg = model_filter_1d(data[mask], x[mask], flags=flags, **kwargs)
        if kwargs.get("return_models", True):
            res_collect.append(res)
            std_collect.append(std)
            mdl_collect.append(mdl)
            flg_collect.append(flg)
            data_collect.append(data[mask])
            x_collect.append(x[mask])

        flags[mask] = out_flags

    meta = {
        "flags": flg_collect,
        "residuals": res_collect,
        "stds": std_collect,
        "models": mdl_collect,
        "data": data_collect,
        "x": x_collect,
        "bins": bins,
    }

    # Need an extra modeling step here.
    return flags, meta


def model_filter_1d(
    data: np.ndarray,
    x: np.ndarray,
    flags: Optional[np.ndarray] = None,
    init_flags: Optional[np.ndarray] = None,
    n_sigma: int = 3,
    std_estimator: str = "running_mad",
    std_threshold: float = np.inf,
    model: [str, Type[Model]] = Polynomial,
    return_models: bool = True,
    median_width: int = 100,
    max_iter: int = 20,
    **model_kwargs,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    A non-parametric filter over a 1D dataset that uses a model fit to detrend the data.

    This algorithm is iterative. For each iteration, it fits a model to unflagged data,
    and then estimates the standard deviation of the residuals (as a function of the
    coordinates). Residuals greater than some sigma threshold are flagged, and the process
    is repeated (on each iteration, the flags of the previous iteration can be reversed
    either way). This continues until no new flags are identified.

    Parameters
    ----------
    data
        The data to be filtered.
    x
        The coordinates of the data.
    flags
        Optional initial flags to apply to the data. The final flags
        will always include these.
    init_flags
        Optional initial *temporary* flags to apply to the data. These can be modified
        after the first iteration.
    std_estimator
        An idenfifier for which kind of method to use to estimate the standard deviation
        of the residuals. Options are 'running_mad' for a running/sliding median absolute
        deviation, 'absres' for a model fit to the absolute residuals, 'std' for the
        standard deviation of the entire series, and 'mad' for the median absolute deviation
        of the entire series.
    std_threshold
        If the estimated std is above this threshold, flags will be applied at 1-sigma.
    model
        The kind of model to fit to the data.
    return_models
        Whether to return all the information about the fit (useful for diagnostics)
    median_width
        If using "running_mad" for the ``std_estimator``, the width of the window
        used to estimate the running MAD.
    max_iter
        The maximum number of iterations to use.

    Other Parameters
    ----------------
    model_kwargs
        Any parameters to be passed onto the :class:`Model`.

    Returns
    -------
    flags
        The final flags.
    res
        A list of model residuals, one for each flagging iteration.
        Only if ``return_models=True``.
    std
        A list of residual standard deviation estimates, one for each flagging iteration.
        Only if ``return_models=True``.
    mdl
        A list of models, one for each flagging iteration.
        Only if ``return_models=True``.
    flg
        A list of flag arrays, one for each flagging iteration.
        Only if ``return_models=True``.
    """
    if flags is None:
        flags = np.zeros(len(data), dtype=bool)

    intern_flags = flags | np.isnan(data) | init_flags

    ndata = len(x)

    if isinstance(model, str):
        model = Model.get_mdl(model)

    model = model(default_x=x, **model_kwargs)

    res_collect = []
    std_collect = []
    mdl_collect = []
    flg_collect = []

    nflags = -1
    while nflags < np.sum(intern_flags):
        nflags = np.sum(intern_flags)

        if (ndata - nflags) < model.n_terms:
            # If we've flagged too much, just get rid of everything here.
            intern_flags[:] = True
            break

        wght = (~intern_flags).astype(int)
        fit = model.fit(ydata=data, weights=wght)
        res = fit.residuals

        # Estimate the standard deviation of the residuals.
        if std_estimator == "medfilt":
            sig = np.sqrt(
                flagged_filter(
                    res ** 2, size=2 * (median_width // 2) + 1, kind="median", flags=intern_flags
                )
                / 0.456
            )
        elif std_estimator == "absres":
            sig = model.fit(ydata=np.abs(res), weights=wght).residuals
        elif std_estimator == "std":
            sig = np.std(res) * np.ones_like(x)
        elif std_estimator == "mad":
            sig = _get_mad(res) * np.ones_like(x)

        zscore = robust_divide(res, sig)

        mask = sig > std_threshold
        intern_flags[mask] = flags[mask] | (zscore[mask] > 1)
        intern_flags[~mask] = flags[~mask] | (zscore[~mask] > n_sigma)

        if return_models:
            res_collect.append(res)
            std_collect.append(sig)
            mdl_collect.append(model)
            flg_collect.append(intern_flags)

    return intern_flags, res_collect, std_collect, mdl_collect, flg_collect


def mean_power_model(gha, nu_min, nu_max, beta=-2.5):
    """A really rough model of expected mean power between two frequencies"""
    t75 = 1750 * np.cos(np.pi * gha / 12) + 3250  # approximate model based on haslam
    return (t75 / ((beta + 1) * 75.0 ** beta) * (nu_max ** (beta + 1) - nu_min ** (beta + 1))) / (
        nu_max - nu_min
    )


def explicit_filter(times, bad, ret_times=False):
    """
    Explicitly filter out certain times.

    Parameters
    ----------
    times : array-like
        The input times. This can be either a recarray, a list of tuples, a list
        of ints, or a 2D array of ints. The columns of the recarray (or the entries
        of the tuples) should correspond to `year`, 'day` and `hour`. The last two
        are not required, eg. 2-tuples will be interpreted as ``(year, hour)``, and a
        list of ints will be interpreted as just years.
    bad : str or array-like
        Like `times`, but specifying the bad entries. Need not have the same columns
        as `times`. If any bad exists within a given time frame, it will be considered
        bad. Likewise, if bad has higher scope than times, then it will also be bad.
        Eg.: ``times = [2018], bad=[(2018, 125)]``, times will be considered bad.
        Also, ``times=[(2018, 125)], bad=[2018]``, times will be considered bad.
        If a str, reads the bad times from a properly configured YAML file.
    ret_times : bool, optional
        If True, return the good times as well as the indices of such in original array.

    Returns
    -------
    keep :
        indices marking which times are not bad if inplace=False.
    times :
        Only if `ret_times=True`. An array of the times that are good.
    """
    if isinstance(bad, str):
        with open(bad, "r") as fl:
            bad = yaml.load(fl, Loader=yaml.FullLoader)["bad_days"]

    try:
        nt = len(times[0])
    except AttributeError:
        nt = 1

    try:
        nb = len(bad[0])
    except AttributeError:
        nb = 1

    assert nt in (1, 2, 3), "times must be an array of 1,2 or 3-tuples"
    assert nb in (1, 2, 3), "bad must be an array of 1,2 or 3-tuples"

    if nt < nb:
        bad = {b[:nt] for b in bad}
        nb = nt

    keep = [t[:nb] not in bad for t in times]

    if ret_times:
        return keep, times[keep]
    else:
        return keep


def time_filter_auxiliary(
    gha: np.ndarray,
    sun_el: np.ndarray,
    moon_el: np.ndarray,
    humidity: np.ndarray,
    receiver_temp: np.ndarray,
    gha_range: Tuple[float, float] = (0, 24),
    sun_el_max: float = 90,
    moon_el_max: float = 90,
    amb_hum_max: float = 200,
    min_receiver_temp: float = 0,
    max_receiver_temp: float = 100,
    flags: [None, np.ndarray] = None,
) -> np.ndarray:
    loc_flgs = np.zeros(len(gha), dtype=bool)

    def filter(condition, message, loc_flgs):
        nflags = np.sum(loc_flgs)

        loc_flgs |= condition
        nnew = np.sum(loc_flgs) - nflags
        if nnew:
            logger.debug(f"{nnew}/{len(loc_flgs) - nflags} times flagged due to {message}")

    filter((gha < gha_range[0]) | (gha >= gha_range[1]), "GHA range", loc_flgs)
    filter(sun_el > sun_el_max, "sun position", loc_flgs)
    filter(moon_el > moon_el_max, "moon position", loc_flgs)
    filter(humidity > amb_hum_max, "humidity", loc_flgs)
    filter(
        (receiver_temp >= max_receiver_temp) | (receiver_temp <= min_receiver_temp),
        "receiver temp",
        loc_flgs,
    )

    if flags is not None:
        assert flags.shape[-1] == len(
            gha
        ), f"flags must be an array with last axis len(gha). Got {flags.shape}"
        flags |= loc_flgs
    else:
        flags = loc_flgs

    return flags


def filter_explicit_gha(gha, first_day, last_day):
    # TODO: this should not be used -- it's arbitrary!
    if gha in [0, 2, 3, 4, 5, 16, 17, 18, 19, 20, 21, 22]:
        good_days = np.arange(140, 300, 1)
    else:
        good_day_dct = {
            1: np.concatenate((np.arange(148, 160), np.arange(161, 220))),
            10: np.concatenate(
                (
                    np.arange(148, 168),
                    np.arange(177, 194),
                    np.arange(197, 202),
                    np.arange(205, 216),
                )
            ),
            11: np.arange(187, 202),
            12: np.arange(147, 150),
            13: np.array([147, 149, 157, 159]),
            14: np.arange(148, 183),
            15: np.concatenate((np.arange(140, 183), np.arange(187, 206), np.arange(210, 300))),
            23: np.arange(148, 300),
            6: np.arange(147, 300),
            7: np.concatenate(
                (
                    np.arange(147, 153),
                    np.arange(160, 168),
                    np.arange(174, 202),
                    np.arange(210, 300),
                )
            ),
            8: np.concatenate((np.arange(147, 151), np.arange(160, 168), np.arange(174, 300))),
            9: np.concatenate(
                (
                    np.arange(147, 153),
                    np.arange(160, 168),
                    np.arange(174, 194),
                    np.arange(197, 202),
                    np.arange(210, 300),
                )
            ),
        }
        good_days = good_day_dct[gha]

    return good_days[(good_days >= first_day) & (good_days <= last_day)]
