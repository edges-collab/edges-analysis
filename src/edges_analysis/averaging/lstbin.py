"""Functions for doing LST binning on GSData objects."""

from __future__ import annotations

import logging
import warnings

import edges_cal.modelling as mdl
import numpy as np
import pygsdata.coordinates as crd
from astropy import units as un
from astropy.coordinates import Longitude
from pygsdata import GSData, GSFlag, gsregister
from pygsdata.coordinates import lsts_to_times

from ..datamodel import add_model
from .averaging import bin_data

logger = logging.getLogger(__name__)


def get_lst_bins(
    binsize: float, first_edge: float = 0, max_edge: float = 24
) -> np.ndarray:
    """Get the LST bins.

    Parameters
    ----------
    binsize
        The size of the bins in hours.
    first_edge
        The first edge of the first bin.

    Returns
    -------
    np.ndarray
        The LST bin edges.
    """
    if binsize > 24:
        raise ValueError("Binsize must be <= 24 hours")
    if binsize <= 0:
        raise ValueError("Binsize must be greater than 0 hours")

    first_edge %= 24

    while max_edge <= (first_edge + 1e-4):
        max_edge += 24

    bins = np.arange(first_edge, max_edge, binsize)
    if np.isclose(bins.max() + binsize, max_edge):
        bins = np.append(bins, max_edge)

    return bins


@gsregister("reduce")
def lst_bin(
    data: GSData,
    binsize: float = 24.0,
    first_edge: float = 0.0,
    model: mdl.Model | None = None,
    max_edge: float = 24.0,
    in_gha: bool = False,
    use_model_residuals: bool | None = None,
):
    """LST-bin by using model information to de-bias the mean.

    Parameters
    ----------
    data
        The :class:`GSData` object to bin.
    binsize
        The size of the LST bins in hours.
    first_edge
        The first edge of the first bin in hours.
    model
        A :class:`edges_cal.modelling.Model` object to use for de-biasing the mean.
        If ``data`` already has a ``data_model`` defined, this argument is ignored.
    in_gha
        Whether to bin in GHA or LST (default).
    use_model_residuals
        Whether to use the model residuals to de-bias the mean in each bin. If
        True, either `model` must be provided, or `residuals` must be specified on the
        GSData object.

    Returns
    -------
    GSData
        A new :class:`GSData` object with the binned data.
    """
    if use_model_residuals is None:
        use_model_residuals = data.residuals is not None or model is not None

    if use_model_residuals:
        if data.residuals is None and model is None:
            raise ValueError("Cannot bin with models without a model!")

        if data.residuals is None:
            data = add_model(data, model=model)

    if in_gha:
        first_edge = crd.gha2lst(first_edge * un.hourangle)
        max_edge = crd.gha2lst(max_edge * un.hourangle)

    bins = get_lst_bins(binsize, first_edge, max_edge=max_edge)
    logger.debug(f"Got LST bins: {bins}")

    lsts = data.lsts.hour.copy()

    lsts %= 24
    lsts[lsts < bins[0]] += 24

    spec = np.zeros((data.nloads, data.npols, len(bins) - 1, data.nfreqs))
    resids = np.zeros_like(spec) if use_model_residuals else None
    nsmpls = np.zeros_like(spec)

    for iload in range(data.nloads):
        bbins = [
            (b[0] <= lsts[:, iload]) & (lsts[:, iload] < b[1])
            for b in zip(bins[:-1], bins[1:])
        ]

        spec[iload], nsmpls[iload], r = bin_data(
            data.data[iload],
            residuals=data.residuals[iload] if use_model_residuals else None,
            weights=data.flagged_nsamples[iload],
            bins=bbins,
            axis=-2,
        )

        if use_model_residuals:
            resids[iload] = r

    lst_ranges = np.tile(
        np.array(list(zip(bins[:-1], bins[1:])))[:, None], (1, data.nloads, 1)
    )
    lst_ranges = Longitude(lst_ranges * un.hour)

    lstbins = np.mean(lst_ranges, axis=-1)

    # Flag anything that is all nan -- these are just empty LSTs.
    flg = GSFlag(flags=np.all(np.isnan(spec), axis=(0, 1, 3)), axes=("time",))

    if data.auxiliary_measurements is not None:
        warnings.warn("Auxiliary measurements cannot being binned!", stacklevel=2)

    if data._effective_integration_time.size > 1:
        if np.unique(data._effective_integration_time).size > 1:
            warnings.warn(
                "lstbin does not yet support variable integration times!", stacklevel=2
            )
        intg_time = np.mean(data._effective_integration_time)
    else:
        intg_time = data._effective_integration_time

    data = data.update(
        data=spec,
        residuals=resids,
        nsamples=nsmpls,
        flags={"empty_lsts": flg},
        lsts=lstbins,
        lst_ranges=lst_ranges,
        times=lsts_to_times(
            lstbins, ref_time=data.times.min(), location=data.telescope.location
        ),
        time_ranges=lsts_to_times(
            lst_ranges, ref_time=data.times.min(), location=data.telescope.location
        ),
        auxiliary_measurements=None,
        effective_integration_time=intg_time,
    )
    return data
