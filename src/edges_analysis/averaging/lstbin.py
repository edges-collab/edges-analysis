"""Functions for doing LST binning on GSData objects."""

from __future__ import annotations

import logging
import warnings
from enum import Enum

import edges_cal.modelling as mdl
import numpy as np
import pygsdata.coordinates as crd
from astropy import units as un
from astropy.coordinates import Longitude
from astropy.time import Time
from pygsdata import GSData, GSFlag, gsregister
from pygsdata.coordinates import lsts_to_times
from pygsdata.utils import angle_centre

from ..datamodel import add_model
from .averaging import bin_data

logger = logging.getLogger(__name__)


class NsamplesStrategy(Enum):
    """An enumeration of strategies for computing Nsamples when combining data.

    Note that generally the strategy can influence *two* components of the calculation:
    firstly it influences how the data is _weighted_ when it is being averaged, and
    secondly, it influences what the final number of samples are (i.e. the effective
    variance of the data). Generally, these align, but some options specifically choose
    different conventions for each of these choices.

    Options
    -------
    FLAGGED_NSAMPLES
        Combine the underlying Nsamples, setting any flagged data to have zero samples.
    FLAGS_ONLY
        Count each datum as one sample, but only if it is not flagged.
    FLAGGED_NSAMPLES_UNIFORM
        Give each data that is both unflagged and has at least one sample a weight of
        unity in an average/model, but propagate nsamples using the full flagged
        nsamples.
    NSAMPLES_ONLY
        Only consider the Nsamples of the underlying data, not any flags.
    """

    FLAGGED_NSAMPLES = 0
    FLAGS_ONLY = 1
    FLAGGED_NSAMPLES_UNIFORM = 2
    NSAMPLES_ONLY = 3


def get_weights_from_strategy(
    data: GSData, strategy: NsamplesStrategy
) -> tuple[np.ndarray, np.ndarray]:
    """Compute weights and nsamples used for a particular strategy."""
    if strategy == NsamplesStrategy.FLAGGED_NSAMPLES:
        w = data.flagged_nsamples
        n = w
    elif strategy == NsamplesStrategy.FLAGS_ONLY:
        w = (~data.complete_flags).astype(float)
        n = data.flagged_nsamples
    elif strategy == NsamplesStrategy.FLAGGED_NSAMPLES_UNIFORM:
        w = (data.flagged_nsamples > 0).astype(float)
        n = data.flagged_nsamples
    elif strategy == NsamplesStrategy.NSAMPLES_ONLY:
        w = data.nsamples
        n = w
    else:
        raise ValueError(
            f"Invalid nsamples_strategy: {strategy}. "
            f"Must be a member of {NsamplesStrategy}"
        )
    return w, n


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
def average_over_times(
    data: GSData,
    nsamples_strategy: NsamplesStrategy = NsamplesStrategy.FLAGGED_NSAMPLES,
    use_resids: bool | None = None,
    fill_value: float = 0.0,
):
    """Average a GSData object over the time axis.

    Parameters
    ----------
    data : GSData object
        The data over which to average.
    nsamples_strategy : str, optional
        The strategy to use when defining the weights of each sample. Defaults to
        'flagged-nsamples'. The choices are:
        - 'flagged-nsamples': Use the flagged nsamples (i.e. set nsamples at flagged
            data to zero, otherwise use nsamples)
        - 'flags-only': Use the flags only (i.e. set nsamples at flagged data to
            zero, otherwise use 1)
        - 'flagged-nsamples-uniform': Use the flagged nsamples (i.e. set nsamples at
            flagged data to zero, and keep zero-samples as zero, otherwise use 1)
        - 'nsamples-only': Use the nsamples only (don't set nsamples at flagged
            data to zero)
    use_resids : bool, optional
        Whether to average the residuals and add them back to the mean model, or simply
        average the data directly.
    """
    if use_resids is None:
        use_resids = data.residuals is not None

    if use_resids and data.residuals is None:
        raise ValueError("Cannot average residuals without a model.")

    w, _n = get_weights_from_strategy(data, nsamples_strategy)

    ntot = np.sum(w, axis=-2)
    nsamples_tot = np.sum(_n, axis=-2)

    if use_resids:
        sum_resids = np.sum(data.residuals * w, axis=-2)
        mean_resids = sum_resids / ntot
        mean_model = np.mean(data.model, axis=-2)
        new_data = mean_model + mean_resids
    else:
        sum_data = np.sum(data.data * w, axis=-2)
        new_data = sum_data / ntot

    new_data[np.isnan(new_data)] = fill_value

    return data.update(
        data=new_data[:, :, None, :],
        residuals=mean_resids[:, :, None, :] if use_resids else None,
        times=np.atleast_2d(np.mean(data.times)),
        time_ranges=Time(
            np.array([[[data.time_ranges.min().jd, data.time_ranges.max().jd]]]),
            format="jd",
        ),
        lsts=Longitude(np.array([[np.mean(data.lsts.hour)]]) * un.hour),
        lst_ranges=Longitude(
            np.array([[[data.lst_ranges.hour.min(), data.lst_ranges.hour.max()]]])
            * un.hour
        ),
        effective_integration_time=np.mean(data.effective_integration_time, axis=2)[
            :, :, None
        ],
        nsamples=nsamples_tot[:, :, None],
        flags={},
    )


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
        first_edge = crd.gha2lst(first_edge * un.hourangle).to_value("hourangle")
        max_edge = crd.gha2lst(max_edge * un.hourangle).to_value("hourangle")

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

    lstbins = angle_centre(lst_ranges[..., 0], lst_ranges[..., 1])

    # Flag anything that is all nan -- these are just empty LSTs.
    flg = GSFlag(flags=np.all(np.isnan(spec), axis=(0, 1, 3)), axes=("time",))

    if data.auxiliary_measurements is not None:
        warnings.warn("Auxiliary measurements cannot be binned!", stacklevel=2)

    if data._effective_integration_time.size > 1:
        if np.unique(data._effective_integration_time).size > 1:
            warnings.warn(
                "lstbin does not yet support variable integration times!", stacklevel=2
            )
        intg_time = np.mean(data._effective_integration_time)
    else:
        intg_time = data._effective_integration_time

    times = lsts_to_times(
        np.where(lstbins < lstbins[0, 0], lstbins + 2 * np.pi * un.rad, lstbins),
        ref_time=data.times.min(),
        location=data.telescope.location,
    )
    time_ranges = lsts_to_times(
        np.where(
            lst_ranges < lst_ranges[0, 0, 0],
            lst_ranges + 2 * np.pi * un.rad,
            lst_ranges,
        ),
        ref_time=data.times.min(),
        location=data.telescope.location,
    )

    data = data.update(
        data=spec,
        residuals=resids,
        nsamples=nsmpls,
        flags={"empty_lsts": flg},
        lsts=lstbins,
        lst_ranges=lst_ranges,
        times=times,
        time_ranges=time_ranges,
        auxiliary_measurements=None,
        effective_integration_time=intg_time,
    )
    return data
