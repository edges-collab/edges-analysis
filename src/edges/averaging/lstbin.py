"""Functions for doing LST binning on GSData objects."""

import itertools
import logging
import warnings

import numpy as np
import pygsdata.coordinates as crd
from astropy import units as un
from astropy.coordinates import Longitude
from astropy.time import Time
from pygsdata import GSData, GSFlag, gsregister
from pygsdata.coordinates import lsts_to_times
from pygsdata.utils import angle_centre

from .averaging import bin_data
from .utils import NsamplesStrategy, get_weights_from_strategy

logger = logging.getLogger(__name__)


def get_lst_bins(
    binsize: float, first_edge: float = 0, max_edge: float = 24
) -> np.ndarray:
    """Determine LST bins given a bin size and first edge, in hours.

    This function will return equi-spaced bins starting at `first_edge` and with
    width `binsize`. The last bin will be less than or equal to `max_edge`, after
    accounting for wrapping at 24 hours.

    Parameters
    ----------
    binsize
        The size of the bins in hours.
    first_edge
        The first edge of the first bin.
    max_edge
        The maximum edge of the last bin.

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
    reference_lst: Longitude = Longitude(12 * un.hour),
    use_resids: bool | None = None,
    fill_value: float = 0.0,
) -> GSData:
    """Average a GSData object over the time axis.

    Parameters
    ----------
    data
        The data over which to average.
    nsamples_strategy
        The strategy to use when defining the weights of each sample. See
        :class:`~edges.analysis.averaging.NsamplesStrategy` for more information.
    reference_lst
        An LST set as the central LST when finding the new mean LST. All LSTs will
        be wrapped within 12 hours of this reference before taking the mean.
    use_resids : bool, optional
        Whether to average the residuals and add them back to the mean model, or simply
        average the data directly.
    fill_value : float
        The value to impute when no data exists in a bin.
    """
    if use_resids is None:
        use_resids = data.residuals is not None

    if use_resids and data.residuals is None:
        raise ValueError("Cannot average residuals without a model.")

    w, n = get_weights_from_strategy(data, nsamples_strategy)

    ntot = np.sum(w, axis=-2)
    nsamples_tot = np.sum(n, axis=-2)

    if use_resids:
        sum_resids = np.nansum(data.residuals * w, axis=-2)
        mean_resids = sum_resids / ntot
        mean_model = np.nanmean(data.model, axis=-2)
        new_data = mean_model + mean_resids
    else:
        sum_data = np.nansum(data.data * w, axis=-2)
        new_data = sum_data / ntot

    new_data[np.isnan(new_data)] = fill_value

    # The new time will be the mean unflagged time
    ww = np.any(w > 0, axis=(0, 1, 3))
    times = Time(np.atleast_2d(np.mean(data.times.jd[ww], axis=0)), format="jd")
    time_ranges = Time(
        np.array([
            [
                data.time_ranges.jd[ww].min(axis=(0, 2)),
                data.time_ranges.jd[ww].max(axis=(0, 2)),
            ]
        ]).transpose((0, 2, 1)),
        format="jd",
    )

    # Wrap the LSTs into +-12 hours of the reference LST.
    # Note that we de-unit the quantities to do the wrapping
    # because astropy does weird things when trying to wrap
    # a Longitude/Angle
    lsts = data.lsts.hour.copy()
    lsts[lsts <= reference_lst.hour - 12] += 24
    lsts[lsts > reference_lst.hour + 12] -= 24

    lst_ranges = data.lst_ranges.hour.copy()
    lst_ranges[lst_ranges <= reference_lst.hour - 12] += 24
    lst_ranges[lst_ranges > reference_lst.hour + 12] -= 24

    if data.auxiliary_measurements is not None:
        new_aux = {
            key: np.array([np.nanmean(data.auxiliary_measurements[key])])
            for key in data.auxiliary_measurements.columns
        }
    else:
        new_aux = None
    return data.update(
        data=new_data[:, :, None, :],
        residuals=mean_resids[:, :, None, :] if use_resids else None,
        times=times,
        time_ranges=time_ranges,
        lsts=Longitude(np.atleast_2d(np.mean(lsts[ww], axis=0)) * un.hourangle),
        lst_ranges=Longitude(
            np.array([
                [
                    lst_ranges[ww, :, 0].min(axis=0),
                    lst_ranges[ww, :, 1].max(axis=0),
                ]
            ]).transpose((0, 2, 1))
            * un.hourangle
        ),
        effective_integration_time=np.mean(data.effective_integration_time, axis=2)[
            :, :, None
        ],
        nsamples=nsamples_tot[:, :, None],
        flags={},
        auxiliary_measurements=new_aux,
    )


@gsregister("reduce")
def lst_bin(
    data: GSData,
    binsize: float = 24.0,
    first_edge: float = 0.0,
    max_edge: float = 24.0,
    in_gha: bool = False,
    use_model_residuals: bool | None = None,
    reference_time: float | Time | str = "mean",
):
    """Average data within bins of LST.

    Parameters
    ----------
    data
        The :class:`GSData` object to bin.
    binsize
        The size of the LST bins in hours.
    first_edge
        The first edge of the first bin in hours.
    max_edge
        The maximum edge of the last bin in hours, see :func:`get_lst_bins`.
    in_gha
        Whether to bin in GHA or LST (default).
    use_model_residuals
        Whether to use the model residuals to de-bias the mean in each bin. If
        True, either `model` must be provided, or `residuals` must be specified on the
        GSData object.
    reference_time
        The JD at which to reference the LSTs to in the output. The JDs of the output
        will be exactly at the centre of each LST bin, but the _day_ to which they are
        referenced will be set by the `reference_time` (all will be within 24 hours of
        this time). This can be a float (JD), an astropy Time, or one of 'min', 'max',
        'mean' or 'closest' (default). Options 'min', 'max' and 'mean' will use the
        corresponding min/max/mean time in the data object to set the reference time.

    Returns
    -------
    GSData
        A new :class:`GSData` object with the binned data.
    """
    if use_model_residuals is None:
        use_model_residuals = data.residuals is not None

    if use_model_residuals and data.residuals is None:
        raise ValueError("Cannot bin with models without residuals!")

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
            for b in itertools.pairwise(bins)
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
        np.array(list(itertools.pairwise(bins)))[:, None], (1, data.nloads, 1)
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

    # Determine a reference time. The output GSDatawill still need to have "times" in JD
    # which should be exactly at the centre of each LST bin. However, the choice of
    # which JD each LST bin should correspond to is somewhat arbitrary.
    if reference_time == "min":
        reference_time = data.times.min()
    elif reference_time == "max":
        reference_time = data.times.max()
    elif reference_time == "mean":
        reference_time = data.times.mean()

    times = lsts_to_times(
        np.where(lstbins < lstbins[0, 0], lstbins + 2 * np.pi * un.rad, lstbins),
        ref_time=reference_time,
        location=data.telescope.location,
    )
    time_ranges = lsts_to_times(
        np.where(
            lst_ranges < lst_ranges[0, 0, 0],
            lst_ranges + 2 * np.pi * un.rad,
            lst_ranges,
        ),
        ref_time=reference_time,
        location=data.telescope.location,
    )

    return data.update(
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
