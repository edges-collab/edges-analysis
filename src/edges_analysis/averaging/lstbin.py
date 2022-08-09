"""Functions for doing LST binning on GSData objects."""
from __future__ import annotations
from ..gsdata import GSData, gsregister
from .averaging import bin_array_biased_regular, bin_gha_unbiased_regular
import numpy as np
import edges_cal.modelling as mdl
from astropy import units as un
from astropy.coordinates import Longitude


def get_lst_bins(binsize: float, first_edge: float = 0) -> np.ndarray:
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
        raise ValueError("Binsize must be less than 24 hours")
    if binsize <= 0:
        raise ValueError("Binsize must be greater than 0 hours")

    if first_edge < -12:
        first_edge += 24

    if first_edge > 12:
        first_edge -= 24

    bins = np.arange(first_edge, first_edge + 24, binsize)
    if np.isclose(bins.max() + binsize, first_edge + 24):
        bins = np.append(bins, first_edge + 24)

    return bins


@gsregister("reduce")
def lst_bin_direct(data: GSData, binsize: float, first_edge: float = 0) -> GSData:
    """Bin the data in the LST axis.

    This function does a direct weighted averaged within each LST bin. This is biased
    if the mean spectrum is not constant within each bin *and* the weights are not
    uniform.

    Parameters
    ----------
    data
        The :class:`GSData` object to bin.
    binsize
        The size of the LST bins in hours.
    first_edge
        The first edge of the first bin in hours.

    Returns
    -------
    GSData
        A new :class:`GSData` object with the binned data.
    """
    bins = get_lst_bins(binsize, first_edge)

    if not data.in_lst:
        data = data.to_lsts()

    lsts = data.lst_array.copy().hour
    lsts[lsts < first_edge] += 24

    weights = data.nsamples * (~data.complete_flags).astype(int)

    _, spec, wght = bin_array_biased_regular(
        data=data.data,
        weights=weights,
        coords=data.lst_array,
        axis=1,
        bins=bins,
    )

    return data.update(
        time_array=Longitude(bins * un.hour), data=spec, nsamples=wght, flags={}
    )


@gsregister("reduce")
def lst_bin_with_models(
    data: GSData,
    binsize: float = 24.0,
    first_edge: float = 0.0,
    model: mdl.Model | None = None,
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

    Returns
    -------
    GSData
        A new :class:`GSData` object with the binned data.
    """
    if data.data_unit == "power":
        raise ValueError("Can't do LST-binning on power data")

    if data.data_model is None and model is None:
        raise ValueError("Cannot bin with models without a model in the data!")

    if data.npols > 1 or data.nloads > 1:
        raise NotImplementedError(
            "Can't do model-based LST binning on multi-pol/source data yet."
        )

    if data.data_model is None:
        data = data.add_model(model)

    bins = get_lst_bins(binsize, first_edge)

    if not data.in_lst:
        data = data.to_lsts()

    lsts = data.lst_array[:, 0].copy().hour
    lsts[lsts < first_edge] += 24

    # Averaging data within GHA bins
    params, resids, weights = bin_gha_unbiased_regular(
        np.squeeze(data.data_model.parameters),
        np.squeeze(data.resids),
        np.squeeze(data.flagged_nsamples),
        lsts,
        bins,
    )

    times = Longitude((bins[1:] + bins[:-1]) / 2 * un.hour)

    return data.update(
        data=resids[np.newaxis, np.newaxis],
        nsamples=weights[np.newaxis, np.newaxis],
        flags={},
        time_array=times[:, np.newaxis],
        data_unit="model_residuals",
        data_model=data.data_model.update(parameters=params[np.newaxis, np.newaxis]),
    )


@gsregister("reduce")
def lst_bin(
    gsdata: GSData,
    binsize: float = 24.0,
    first_edge: float = 0.0,
) -> GSData:
    """LST-bin the data, auto-choosing the best method available.

    If a ``data_model`` exists on the :class:`GSData` object, this will use the
    model-based method. Otherwise, it will use the direct method.

    Parameters
    ----------
    data
        The :class:`GSData` object to bin.
    binsize
        The size of the LST bins in hours.
    first_edge
        The first edge of the first bin in hours.

    Returns
    -------
    GSData
        A new :class:`GSData` object with the binned data.
    """
    if gsdata.data_model is not None:
        return lst_bin_with_models(gsdata, binsize, first_edge=first_edge)
    else:
        return lst_bin_direct(gsdata, binsize, first_edge)
