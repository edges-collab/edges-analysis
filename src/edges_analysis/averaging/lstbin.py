"""Functions for doing LST binning on GSData objects."""
from __future__ import annotations

import edges_cal.modelling as mdl
import logging
import numpy as np
from astropy import units as un
from astropy.coordinates import Longitude

from .. import coordinates as crd
from ..datamodel import add_model
from ..gsdata import GSData, GSFlag, gsregister
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
        logger.debug("YES, I AM TRYING TO FIX IT")
        max_edge += 24
    if max_edge > first_edge:
        logger.debug("APPARENTLY MAX_EDGE IS BIGGER?")

    logger.debug(
        "lst_bin: first_edge: %f, max_edge: %f, binsize: %f",
        first_edge,
        max_edge,
        binsize,
    )
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
        first_edge = crd.gha2lst(first_edge)
        max_edge = crd.gha2lst(max_edge)

    bins = get_lst_bins(binsize, first_edge, max_edge=max_edge)
    logger.debug(f"Got bins: {bins}")
    if not data.in_lst:
        data = data.to_lsts()

    lsts = data.lst_array.copy().hour

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

    times = (bins[1:] + bins[:-1]) / 2
    times = Longitude(np.tile(times, (data.nloads, 1)).T * un.hour)

    # Flag anything that is all nan -- these are just empty LSTs.
    flg = GSFlag(flags=np.all(np.isnan(spec), axis=(0, 1, 3)), axes=("time",))
    logger.debug(f"Flags in LST BIN: {flg}")
    logger.debug(f"Spec shape after LST bin: {spec.shape}")
    data = data.update(
        data=spec,
        residuals=resids,
        nsamples=nsmpls,
        flags={"empty_lsts": flg},
        time_array=times,
        time_ranges=Longitude(
            np.tile(np.array([bins[:-1], bins[1:]]).T[:, None, :], (1, data.nloads, 1))
            * un.hour
        ),
    )
    return data
