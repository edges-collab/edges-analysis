"""Functions for doing LST binning on GSData objects."""
from __future__ import annotations

import edges_cal.modelling as mdl
import numpy as np
from astropy import units as un
from astropy.coordinates import Longitude

from .. import coordinates as crd
from ..datamodel import add_model
from ..gsdata import GSData, GSFlag, gsregister
from .averaging import bin_data


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

    while max_edge < first_edge:
        max_edge += 24

    bins = np.arange(first_edge, max_edge, binsize)
    if np.isclose(bins.max() + binsize, max_edge):
        bins = np.append(bins, max_edge)

    return bins


# @gsregister("reduce")
# def lst_bin_direct(
#     data: GSData,
#     binsize: float,
#     first_edge: float = 0,
#     max_edge: float = 24.0,
#     in_gha: bool = False,
# ) -> GSData:
#     """Bin the data in the LST axis.

#     This function does a direct weighted averaged within each LST bin. This is biased
#     if the mean spectrum is not constant within each bin *and* the weights are not
#     uniform.

#     Parameters
#     ----------
#     data
#         The :class:`GSData` object to bin.
#     binsize
#         The size of the LST bins in hours.
#     first_edge
#         The first edge of the first bin in hours.

#     Returns
#     -------
#     GSData
#         A new :class:`GSData` object with the binned data.
#     """
#     if in_gha:
#         first_edge = crd.gha2lst(first_edge)
#         max_edge = crd.gha2lst(max_edge)

#     bins = get_lst_bins(binsize, first_edge, max_edge=max_edge)

#     if not data.in_lst:
#         data = data.to_lsts()

#     lsts = data.lst_array.copy().hour

#     lsts[lsts < bins[0]] += 24

#     weights = data.nsamples * (~data.complete_flags).astype(int)
#     specs = np.zeros(
#         (data.data.shape[0], data.data.shape[1], len(bins) - 1, data.data.shape[3])
#     )
#     wghts = np.zeros_like(specs)
#     for i, (d, w) in enumerate(zip(data.data, weights)):
#         _, specs[i], wghts[i] = bin_array_biased_regular(
#             data=d,
#             weights=w,
#             coords=lsts[:, i],
#             axis=1,
#             bins=bins,
#         )
#     times = Longitude((bins[1:] + bins[:-1]) / 2 * un.hour)
#     data = data.update(
#         time_array=np.repeat(times, data.data.shape[0]).reshape(
#             (len(times), data.data.shape[0])
#         ),
#         time_ranges=Longitude(
#             np.tile(
#                 np.array([bins[:-1], bins[1:]]).T[:, None, :],
#                 (1, data.data.shape[0], 1),
#             )
#             * un.hour
#         ),
#         data=specs,
#         nsamples=wghts,
#         flags={
#             "empty_lsts": GSFlag(
#                 flags=np.all(np.isnan(specs), axis=(0, 1, 3)), axes=("time",)
#             )
#         },
#     )

#     return data


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

        # for ipol in range(data.npols):
        #     # Averaging data within GHA bins
        #     (
        #         params[iload, ipol],
        #         resids[iload, ipol],
        #         nsmpls[iload, ipol],
        #     ) = bin_gha_unbiased_regular(
        #         data.data_model.parameters[iload, ipol],
        #         data.resids[iload, ipol],
        #         data.flagged_nsamples[iload, ipol],
        #         lsts[:, iload],
        #         bins,
        #     )
    times = (bins[1:] + bins[:-1]) / 2
    times = Longitude(np.tile(times, (data.nloads, 1)).T * un.hour)

    # Flag anything that is all nan -- these are just empty LSTs.
    flg = GSFlag(flags=np.all(np.isnan(spec), axis=(0, 1, 3)), axes=("time",))
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


# @gsregister("reduce")
# def lst_bin(
#     gsdata: GSData,
#     binsize: float = 24.0,
#     first_edge: float = 0.0,
#     max_edge: float = 24.0,
#     in_gha: bool = False,
# ) -> GSData:
#     """LST-bin the data, auto-choosing the best method available.

#     If a ``data_model`` exists on the :class:`GSData` object, this will use the
#     model-based method. Otherwise, it will use the direct method.

#     Parameters
#     ----------
#     data
#         The :class:`GSData` object to bin.
#     binsize
#         The size of the LST bins in hours.
#     first_edge
#         The first edge of the first bin in hours.
#     max_edge
#         Return no bins larger than max_edge.

#     Returns
#     -------
#     GSData
#         A new :class:`GSData` object with the binned data.
#     """
#     if gsdata.residuals is not None:
#         return lst_bin_with_models(
#             gsdata, binsize, first_edge=first_edge, max_edge=max_edge, in_gha=in_gha
#         )
#     else:
#         return lst_bin_direct(
#             gsdata, binsize, first_edge=first_edge, max_edge=max_edge, in_gha=in_gha
#         )
