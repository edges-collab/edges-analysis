"""Functions for doing LST binning on GSData objects."""
from ..gsdata import GSData, register_gsprocess
from .averaging import bin_array_biased_regular, bin_gha_unbiased_regular
import numpy as np
import edges_cal.modelling as mdl

def get_lst_bins(binsize: float, first_edge: float=0) -> np.ndarray:
    """
    Get the LST bins.
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

@register_gsprocess
def lst_bin_direct(data: GSData, binsize: float, first_edge: float = 0) -> 'GSData':
    """
    Bin the data in the LST axis.
    """
    bins = get_lst_bins(binsize, first_edge)

    if not data.in_lst:
        data = data.to_lsts()
    
    lsts = data.lst_array.copy()
    lsts[lsts < first_edge] += 24

    weights = data.nsamples * (~data.complete_flags).astype(int)

    _, spec, wght = bin_array_biased_regular(
        data = data.data,
        weights = weights,
        coords = data.lst_array,
        axis=1,
        bins=bins,
    )


    return data.update(time_array=bins, data=spec, nsamples=wght, flags=())

@register_gsprocess
def lst_bin_with_models(
    data: GSData, 
    binsize: float = 24.0,
    model: mdl.Model | None = None,
    first_edge: float=0.0,
):
    """LST-bin by using model information to de-bias the mean."""
    if data.data_unit == 'power':
        raise ValueError("Can't do LST-binning on power data")
    
    if data.data_model is None and model is None:
        raise ValueError("Cannot bin with models without a model in the data!")

    if data.data_model is None:
        data = data.add_model(model)

    bins = get_lst_bins(binsize, first_edge)

    if not data.in_lst:
        data = data.to_lsts()

    lsts = data.lst_array.copy()
    lsts[lsts < first_edge] += 24

    # Averaging data within GHA bins
    params, resids, weights = bin_gha_unbiased_regular(
        data.data_model.parameters, data.resids, data.flagged_nsamples, lsts, bins
    )

    times = (bins[1:] + bins[:-1])/2
    
    return data.update(
        data=resids, nsamples=weights, flags=(), time_array = times, data_unit='model_residuals',
        data_model=data.data_model.update(parameters=params)
    )

@register_gsprocess
def lst_bin(
    gsdata: GSData, 
    binsize: float = 24.0,
    first_edge: float=0.0,
) -> GSData:
    """
    LST-bin the data.
    """
    if gsdata.data_model is not None:
        return lst_bin_with_models(gsdata, binsize, first_edge=first_edge)
    else:
        return lst_bin_direct(gsdata, binsize, first_edge)