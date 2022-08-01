"""Functions for binning GSData objects in frequency."""

from ..gsdata import GSData, register_gsprocess
from .averaging import bin_array_biased_regular, bin_freq_unbiased_regular
import numpy as np
import edges_cal.modelling as mdl
from astropy import units as un

def freq_bin_direct(data: GSData, resolution: int | float) -> GSData:
    """
    Bin the data in the LST axis.
    """

    weights = data.nsamples * (~data.complete_flags).astype(int)

    new_freqs, spec, wght = bin_array_biased_regular(
        data = data.data,
        weights = weights,
        coords = data.freq_array,
        axis=2,
        bins=resolution,
    )


    return data.update(freq_array=new_freqs, data=spec, nsamples=wght, flags=())

@register_gsprocess
def freq_bin_with_models(
    data: GSData, 
    resolution: int | float, 
    model: mdl.Model | None = None, 
):

    if data.data_model is None and model is None:
        raise ValueError("Cannot bin with models without a model in the data!")

    if data.data_model is None:
        data = data.add_model(model)
    
    # Averaging data within GHA bins
    f, weights, _ , resids, params = bin_freq_unbiased_regular(
        model=data.data_model.model,
        params=data.data_model.parameters,
        freq=data.freq_array.to_value("MHz"),
        resids=data.resids,
        weights=data.flagged_nsamples,
        resolution=resolution,
    )

    return data.update(
        data=resids, 
        nsamples=weights, 
        flags=(), 
        freq_array = f*un.MHz,
        data_model=data.data_model.update(parameters=params)
    )
