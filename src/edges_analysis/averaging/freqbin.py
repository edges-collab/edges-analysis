"""Functions for binning GSData objects in frequency."""
from __future__ import annotations

import edges_cal.modelling as mdl
from astropy import units as un

<<<<<<< Updated upstream
from ..gsdata import GSData, add_model, gsregister

=======
from ..gsdata import GSData, add_model, gsregister

>>>>>>> Stashed changes
from .averaging import bin_array_biased_regular, bin_freq_unbiased_regular


def freq_bin_direct(data: GSData, resolution: int | float) -> GSData:
    """Bin the data in the LST axis.

    This function does a direct weighted averaged within each LST bin. This is biased
    if the mean spectrum is not constant within each bin *and* the weights are not
    uniform.

    Parameters
    ----------
    data
        The :class:`GSData` object to bin.
    resolution
        The resolution of the binning in MHz or number of existing channels (if int).
    """
    weights = data.nsamples * (~data.complete_flags).astype(int)

    new_freqs, spec, wght = bin_array_biased_regular(
        data=data.data,
        weights=weights,
        coords=data.freq_array,
        axis=2,
        bins=resolution,
    )

    return data.update(freq_array=new_freqs, data=spec, nsamples=wght, flags=())


@gsregister("reduce")
def freq_bin_with_models(
    data: GSData,
    resolution: int | float,
    model: mdl.Model | None = None,
):
    """Bin the data in the LST axis using model information to de-bias the mean.

    Parameters
    ----------
    data
        The :class:`GSData` object to bin.
    resolution
        The resolution of the binning in MHz or number of existing channels (if int).
    model
        A :class:`edges_cal.modelling.Model` object to use for de-biasing the mean. If
        the ``data`` object has a ``data_model`` defined, this will be used instead.
    """
    if data.data_model is None and model is None:
        raise ValueError("Cannot bin with models without a model in the data!")

    if data.data_model is None:
<<<<<<< Updated upstream
        data = add_model(data, model=model, append_to_file=False)
=======
        data = add_model(data,model=model, append_to_file=False)
>>>>>>> Stashed changes

    # Averaging data within GHA bins
    f, weights, _, resids, params = bin_freq_unbiased_regular(
        model=data.data_model.model,
        params=data.data_model.parameters.reshape((-1, data.data_model.model.n_terms)),
        freq=data.freq_array.to_value("MHz"),
        resids=data.resids.reshape((-1, data.nfreqs)),
        weights=data.flagged_nsamples.reshape((-1, data.nfreqs)),
        resolution=resolution,
    )

    params = params.reshape(data.data.shape[:-1] + (data.data_model.model.n_terms,))
    return data.update(
        data=resids.reshape(data.data.shape[:-1] + (-1,)),
        nsamples=weights.reshape(data.data.shape[:-1] + (-1,)),
        flags={},
        freq_array=f * un.MHz,
        data_model=data.data_model.update(parameters=params),
    )


@gsregister("reduce")
def freq_bin(
    data: GSData,
    resolution: int | float,
    model: mdl.Model | None = None,
) -> GSData:
    """Frequency binning that auto-selects which kind of binning to do.

    Parameters
    ----------
    data
        The :class:`GSData` object to bin.
    resolution
        The resolution of the binning in MHz or number of existing channels (if int).
    model
        A :class:`edges_cal.modelling.Model` object to use for de-biasing the mean. If
        the ``data`` object has a ``data_model`` defined, this will be used instead.
        If not provided, simple direct binning will be used.
    """
    if data.data_model is None and model is None:
        return freq_bin_direct(data, resolution)
    else:
        return freq_bin_with_models(data, resolution, model)
