"""Data models for GSData objects."""

import logging

import h5py
import numpy as np
import yaml
from attrs import define, evolve, field

from .. import modeling as mdl

try:
    from typing import Self
except ImportError:
    from typing import Self

from pygsdata import GSData
from pygsdata.attrs import npfield
from pygsdata.register import gsregister

from ..averaging import NsamplesStrategy, get_weights_from_strategy

logger = logging.getLogger(__name__)


@define
class GSDataLinearModel:
    """A model of a GSData object."""

    model: mdl.Model = field()
    parameters: np.ndarray = npfield(possible_ndims=(4,))

    @parameters.validator
    def _params_vld(self, att, val):
        if val.shape[-1] != self.model.n_terms:
            raise ValueError(
                f"parameters array has {val.shape[-1]} parameters, "
                f"but model has {self.model.n_terms}"
            )

    @property
    def nloads(self) -> int:
        """Number of loads in the model."""
        return self.parameters.shape[0]

    @property
    def npols(self) -> int:
        """Number of polarisations in the model."""
        return self.parameters.shape[1]

    @property
    def ntimes(self) -> int:
        """Number of times in the model."""
        return self.parameters.shape[2]

    @property
    def nparams(self) -> int:
        """Number of parameters in the model."""
        return self.parameters.shape[3]

    def get_residuals(self, gsdata: GSData) -> np.ndarray:
        """Calculate the residuals of the model given the input GSData object."""
        d = gsdata.data.reshape((-1, gsdata.nfreqs))
        p = self.parameters.reshape((-1, self.nparams))

        model = self.model.at(x=gsdata.freqs.to_value("MHz"))

        resids = np.zeros_like(d)
        for i, (dd, pp) in enumerate(zip(d, p, strict=False)):
            resids[i] = dd - model(parameters=pp)

        resids.shape = gsdata.data.shape
        return resids

    def get_spectra(self, gsdata: GSData) -> np.ndarray:
        """Calculate the data spectra given the input GSData object."""
        d = gsdata.residuals.reshape((-1, gsdata.nfreqs))
        p = self.parameters.reshape((-1, self.nparams))

        model = self.model.at(x=gsdata.freqs.to_value("MHz"))

        spectra = np.zeros_like(d)
        for i, (dd, pp) in enumerate(zip(d, p, strict=False)):
            spectra[i] = dd + model(parameters=pp)

        spectra.shape = gsdata.data.shape
        return spectra

    @classmethod
    def from_gsdata(
        cls,
        model: mdl.Model,
        gsdata: GSData,
        nsamples_strategy: NsamplesStrategy.FLAGGED_NSAMPLES,
        **fit_kwargs,
    ) -> Self:
        """Create a GSDataModel from a GSData object.

        Parameters
        ----------
        model
            The model to use. Applied separately to each time, load and pol.
        gsdata : GSData object
            The GSData object to fit to.
        nsamples_strategy
            The strategy to use when defining the weights of each sample.
        """
        shp = (-1, gsdata.nfreqs)
        d = gsdata.data.reshape(shp)
        w = get_weights_from_strategy(gsdata, nsamples_strategy)[0].reshape(shp)
        xmodel = model.at(x=gsdata.freqs.to_value("MHz"))

        params = np.zeros((gsdata.nloads * gsdata.npols * gsdata.ntimes, model.n_terms))

        try:
            for i, (dd, ww) in enumerate(zip(d, w, strict=False)):
                params[i] = xmodel.fit(
                    ydata=dd, weights=ww, **fit_kwargs
                ).model_parameters
        except np.linalg.LinAlgError as e:
            raise ValueError(
                f"Linear algebra error: {e}.\nIndex={i}\ndata={dd}\nweights={ww}"
            ) from e

        params.shape = (gsdata.nloads, gsdata.npols, gsdata.ntimes, model.n_terms)
        return cls(model=model, parameters=params)

    def update(self, **kw) -> Self:
        """Return a new GSDataModel instance with updated attributes."""
        return evolve(self, **kw)

    def write(self, fl: h5py.File | h5py.Group, path: str = ""):
        """Write the object to an HDF5 file, potentially to a particular path."""
        grp = fl.create_group(path) if path else fl
        grp.attrs["model"] = yaml.dump(self.model)
        grp.create_dataset("parameters", data=self.parameters)

    @classmethod
    def from_h5(cls, fl: h5py.File | h5py.Group, path: str = "") -> Self:
        """Read the object from an HDF5 file, potentially from a particular path."""
        grp = fl[path] if path else fl
        model = yaml.load(grp.attrs["model"], Loader=yaml.FullLoader)
        params = grp["parameters"][Ellipsis]
        return cls(model=model, parameters=params)


@gsregister("supplement")
def add_model(
    data: GSData,
    *,
    model: mdl.Model,
    nsamples_strategy: NsamplesStrategy = NsamplesStrategy.FLAGGED_NSAMPLES,
) -> GSData:
    """Return a new GSData instance which contains a data model.

    Parameters
    ----------
    data
        The GSData instance to add the model to.
    model
        The model to add/fit.
    append_to_file
        Whether to directly add the model residuals to the file that is attached to the
        GSData object. DON'T DO THIS.
    nsamples_strategy
        The strategy to use when defining the weights of each sample.
    """
    data_model = GSDataLinearModel.from_gsdata(
        model, data, nsamples_strategy=nsamples_strategy
    )
    return data.update(residuals=data_model.get_residuals(data))
