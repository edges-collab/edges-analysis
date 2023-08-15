"""Data models for GSData objects."""
from __future__ import annotations

import h5py
import logging
import numpy as np
import yaml
from attrs import define, evolve, field
from edges_cal import modelling as mdl

from .attrs import npfield
from .gsdata import GSData
from .register import gsregister

logger = logging.getLogger(__name__)


@define
class GSDataModel:
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
        """Calculates the residuals of the model given the input GSData object."""
        d = gsdata.spectra.reshape((-1, gsdata.nfreqs))
        p = self.parameters.reshape((-1, gsdata.data_model.nparams))

        model = self.model.at(x=gsdata.freq_array.to_value("MHz"))

        resids = np.zeros_like(d)
        for i, (dd, pp) in enumerate(zip(d, p)):
            resids[i] = dd - model(parameters=pp)

        resids.shape = gsdata.data.shape
        return resids

    def get_spectra(self, gsdata: GSData) -> np.ndarray:
        """Calculates the data spectra given the input GSData object."""
        d = gsdata.resids.reshape((-1, gsdata.nfreqs))
        p = self.parameters.reshape((-1, self.nparams))

        model = self.model.at(x=gsdata.freq_array.to_value("MHz"))

        spectra = np.zeros_like(d)
        for i, (dd, pp) in enumerate(zip(d, p)):
            spectra[i] = dd + model(parameters=pp)

        spectra.shape = gsdata.data.shape
        return spectra

    @classmethod
    def from_gsdata(cls, model: mdl.Model, gsdata: GSData, **fit_kwargs) -> GSDataModel:
        """Creates a GSDataModel from a GSData object."""
        d = gsdata.spectra.reshape((-1, gsdata.nfreqs))
        w = gsdata.flagged_nsamples.reshape((-1, gsdata.nfreqs))

        xmodel = model.at(x=gsdata.freq_array.to_value("MHz"))

        params = np.zeros((gsdata.nloads * gsdata.npols * gsdata.ntimes, model.n_terms))

        for i, (dd, ww) in enumerate(zip(d, w)):
            try:
                params[i] = xmodel.fit(
                    ydata=dd, weights=ww, **fit_kwargs
                ).model_parameters
            except np.linalg.LinAlgError as e:
                raise ValueError(
                    f"Linear algebra error: {e}.\nIndex={i}\ndata={dd}\nweights={ww}"
                ) from e

        params.shape = (gsdata.nloads, gsdata.npols, gsdata.ntimes, model.n_terms)
        return cls(model=model, parameters=params)

    def update(self, **kw) -> GSDataModel:
        """Return a new GSDataModel instance with updated attributes."""
        return evolve(self, **kw)

    def write(self, fl: h5py.File | h5py.Group, path: str = ""):
        """Write the object to an HDF5 file, potentially to a particular path."""
        grp = fl.create_group(path) if path else fl
        grp.attrs["model"] = yaml.dump(self.model)
        grp.create_dataset("parameters", data=self.parameters)

    @classmethod
    def from_h5(cls, fl: h5py.File | h5py.Group, path: str = "") -> GSDataModel:
        """Read the object from an HDF5 file, potentially from a particular path."""
        grp = fl[path] if path else fl
        model = yaml.load(grp.attrs["model"], Loader=yaml.FullLoader)
        params = grp["parameters"][Ellipsis]
        return cls(model=model, parameters=params)


@gsregister("supplement")
def add_model(data: GSData, *, model: mdl.Model, append_to_file: bool | None = None):
    """Return a new GSData instance which contains a data model."""
    new = data.update(data_model=GSDataModel.from_gsdata(model, data))

    if append_to_file is None:
        append_to_file = new.filename is not None

    if append_to_file and new.filename is None:
        raise ValueError(
            "Cannot append to file without a filename specified on the object!"
        )

    if append_to_file:
        with h5py.File(new.filename, "a") as fl:
            if "data_model" in fl.keys():
                logger.warning(
                    f"Data model already exists in {new.filename}, not overwriting."
                )
            elif fl["data"].shape[:3] != new.data_model.parameters.shape[:3]:
                logger.warning(
                    "File on disk is incompatible with the data model. Write new file."
                )
            else:
                new.data_model.write(fl, "data_model")

    return new
