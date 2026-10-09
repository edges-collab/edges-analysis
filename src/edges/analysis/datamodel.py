"""Data models for GSData objects."""

import logging
from typing import Self

import h5py
import numpy as np
import yaml
from astropy import units as un
from attrs import define, evolve, field
from pygsdata import GSData
from pygsdata.attrs import npfield
from pygsdata.register import gsregister

from .. import modeling as mdl
from ..averaging import NsamplesStrategy, get_weights_from_strategy
from ..modeling.fitting import _RepeatedFit

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

    def _apply_model(self, gsdata: GSData, d: np.ndarray, sign: int) -> np.ndarray:
        """Add (``sign=1``) or subtract (``sign=-1``) the model from each spectrum."""
        d = d.reshape((-1, gsdata.nfreqs))
        p = self.parameters.reshape((-1, self.nparams))

        model = self.model.at(x=gsdata.freqs.to_value("MHz"))

        out = np.zeros_like(d)

        # Work on the values of dimensionless quantities: the arithmetic is
        # identical, but much faster.
        d_values, out_values = _plain_values(d), _plain_values(out)
        if d_values is None or out_values is None:
            d_values, out_values = d, out

        for i, (dd, pp) in enumerate(zip(d_values, p, strict=False)):
            if sign > 0:
                out_values[i] = dd + model(parameters=pp)
            else:
                out_values[i] = dd - model(parameters=pp)

        out.shape = gsdata.data.shape
        return out

    def get_residuals(self, gsdata: GSData) -> np.ndarray:
        """Calculate the residuals of the model given the input GSData object."""
        return self._apply_model(gsdata, gsdata.data, sign=-1)

    def get_spectra(self, gsdata: GSData) -> np.ndarray:
        """Calculate the data spectra given the input GSData object."""
        return self._apply_model(gsdata, gsdata.residuals, sign=1)

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

        params = None
        d_values = _plain_values(d)
        if d_values is not None and set(fit_kwargs) <= {"method"}:
            try:
                params = _fit_spectra_grouped(xmodel, d_values, w, **fit_kwargs)
            except (np.linalg.LinAlgError, ValueError):
                # Re-do the fits one at a time, to raise the error for the first
                # failing spectrum.
                params = None

        if params is None:
            params = np.zeros((d.shape[0], model.n_terms))
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


def _plain_values(arr: np.ndarray) -> np.ndarray | None:
    """The values of ``arr`` as a plain array, or None if it has a non-trivial unit.

    For dimensionless (unscaled) quantities, arithmetic on the values is identical
    to that on the quantities, but faster.
    """
    if not isinstance(arr, un.Quantity):
        return np.asarray(arr)
    if arr.unit == un.dimensionless_unscaled:
        return arr.view(np.ndarray)
    return None


def _group_equal_rows(*arrays: np.ndarray) -> list[list[int]]:
    """Group the row indices for which the rows of all ``arrays`` are exactly equal.

    Rows are first bucketed by a cheap fingerprint (a random projection), and the
    rows in a bucket are then compared exactly, so the grouping does not rely on the
    fingerprint being collision-free.

    Parameters
    ----------
    arrays
        2D arrays with the same number of rows.

    Returns
    -------
    groups
        The row indices of each group, in order of first appearance.
    """
    rng = np.random.default_rng(0)
    fingerprint = sum(
        np.asarray(arr, dtype=float) @ rng.uniform(1, 2, size=arr.shape[1])
        for arr in arrays
    )

    buckets: dict[float, list[list[int]]] = {}
    groups: list[list[int]] = []
    for i, fp in enumerate(fingerprint.tolist()):
        candidates = buckets.setdefault(fp, [])
        for group in candidates:
            if all(np.array_equal(arr[i], arr[group[0]]) for arr in arrays):
                group.append(i)
                break
        else:
            candidates.append([i])
            groups.append(candidates[-1])
    return groups


def _fit_spectra_grouped(
    xmodel: mdl.FixedLinearModel,
    data: np.ndarray,
    weights: np.ndarray,
    method: str = "lstsq",
) -> np.ndarray:
    """Fit the model to each spectrum, sharing the solve setup between spectra.

    Spectra with identical weights and identical masks of finite data are fitted with
    one :class:`~edges.modeling.fitting._RepeatedFit`, which prepares the (weighted)
    design matrix once. The parameters are identical to fitting each spectrum with
    ``xmodel.fit(ydata=spectrum, weights=w, method=method)``.

    Parameters
    ----------
    xmodel
        The model, fixed at the frequencies of the spectra.
    data
        The spectra, shape ``(nspectra, nfreqs)``.
    weights
        The weights of the spectra, with the same shape as ``data``.
    method
        The fitting method (see :class:`~edges.modeling.ModelFit`).

    Returns
    -------
    params
        The best-fit parameters of each spectrum, shape ``(nspectra, n_terms)``.
    """
    weights = np.asarray(weights, dtype=float)
    params = np.zeros((data.shape[0], xmodel.n_terms))
    for members in _group_equal_rows(weights, np.isfinite(data)):
        first = members[0]
        if len(members) == 1:
            params[first] = xmodel.fit(
                ydata=data[first], weights=weights[first], method=method
            ).model_parameters
            continue

        fitter = _RepeatedFit(xmodel, weights=weights[first], method=method)
        for i in members:
            params[i] = fitter.parameters(data[i])

    return params


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
