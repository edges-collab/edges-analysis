"""Module defining composite models."""

from collections.abc import Sequence
from functools import cached_property
from typing import ClassVar

import attrs
import numpy as np
import yaml

from ..io.serialization import hickleable
from . import data_transforms as dt
from . import fitting
from .core import FixedLinearModel, Model


@hickleable
@attrs.define(frozen=True, kw_only=True, slots=False)
class CompositeModel:
    """
    Define a composite model from a set of sub-models.

    In totality, the resulting model is still
    """

    models: dict[str, Model] = attrs.field()
    data_transform: dt.DataTransform = attrs.field()

    @models.validator
    def _models_vld(self, attribute, value):
        if not all(isinstance(v, Model) for v in value.values()):
            raise ValueError("All models must be instances of Model.")

        mdls = list(value.values())
        if any(m.data_transform != mdls[0].data_transform for m in mdls):
            raise ValueError("All models must have the same data transform.")

    @data_transform.default
    def _data_transform_default(self):
        # The data transform from the models
        return self.models[next(iter(self.models.keys()))].data_transform

    @data_transform.validator
    def _data_transform_vld(self, attribute, value):
        idt = dt.IdentityTransform()
        if any(m.data_transform not in (value, idt) for m in self.models.values()):
            raise ValueError(
                "If data_transform is set, all models must not have it set."
            )

    @cached_property
    def n_terms(self) -> int:
        """The number of terms in the full composite model."""
        return sum(m.n_terms for m in self.models.values())

    @cached_property
    def parameters(self) -> np.ndarray | None:
        """The read-only list of parameters of all sub-models."""
        if any(m.parameters is None for m in self.models.values()):
            return None

        return np.concatenate(tuple(m.parameters for m in self.models.values()))

    @cached_property
    def _index_map(self):
        index_map = {}

        indx = 0
        for name, model in self.models.items():
            for i in range(model.n_terms):
                index_map[indx] = (name, i)
                indx += 1

        return index_map

    def __getitem__(self, item):
        """Get sub-models as if they were top-level attributes."""
        if item not in self.models:
            raise KeyError(f"{item} not one of the models.")

        return self.models[item]

    def _get_model_param_indx(self, model: str):
        indx = list(self.models.keys()).index(model)
        n_before = sum(m.n_terms for m in list(self.models.values())[:indx])
        model = self.models[model]
        return slice(n_before, n_before + model.n_terms, 1)

    @cached_property
    def model_idx(self) -> dict[str, slice]:
        """Dictionary of parameter indices correponding to each model."""
        return {name: self._get_model_param_indx(name) for name in self.models}

    def get_model(
        self,
        model: str,
        parameters: np.ndarray = None,
        x: np.ndarray | None = None,
        with_scaler: bool = True,
    ):
        """Calculate a sub-model."""
        indx = self.model_idx[model]

        model = self.models[model]

        if parameters is None:
            parameters = self.parameters

        if parameters is None:
            raise ValueError("Cannot evaluate a model without providing parameters!")

        p = parameters if len(parameters) == model.n_terms else parameters[indx]
        return model(x=x, parameters=p, with_scaler=with_scaler)

    def get_basis_term(self, indx: int, x: np.ndarray) -> np.ndarray:
        """Define the basis terms for the model."""
        model, indx = self._index_map[indx]

        return self[model].get_basis_term(indx, x)

    def get_basis_term_transformed(
        self, indx: int, x: np.ndarray, with_scaler: bool = True
    ) -> np.ndarray:
        """Get the basis function term after coordinate tranformation."""
        model, indx = self._index_map[indx]
        return self[model].get_basis_term_transformed(indx, x, with_scaler=with_scaler)

    def get_basis_terms(self, x: np.ndarray, with_scaler: bool = True) -> np.ndarray:
        """Get a 2D array of all basis terms at ``x``."""
        return np.array([
            self.get_basis_term_transformed(indx, x, with_scaler=with_scaler)
            for indx in range(self.n_terms)
        ])

    def with_nterms(
        self, model: str, n_terms: int | None = None, parameters: Sequence | None = None
    ) -> Model:
        """Return a new :class:`Model` with given nterms and parameters."""
        model_ = self[model]

        if parameters is not None:
            n_terms = len(parameters)

        model_ = model_.with_nterms(n_terms=n_terms, parameters=parameters)

        return attrs.evolve(self, models={**self.models, model: model_})

    def with_params(self, parameters: Sequence):
        """Get a new model with specified parameters."""
        assert len(parameters) == self.n_terms
        models = {
            name: model.with_params(
                parameters=parameters[self._get_model_param_indx(name)]
            )
            for name, model in self.models.items()
        }
        return attrs.evolve(self, models=models)

    def at(self, **kwargs) -> FixedLinearModel:
        """Get an evaluated linear model."""
        return FixedLinearModel(model=self, **kwargs)

    def __call__(
        self,
        x: np.ndarray | None = None,
        basis: np.ndarray | None = None,
        parameters: Sequence | None = None,
        indices: Sequence | slice = slice(None),
        with_scaler: bool = True,
    ) -> np.ndarray:
        """Evaluate the model.

        Parameters
        ----------
        x : np.ndarray, optional
            The co-ordinates at which to evaluate the model (by default, use
            ``default_x``).
        basis : np.ndarray, optional
            The basis functions at which to evaluate the model. This is useful if
            calling the model multiple times, as the basis itself can be cached and
            re-used.
        params
            A list/array of parameters at which to evaluate the model. Will use the
            instance's parameters if available. If using a subset of the basis
            functions, you can pass a subset of parameters.

        Returns
        -------
        model : np.ndarray
            The model evaluated at the input ``x`` or ``basis``.
        """
        return Model.__call__(
            self,
            x=x,
            basis=basis,
            parameters=parameters,
            indices=indices,
            with_scaler=with_scaler,
        )

    def fit(
        self,
        xdata: np.ndarray,
        ydata: np.ndarray,
        weights: np.ndarray | float = 1.0,
        **kwargs,
    ) -> fitting.ModelFit:
        """Create a linear-regression fit object."""
        return self.at(x=xdata).fit(ydata, weights=weights, **kwargs)


@hickleable
@attrs.define(frozen=True, slots=False)
class ComplexRealImagModel(yaml.YAMLObject):
    """A composite model that is specifically for complex functions in real/imag."""

    yaml_tag = "ComplexRealImagModel"

    real: Model | FixedLinearModel = attrs.field()
    imag: Model | FixedLinearModel = attrs.field()

    def at(self, **kwargs) -> FixedLinearModel:
        """Get an evaluated linear model."""
        return attrs.evolve(
            self,
            real=self.real.at(**kwargs),
            imag=self.imag.at(**kwargs),
        )

    def __call__(
        self,
        x: np.ndarray | None = None,
        parameters: Sequence | None = None,
    ) -> np.ndarray:
        """Evaluate the model.

        Parameters
        ----------
        x
            The co-ordinates at which to evaluate the model (by default, use
            ``default_x``).
        params
            A list/array of parameters at which to evaluate the model. Will use the
            instance's parameters if available. If using a subset of the basis
            functions, you can pass a subset of parameters.

        Returns
        -------
        model
            The model evaluated at the input ``x`` or ``basis``.
        """
        return self.real(
            x=x,
            parameters=parameters[: self.real.n_terms]
            if parameters is not None
            else None,
        ) + 1j * self.imag(
            x=x,
            parameters=parameters[self.real.n_terms :]
            if parameters is not None
            else None,
        )

    def fit(
        self,
        ydata: np.ndarray,
        weights: np.ndarray | float = 1.0,
        xdata: np.ndarray | None = None,
        **kwargs,
    ):
        """Create a linear-regression fit object."""
        if isinstance(self.real, FixedLinearModel):
            real = self.real
        else:
            real = self.real.at(x=xdata)

        if isinstance(self.imag, FixedLinearModel):
            imag = self.imag
        else:
            imag = self.imag.at(x=xdata)

        real = real.fit(np.real(ydata), weights=weights, **kwargs).fit
        imag = imag.fit(np.imag(ydata), weights=weights, **kwargs).fit
        return attrs.evolve(self, real=real, imag=imag)


@hickleable
@attrs.define(frozen=True, slots=False)
class ComplexMagPhaseModel(yaml.YAMLObject):
    """A composite model that is specifically for complex functions in mag/phase."""

    yaml_tag: ClassVar[str] = "ComplexMagPhaseModel"

    mag: Model | FixedLinearModel = attrs.field()
    phs: Model | FixedLinearModel = attrs.field()

    def at(self, **kwargs) -> FixedLinearModel:
        """Get an evaluated linear model."""
        return attrs.evolve(
            self,
            mag=self.mag.at(**kwargs),
            phs=self.phs.at(**kwargs),
        )

    def __call__(
        self,
        x: np.ndarray | None = None,
        parameters: Sequence | None = None,
    ) -> np.ndarray:
        """Evaluate the model.

        Parameters
        ----------
        x
            The co-ordinates at which to evaluate the model (by default, use
            ``default_x``).
        params
            A list/array of parameters at which to evaluate the model. Will use the
            instance's parameters if available. If using a subset of the basis
            functions, you can pass a subset of parameters.

        Returns
        -------
        model : np.ndarray
            The model evaluated at the input ``x`` or ``basis``.
        """
        return self.mag(
            x=x,
            parameters=parameters[: self.mag.n_terms]
            if parameters is not None
            else None,
        ) * np.exp(
            1j
            * self.phs(
                x=x,
                parameters=parameters[self.mag.n_terms :]
                if parameters is not None
                else None,
            )
        )

    def fit(
        self,
        ydata: np.ndarray,
        weights: np.ndarray | float = 1.0,
        xdata: np.ndarray | None = None,
        **kwargs,
    ):
        """Create a linear-regression fit object."""
        if isinstance(self.mag, FixedLinearModel):
            mag = self.mag
        else:
            mag = self.mag.at(x=xdata)

        if isinstance(self.phs, FixedLinearModel):
            phs = self.phs
        else:
            phs = self.phs.at(x=xdata)

        mag = mag.fit(np.abs(ydata), weights=weights, **kwargs).fit
        phs = phs.fit(np.unwrap(np.angle(ydata)), weights=weights, **kwargs).fit
        return attrs.evolve(self, mag=mag, phs=phs)
