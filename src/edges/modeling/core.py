"""Core classes for linear modelling."""

from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Sequence
from functools import cached_property
from typing import ClassVar, Self

import attrs
import numpy as np
import yaml

from ..io.serialization import hickleable
from ..tools import as_readonly
from . import composite, fitting
from . import data_transforms as dt
from . import xtransforms as xt

_MODELS = {}


@hickleable
@attrs.define(frozen=True, kw_only=True, slots=False)
class Model(metaclass=ABCMeta):
    """A base class for a linear model."""

    default_n_terms: ClassVar[int | None] = None
    n_terms_min: ClassVar[int] = 1
    n_terms_max: ClassVar[int] = 1000000

    parameters: tuple | None = attrs.field(
        default=None,
        converter=attrs.converters.optional(tuple),
    )
    n_terms: int = attrs.field(converter=attrs.converters.optional(int))
    _transform: xt.XTransform = attrs.field(default=xt.IdentityTransform())
    xtransform: xt.XTransform | None = attrs.field()
    basis_scaler: Callable | None = attrs.field(default=None)
    data_transform: dt.DataTransform = attrs.field(default=dt.IdentityTransform())

    def __init_subclass__(cls, is_meta=False, **kwargs):
        """Initialize a subclass and add it to the registered models."""
        super().__init_subclass__(**kwargs)
        if not is_meta:
            _MODELS[cls.__name__.lower()] = cls

    @n_terms.default
    def _n_terms_default(self):
        if self.parameters is not None:
            return len(self.parameters)
        return self.__class__.default_n_terms

    @n_terms.validator
    def _n_terms_validator(self, att, val):
        if val is None:
            raise ValueError("Either n_terms or explicit parameters must be given.")

        if not (self.n_terms_min <= val <= self.n_terms_max):
            raise ValueError(
                f"n_terms must be between {self.n_terms_min} and {self.n_terms_max}"
            )

        if self.parameters is not None and val != len(self.parameters):
            raise ValueError(f"Wrong number of parameters! Should be {val}.")

    @xtransform.default
    def _xt_default(self):
        return self._transform

    @abstractmethod
    def get_basis_term(self, indx: int, x: np.ndarray) -> np.ndarray:
        """Define the basis terms for the model."""

    def get_basis_term_transformed(
        self, indx: int, x: np.ndarray, with_scaler: bool = True
    ) -> np.ndarray:
        """Get the basis term after coordinate transformation."""
        s = self.basis_scaler(x) if with_scaler and self.basis_scaler is not None else 1
        return self.get_basis_term(indx=indx, x=self.xtransform(x)) * s

    def get_basis_terms(self, x: np.ndarray, with_scaler: bool = True) -> np.ndarray:
        """Get a 2D array of all basis terms at ``x``."""
        x = self.xtransform(x)
        s = self.basis_scaler(x) if with_scaler and self.basis_scaler is not None else 1

        return np.array([
            self.get_basis_term(indx, x) * s for indx in range(self.n_terms)
        ])

    def with_nterms(
        self, n_terms: int | None = None, parameters: Sequence | None = None
    ) -> Self:
        """Return a new :class:`Model` with given nterms and parameters."""
        if parameters is not None:
            n_terms = len(parameters)

        return attrs.evolve(self, n_terms=n_terms, parameters=parameters)

    def with_params(self, parameters: Sequence | None) -> Self:
        """Get new model with different parameters."""
        assert len(parameters) == self.n_terms
        return self.with_nterms(parameters=parameters)

    @staticmethod
    def from_str(model: str, **kwargs) -> Self:
        """Obtain a :class:`Model` given a string name."""
        return get_mdl(model)(**kwargs)

    def at(self, **kwargs) -> "FixedLinearModel":
        """Get an evaluated linear model."""
        return FixedLinearModel(model=self, **kwargs)

    def __call__(
        self,
        x: np.ndarray | None = None,
        basis: np.ndarray | None = None,
        parameters: Sequence | None = None,
        indices: Sequence[int] | slice = slice(None),
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
        indices
            Specifies which parameters/basis functions to use. Default is all of them.

        Returns
        -------
        model : np.ndarray
            The model evaluated at the input ``x`` or ``basis``.
        """
        if parameters is None and self.parameters is None:
            raise ValueError("You must supply parameters to evaluate the model!")

        if parameters is None:
            parameters = np.asarray(self.parameters)
        else:
            parameters = np.asarray(parameters)

        if x is None and basis is None:
            raise ValueError("You must supply either x or basis!")

        if basis is None:
            basis = self.get_basis_terms(x, with_scaler=with_scaler)

        if not isinstance(indices, slice):
            indices = np.array(indices)

            if any(idx >= len(basis) for idx in indices):
                raise ValueError("Cannot use more basis sets than available!")

            if len(parameters) != len(indices):
                parameters = parameters[indices]
        elif len(parameters) != basis.shape[0] and indices == slice(None):
            indices = slice(0, len(parameters))

        return self.data_transform.inverse(x, np.dot(parameters, basis[indices]))

    def fit(
        self,
        xdata: np.ndarray,
        ydata: np.ndarray,
        weights: np.ndarray | float = 1.0,
        **kwargs,
    ) -> "fitting.ModelFit":
        """Create a linear-regression fit object."""
        return self.at(x=xdata).fit(ydata=ydata, weights=weights, **kwargs)


@hickleable
@attrs.define(frozen=True, kw_only=True, slots=False)
class FixedLinearModel(yaml.YAMLObject):
    """
    A base class for a linear model fixed at a certain set of co-ordinates.

    Using this class caches the basis functions at the particular coordinates, and thus
    speeds up the fitting of multiple sets of data at those co-ordinates.

    Parameters
    ----------
    model
        The linear model to evaluate at the co-ordinates
    x
        A set of co-ordinates at which to evaluate the model.
    init_basis
        If the basis functions of the model, evaluated at x, are known already, they
        can be input directly to save computation time.
    """

    yaml_tag: ClassVar[str] = "!Model"

    model: Model = attrs.field()
    x: np.ndarray = attrs.field(converter=np.asarray)
    _init_basis: np.ndarray | None = attrs.field(
        default=None, converter=attrs.converters.optional(np.asarray)
    )

    @classmethod
    def to_yaml(cls, dumper, data):
        """Method to convert to YAML format."""
        return _model_yaml_representer(dumper, data.model)

    @model.validator
    def _model_vld(self, att, val):
        assert isinstance(val, Model | composite.CompositeModel)

    @_init_basis.validator
    def _init_basis_vld(self, att, val):
        if val is None:
            return

        if val.shape[1] != len(self.x):
            raise ValueError("The init_basis values must be the same shape as x.")

    @property
    def n_terms(self):
        """The number of terms/parameters in the model."""
        return self.model.n_terms

    @cached_property
    def basis(self) -> np.ndarray:
        """The (cached) basis functions at default_x.

        Shape ``(n_terms, x)``.
        """
        out = np.zeros((self.model.n_terms, len(self.x)))
        for indx in range(self.model.n_terms):
            if self._init_basis is not None and indx < len(self._init_basis):
                out[indx] = self._init_basis[indx]
            else:
                out[indx] = self.model.get_basis_term_transformed(indx, self.x)

        return out

    def __call__(
        self,
        x: np.ndarray | None = None,
        parameters: Sequence | None = None,
        indices: Sequence | slice = slice(None),
    ) -> np.ndarray:
        """Evaluate the model.

        Parameters
        ----------
        x
            The coordinates at which to evaluate the model (by default, use ``self.x``).
        params
            A list/array of parameters at which to evaluate the model. Will use the
            instance's parameters if available. If using a subset of the basis
            functions, you can pass a subset of parameters.
        indices
            Sequence of parameters indices to use (other parameters are set to zero).

        Returns
        -------
        model
            The model evaluated at the input ``x``.
        """
        return self.model(
            basis=self.basis if x is None else None,
            x=x,
            parameters=parameters,
            indices=indices,
        )

    def fit(
        self,
        ydata: np.ndarray,
        weights: np.ndarray | float = 1.0,
        xdata: np.ndarray | None = None,
        **kwargs,
    ) -> "fitting.ModelFit":
        """Create a linear-regression fit object.

        Parameters
        ----------
        ydata
            The data to fit.
        weights
            The weights to apply to the data.
        xdata
            The co-ordinates at which to fit the data. If not given, use ``self.x``.

        Other Parameters
        ----------------
        All other parameters used to construct the :class:`ModelFit` object. Includes
        ``method`` to specify the lstsq solving method.

        Returns
        -------
        fit
            The :class:`ModelFit` object.
        """
        thing = self.at_x(xdata) if xdata is not None else self
        d = self.model.data_transform.transform(thing.x, ydata)
        return fitting.ModelFit(
            thing,
            ydata=d,
            weights=weights,
            **kwargs,
        )

    def at_x(self, x: np.ndarray) -> Self:
        """Return a new :class:`FixedLinearModel` at given co-ordinates."""
        return attrs.evolve(self, x=x, init_basis=None)

    def with_nterms(self, n_terms: int, parameters: Sequence | None = None) -> Self:
        """Return a new :class:`FixedLinearModel` with given nterms and parameters."""
        init_basis = as_readonly(self.basis[: min(self.model.n_terms, n_terms)])
        model = self.model.with_nterms(n_terms=n_terms, parameters=parameters)
        return attrs.evolve(self, model=model, init_basis=init_basis)

    def with_params(self, parameters: Sequence) -> Self:
        """Return a new :class:`FixedLinearModel` with givne parameters."""
        assert len(parameters) == self.model.n_terms

        init_basis = as_readonly(self.basis)
        model = self.model.with_params(parameters=parameters)
        return attrs.evolve(self, model=model, init_basis=init_basis)

    @property
    def parameters(self) -> np.ndarray | None:
        """The parameters of the model, if set."""
        return self.model.parameters


def get_mdl(model: str | type[Model]) -> type[Model]:
    """Get a linear model class from a string input."""
    if isinstance(model, str):
        return _MODELS[model]
    try:
        if issubclass(model, Model):
            return model
    except TypeError:
        pass

    raise ValueError("model needs to be a string or Model subclass")


def get_mdl_inst(model: str | Model | type[Model], **kwargs) -> Model:
    """Get a model instance from given string input."""
    if isinstance(model, Model):
        return attrs.evolve(model, **kwargs) if kwargs else model
    return get_mdl(model)(**kwargs)


def _model_yaml_constructor(
    loader: yaml.SafeLoader, node: yaml.nodes.MappingNode
) -> Model:
    mapping = loader.construct_mapping(node, deep=True)
    model = get_mdl(mapping.pop("model"))
    if "_transform" in mapping:
        mapping["xtransform"] = mapping.pop("_transform")

    return model(**mapping)


def _model_yaml_representer(
    dumper: yaml.SafeDumper, model: Model
) -> yaml.nodes.MappingNode:
    model_dct = attrs.asdict(model, recurse=False)
    model_dct.update(model=model.__class__.__name__.lower())
    if model_dct["parameters"] is not None:
        model_dct["parameters"] = tuple(float(x) for x in model_dct["parameters"])

    if "_transform" in model_dct:
        del model_dct["_transform"]  # deprecated, use xtransform

    return dumper.represent_mapping("!Model", model_dct)


yaml.FullLoader.add_constructor("!Model", _model_yaml_constructor)
yaml.Loader.add_constructor("!Model", _model_yaml_constructor)
yaml.BaseLoader.add_constructor("!Model", _model_yaml_constructor)


yaml.add_multi_representer(Model, _model_yaml_representer)
Modelable = str | type[Model]
