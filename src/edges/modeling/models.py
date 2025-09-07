"""Specific linear models for edges-cal."""

from functools import cached_property
from typing import ClassVar

import attrs
import numpy as np

from ..io.serialization import hickleable
from . import core
from .xtransforms import Log10Transform, LogTransform, ScaleTransform, XTransform


@hickleable
@attrs.define(frozen=True, kw_only=True, slots=False)
class Foreground(core.Model, is_meta=True):
    """
    Base class for Foreground models.

    Parameters
    ----------
    f_center : float
        A "center" or "reference" frequency. Typically models will have their
        co-ordindates divided by this frequency before solving for the
        co-efficients.
    with_cmb : bool
        Whether to add a simple CMB component to the foreground.
    """

    with_cmb: bool = attrs.field(default=False, converter=bool)
    f_center: float = attrs.field(default=75.0, converter=float)
    _transform: XTransform = attrs.field()
    xtransform: XTransform = attrs.field()

    @_transform.default
    def _tr_default(self):
        return ScaleTransform(scale=self.f_center)

    @xtransform.default
    def _xt_default(self):
        return self._transform


@hickleable
@attrs.define(frozen=True, kw_only=True, slots=False)
class PhysicalLin(Foreground):
    """Foreground model using a linearized physical model of the foregrounds."""

    n_terms_max: ClassVar[int] = 5
    default_n_terms: ClassVar[int] = 5
    spectral_index: float = attrs.field(default=-2.5, converter=float)

    def get_basis_term(self, indx: int, x: np.ndarray) -> np.ndarray:
        """Define the basis functions of the model."""
        if indx < 3:
            logy = np.log(x)
            y25 = x**self.spectral_index
            return y25 * logy**indx

        if indx == 3:
            return x ** (self.spectral_index - 2)
        if indx == 4:
            return 1 / (x * x)
        raise ValueError("too many terms supplied!")


@hickleable
@attrs.define(frozen=True, kw_only=True, slots=False)
class PhysicalIono(Foreground):
    """Foreground model using a linearized physical model of the foregrounds."""

    n_terms_max: ClassVar[int] = 5
    default_n_terms: ClassVar[int] = 5
    spectral_index: float = attrs.field(default=-2.5, converter=float)

    def get_basis_term(self, indx: int, x: np.ndarray) -> np.ndarray:
        """Define the basis functions of the model."""
        if indx == 0:
            return x**self.spectral_index
        if indx == 1:
            return x**self.spectral_index * np.log(x)
        if indx == 2:
            return x ** (self.spectral_index - 2)
        if indx == 3:
            return 1 / (x * x)
        if indx == 4:
            return x**self.spectral_index * np.log(x) ** 2

        raise ValueError("too many terms supplied!")


@hickleable
@attrs.define(frozen=True, kw_only=True, slots=False)
class Polynomial(core.Model):
    r"""A polynomial foreground model.

    Parameters
    ----------
    offset : float
        An offset to use for each index in the polynomial model.

    Notes
    -----
    The polynomial model can be written

    .. math:: \sum_{i=0}^{n} c_i x^{i + offset},
    """

    offset: float = attrs.field(default=0, converter=float)
    spacing: float = attrs.field(default=1.0, converter=float)

    def get_basis_term(self, indx: int, x: np.ndarray) -> np.ndarray:
        """Define the basis functions of the model."""
        return x ** (indx * self.spacing + self.offset)


@hickleable
@attrs.define(frozen=True, kw_only=True, slots=False)
class EdgesPoly(Polynomial):
    """
    Polynomial with an offset corresponding to approximate galaxy spectral index.

    Parameters
    ----------
    offset : float
        The offset to use. Default is close to the Galactic spectral index.
    kwargs
        All other arguments are passed through to :class:`Polynomial`.
    """

    offset: float = attrs.field(default=-2.5, converter=float)


@hickleable
@attrs.define(frozen=True, kw_only=True)
class LinLog(Foreground):
    """LinLog foreground mmodel."""

    beta: float = attrs.field(default=-2.5, converter=float)

    @property
    def _poly(self):
        return Polynomial(
            transform=LogTransform(),
            offset=0,
            n_terms=self.n_terms,
            parameters=self.parameters,
        )

    def get_basis_term(self, indx: int, x: np.ndarray) -> np.ndarray:
        """Define the basis functions of the model."""
        term = self._poly.get_basis_term_transformed(indx, x)
        return term * x**self.beta


def LogPoly(**kwargs):  # noqa: N802
    """A factory function for a LogPoly model."""
    return Polynomial(transform=Log10Transform(), offset=0, **kwargs)


@hickleable
@attrs.define(frozen=True, kw_only=True, slots=False)
class Fourier(core.Model):
    """A Fourier-basis model."""

    period: float = attrs.field(default=2 * np.pi, converter=float)

    @cached_property
    def _period_fac(self):
        return 2 * np.pi / self.period

    def get_basis_term(self, indx: int, x: np.ndarray) -> np.ndarray:
        """Define the basis functions of the model."""
        if indx == 0:
            return np.ones_like(x)
        if indx % 2:
            return np.cos(self._period_fac * ((indx + 1) // 2) * x)
        return np.sin(self._period_fac * ((indx + 1) // 2) * x)


@hickleable
@attrs.define(frozen=True, kw_only=True, slots=False)
class FourierDay(core.Model):
    """A Fourier-basis model with period of 24 (hours)."""

    @property
    def _fourier(self):
        return Fourier(period=48.0, n_terms=self.n_terms, parameters=self.parameters)

    def get_basis_term(self, indx: int, x: np.ndarray) -> np.ndarray:
        """Define the basis functions of the model."""
        return self._fourier.get_basis_term(indx, x)
