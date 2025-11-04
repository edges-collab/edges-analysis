"""Fitting routines for models."""

from copy import copy
from ctypes import CDLL, POINTER, c_double, c_int, c_longdouble
from functools import cached_property
from pathlib import Path
from typing import Literal

import attrs
import numpy as np
import scipy as sp

from ..io.serialization import hickleable
from . import core


@hickleable
@attrs.define(frozen=True, slots=False)
class ModelFit:
    """A class representing a fit of model to data.

    Parameters
    ----------
    model
        The evaluatable model to fit to the data.
    ydata
        The values of the measured data.
    weights
        The weight of the measured data at each point. This corresponds to the
        *variance* of the measurement (not the standard deviation). This is
        appropriate if the weights represent the number of measurements going into
        each piece of data.
    method
        The method to solve the linear least squares problem. This can be either
        'lstsq', 'qr' or 'alan-qrd'. The 'leastsq' method uses the np.linalg.lstsq
        function, while the 'qr' method uses the np.linalg.solve function after
        scipy.linalg.qr. The 'alan-qr' method is a python-port of the QR decomposition
        algorithm found in Alan's C Codebase.

    Raises
    ------
    ValueError
        If model_type is not str, or a subclass of :class:`Model`.
    """

    model: core.FixedLinearModel = attrs.field()
    ydata: np.ndarray = attrs.field()
    weights: np.ndarray | float = attrs.field(
        default=1.0, validator=attrs.validators.instance_of((np.ndarray, float))
    )
    method: Literal["lstsq", "qr", "alan-qrd"] = attrs.field(
        default="lstsq",
        validator=attrs.validators.in_(["lstsq", "qr", "alan-qrd", "qrd-c"]),
    )

    @ydata.validator
    def _ydata_vld(self, att, val):
        assert val.shape == self.model.x.shape

    @weights.validator
    def _weights_vld(self, att, val):
        if isinstance(val, np.ndarray):
            assert val.shape == self.model.x.shape

    @cached_property
    def degrees_of_freedom(self) -> int:
        """The number of degrees of freedom of the fit."""
        return self.model.x.size - self.model.model.n_terms - 1

    @cached_property
    def fit(self) -> core.FixedLinearModel:
        """A model that has parameters set based on the best fit to this data."""
        mask = np.isfinite(self.ydata)

        w = self.weights if np.isscalar(self.weights) else self.weights[mask]

        if self.method == "lstsq":
            if np.isscalar(self.weights):
                pars = self._ls(self.model.basis[:, mask], self.ydata[mask])
            else:
                pars = self._wls(self.model.basis[:, mask], self.ydata[mask], w=w)
        elif self.method == "qr":
            pars = self._qr(self.model.basis[:, mask], self.ydata[mask], w=w)
        elif self.method == "qrd-c":
            pars = self._qrd_c(self.model.basis[:, mask], self.ydata[mask], w=w)
        elif self.method == "alan-qrd":
            pars = self._alan_qrd(self.model.basis[:, mask], self.ydata[mask], w=w)

        # Create a new model with the same parameters but specific parameters and xdata.
        return self.model.with_params(parameters=pars)

    def _qr(self, basis: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Solve a linear system using QR decomposition.

        Here the system is defined as A*theta = y, where A is an (n, m) matrix of basis
        vectors, y is an (n,) vector of data, and theta is an (m,) vector of parameters.

        See: http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/2804/pdf/imm2804.pdf
        """
        if np.isscalar(w):
            w = np.eye(len(y))
        elif np.ndim(w) == 1:
            w = np.diag(w)

        # sqrt of weight matrix
        sqrtw = np.sqrt(w)

        # A and ydata "tilde"
        sqrt_wa = np.dot(sqrtw, basis.T)
        w_ydata = np.dot(sqrtw, y)

        # solving system using 'short' QR decomposition (see R. Butt, Num. Anal.
        # Using MATLAB)
        q, r = sp.linalg.qr(sqrt_wa, mode="economic")
        return sp.linalg.solve(r, np.dot(q.T, w_ydata))

    def _alan_qrd(self, basis: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Solve a linear system using QR decomposition implemented in C.

        This solves the system in the same way as Alan Roger's original C-code that was
        used for Bowman+2018. See
        https://github.com/edges-collab/alans-pipeline/blob/
        0e41156ddc7aaa3dd4b37cd1ee1ada971e68d728/src/edges2k.c#L2735
        for the C-Code.

        This is the same as qrd-c *except* that the preparation of the A and b matrices
        is done in python rather than C. This actually makes a difference because
        both A and b are dot-products, and the sum in numpy uses pairwise summation
        which is more accurate than a naive accumulation as would be done in C.
        """
        if np.isscalar(w):
            w = np.eye(len(y))
        elif np.ndim(w) == 1:
            w = np.diag(w)

        npar, _ = basis.shape

        wa = np.dot(basis, w)

        bbrr = np.dot(wa, y)
        aarr = np.dot(wa, basis.T)
        assert bbrr.shape == (npar,)
        assert aarr.shape == (npar, npar)
        _, bb = _c_qrd(aarr, bbrr)
        return bb

    def _qrd_c(self, basis: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Solve a linear system using QR decomposition implemented in C.

        This solves the system in the same way as Alan Roger's original C-code that was
        used for Bowman+2018. See
        https://github.com/edges-collab/alans-pipeline/blob/
        0e41156ddc7aaa3dd4b37cd1ee1ada971e68d728/src/edges2k.c#L2735
        for the C-Code.
        """
        if np.isscalar(w):
            w = np.ones(len(y))
        elif np.ndim(w) != 1:
            raise ValueError("Weights must be a 1D array or scalar for qrd_c method.")

        # solve the system
        a, b = _get_a_and_b(basis, y, w)
        _, bb = _c_qrd(a, b)
        return bb

    def _wls(self, van, y, w):
        """Ripped straight outta numpy for speed.

        Note: this function is written purely for speed, and is intended to *not*
        be highly generic. Don't replace this by statsmodels or even np.polyfit. They
        are significantly slower (>4x for statsmodels, 1.5x for polyfit).
        """
        # set up the least squares matrices and apply weights.
        # Don't use inplace operations as they
        # can cause problems with NA.
        mask = w > 0

        lhs = van[:, mask] * w[mask]
        rhs = y[mask] * w[mask]

        rcond = y.size * np.finfo(y.dtype).eps

        # Determine the norms of the design matrix columns.
        scl = np.sqrt(np.square(lhs).sum(1))
        scl[scl == 0] = 1

        # Solve the least squares problem.
        c, _resids, _rank, _s = np.linalg.lstsq((lhs.T / scl), rhs.T, rcond)
        return (c.T / scl).T

    def _ls(self, van, y):
        """Ripped straight outta numpy for speed.

        Note: this function is written purely for speed, and is intended to *not*
        be highly generic. Don't replace this by statsmodels or even np.polyfit. They
        are significantly slower (>4x for statsmodels, 1.5x for polyfit).
        """
        rcond = y.size * np.finfo(y.dtype).eps

        # Determine the norms of the design matrix columns.
        scl = np.sqrt(np.square(van.T).sum(axis=0))

        # Solve the least squares problem.
        return np.linalg.lstsq((van.T / scl), y.T, rcond)[0] / scl

    @cached_property
    def model_parameters(self):
        """The best-fit model parameters."""
        # Parameters need to be copied into this object, otherwise a new fit on the
        # parent model will change the model_parameters of this fit!
        return copy(self.fit.model.parameters)

    def evaluate(self, x: np.ndarray | None = None) -> np.ndarray:
        """Evaluate the best-fit model.

        Parameters
        ----------
        x : np.ndarray, optional
            The co-ordinates at which to evaluate the model. By default, use the input
            data co-ordinates.

        Returns
        -------
        y : np.ndarray
            The best-fit model evaluated at ``x``.
        """
        return self.fit(x=x)

    @cached_property
    def residual(self) -> np.ndarray:
        """Residuals of data to model."""
        return self.ydata - self.evaluate()

    @cached_property
    def weighted_chi2(self) -> float:
        """The chi^2 of the weighted fit."""
        return np.dot(self.residual.T, self.weights * self.residual)

    @cached_property
    def reduced_weighted_chi2(self) -> float:
        """The weighted chi^2 divided by the degrees of freedom."""
        return (1 / self.degrees_of_freedom) * self.weighted_chi2

    @cached_property
    def weighted_rms(self) -> float:
        """The weighted root-mean-square of the residuals."""
        return np.sqrt(self.weighted_chi2) / np.sum(self.weights)

    @cached_property
    def hessian(self):
        """The Hessian matrix of the linear parameters."""
        b = self.model.basis
        w = self.weights
        return (b * w).dot(b.T)

    @cached_property
    def parameter_covariance(self) -> np.ndarray:
        """The Covariance matrix of the parameters."""
        return np.linalg.inv(self.hessian)

    @cached_property
    def parameter_correlation(self) -> np.ndarray:
        """The correlation matrix of the parameters."""
        cov = self.parameter_covariance
        std = np.sqrt(np.diag(cov))
        return cov / std[:, None] / std[None, :]

    def get_sample(self, size: int | tuple[int] = 1):
        """Generate a random sample from the posterior distribution."""
        rng = np.random.default_rng()
        return rng.multivariate_normal(
            mean=self.model_parameters, cov=self.parameter_covariance, size=size
        )


def _c_qrd(a: np.ndarray, b: np.ndarray):
    """Solve a linear system using QR decomposition.

    This solves the system in the same way as Alan Roger's original C-code that was
    used for Bowman+2018.
    """
    dllpath = next(iter(Path(__file__).parent.glob("cqrd.*.so")))
    dll = CDLL(dllpath)
    qrd = dll.qrd
    qrd.argtypes = [POINTER(c_longdouble), c_int, POINTER(c_double)]
    qrd.restype = None

    aa = a.copy().astype(np.longdouble)
    bb = b.copy().astype(np.float64)

    qrd(
        aa.ctypes.data_as(POINTER(c_longdouble)),
        b.shape[0],
        bb.ctypes.data_as(POINTER(c_double)),
    )

    return aa, bb


def _get_a_and_b(
    basis: np.ndarray,
    data: np.ndarray,
    weights: np.ndarray,
):
    dllpath = next(iter(Path(__file__).parent.glob("cqrd.*.so")))
    dll = CDLL(dllpath)
    lfit = dll.get_a_and_b

    lfit.argtypes = [
        c_int,
        c_int,
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_longdouble),
        POINTER(c_double),
    ]
    lfit.restype = None

    a = np.zeros(basis.shape[0] * basis.shape[0], dtype=np.longdouble)
    b = np.zeros(basis.shape[0], dtype=np.float64)

    basis = basis.astype(np.float64)
    data = data.astype(np.float64)
    weights = weights.astype(np.float64)

    lfit(
        basis.shape[0],
        basis.shape[1],
        basis.ctypes.data_as(POINTER(c_double)),
        data.ctypes.data_as(POINTER(c_double)),
        weights.ctypes.data_as(POINTER(c_double)),
        a.ctypes.data_as(POINTER(c_longdouble)),
        b.ctypes.data_as(POINTER(c_double)),
    )
    return a, b
