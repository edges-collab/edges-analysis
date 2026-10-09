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


def _weights_converter(val: np.ndarray | float | None) -> np.ndarray | float:
    """Convert input weights to either a float or a float array.

    ``None`` is interpreted as uniform (unit) weights.
    """
    if val is None:
        return 1.0
    if np.ndim(val) == 0:
        return float(val)
    return np.asarray(val, dtype=float)


@hickleable
@attrs.define(frozen=True, slots=False)
class ModelFit:
    r"""A class representing a fit of model to data.

    Parameters
    ----------
    model
        The evaluatable model to fit to the data.
    ydata
        The values of the measured data. Non-finite values (NaN or inf) are masked:
        they are excluded from the fit and from the chi^2, RMS and Hessian (and
        thus the parameter covariance). The :attr:`residual` is NaN at those points.
    weights
        The weight of the measured data at each point, either a scalar or an array
        with the same shape as ``ydata``. The weights are the *inverse variance*
        (:math:`1/\sigma^2`) of each measurement, which is also appropriate if the
        weights represent the number of measurements going into each piece of data.
        ``None`` means uniform weights. Weights must be finite and non-negative; a
        weight of zero excludes the point from the fit.

        For correlated noise, ``weights`` may instead be an ``(n, n)`` matrix: the
        *inverse covariance* of the data. The fit is then a generalised least-squares
        fit, minimising :math:`r^T W r`, and the chi^2 and Hessian use the full
        matrix. The matrix must be symmetric and positive definite, all the data must
        be finite, and the 'qrd-c' method is not supported.

        All methods minimise :math:`\sum w r^2` (or :math:`r^T W r`), consistent with
        the Hessian, parameter covariance and chi^2. Note that this differs from the
        convention of :func:`numpy.polyfit`, whose ``w`` is :math:`1/\sigma`.
    method
        The method to solve the linear least squares problem. This can be 'lstsq',
        'qr', 'alan-qrd' or 'qrd-c'. The 'lstsq' method uses the np.linalg.lstsq
        function, while the 'qr' method uses the np.linalg.solve function after
        scipy.linalg.qr. The 'alan-qrd' method solves the normal equations with the
        QR decomposition from Alan's C codebase (building the normal equations in
        Python), while 'qrd-c' does the same but builds the normal equations in C
        as well.

    Raises
    ------
    ValueError
        If the weights are not finite and non-negative.
    """

    model: core.FixedLinearModel = attrs.field()
    ydata: np.ndarray = attrs.field()
    weights: np.ndarray | float = attrs.field(
        default=1.0,
        converter=_weights_converter,
        validator=attrs.validators.instance_of((np.ndarray, float)),
    )
    method: Literal["lstsq", "qr", "alan-qrd", "qrd-c"] = attrs.field(
        default="lstsq",
        validator=attrs.validators.in_(["lstsq", "qr", "alan-qrd", "qrd-c"]),
    )

    @ydata.validator
    def _ydata_vld(self, att, val):
        assert val.shape == self.model.x.shape

    @weights.validator
    def _weights_vld(self, att, val):
        if not np.all(np.isfinite(val)):
            raise ValueError(
                "Weights must be finite (got NaN or inf). To exclude a data point, "
                "give it a weight of zero (or set its data to NaN)."
            )

        ndim = np.ndim(val)
        if ndim == 2:
            self._validate_weight_matrix(val)
        elif ndim > 2:
            raise ValueError(
                "Weights must be a scalar, a 1D array or a 2D (inverse-covariance) "
                f"matrix; got an array with {ndim} dimensions."
            )
        else:
            if isinstance(val, np.ndarray):
                assert val.shape == self.model.x.shape
            if np.any(np.asarray(val) < 0):
                raise ValueError("Weights must be non-negative.")

    def _validate_weight_matrix(self, val: np.ndarray):
        """Check that a 2D weight matrix is a valid inverse covariance for the data."""
        n = self.model.x.size
        if val.shape != (n, n):
            raise ValueError(
                f"A weight matrix must have shape ({n}, {n}); got {val.shape}."
            )
        if not np.allclose(val, val.T):
            raise ValueError("A weight matrix must be symmetric.")
        try:
            np.linalg.cholesky(val)
        except np.linalg.LinAlgError as e:
            raise ValueError("A weight matrix must be positive definite.") from e
        if not np.all(np.isfinite(self.ydata)):
            raise ValueError(
                "A weight matrix cannot be used with non-finite data: masking points "
                "requires the covariance of the remaining data. Remove the points (and "
                "the corresponding rows/columns of the covariance) before inverting."
            )

    @property
    def _has_weight_matrix(self) -> bool:
        """Whether the weights are a full (inverse-covariance) matrix."""
        return np.ndim(self.weights) == 2

    @cached_property
    def _mask(self) -> np.ndarray:
        """Boolean mask of the data points that are used (those with finite data)."""
        return np.isfinite(self.ydata)

    def _apply_mask(self, arr: np.ndarray | float) -> np.ndarray | float:
        """Select the used data points along the last axis of ``arr``.

        Scalars are returned unchanged, as are arrays when all data are used (so that
        results are bit-for-bit identical to the unmasked calculation).
        """
        if np.isscalar(arr) or np.all(self._mask):
            return arr
        return arr[..., self._mask]

    @cached_property
    def _masked_weights(self) -> np.ndarray | float:
        """The weights of the used data points (or the scalar weight)."""
        return self._apply_mask(self.weights)

    @cached_property
    def degrees_of_freedom(self) -> int:
        """The number of degrees of freedom of the fit."""
        return self.model.x.size - self.model.model.n_terms - 1

    @cached_property
    def fit(self) -> core.FixedLinearModel:
        """A model that has parameters set based on the best fit to this data."""
        mask = self._mask

        w = self._masked_weights

        if self.method == "lstsq":
            if np.isscalar(self.weights):
                pars = self._ls(self.model.basis[:, mask], self.ydata[mask])
            elif self._has_weight_matrix:
                pars = self._gls_lstsq(self.model.basis, self.ydata, w)
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

        # sqrt of weight matrix. For a full matrix w = L L^T, use L^T so that
        # |L^T r|^2 = r^T w r (an elementwise sqrt is only valid when w is diagonal).
        sqrtw = np.linalg.cholesky(w).T if self._has_weight_matrix else np.sqrt(w)

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

    def _gls_lstsq(self, van, y, w):
        """Generalised least squares with a full weight (inverse-covariance) matrix.

        The system is whitened with the Cholesky factor of ``w`` (``w = L L^T``) and
        solved with ``np.linalg.lstsq``.
        """
        lt = np.linalg.cholesky(w).T
        return np.linalg.lstsq(lt @ van.T, lt @ y, rcond=None)[0]

    def _wls(self, van, y, w):
        """Weighted least squares with inverse-variance weights, minimising sum(w r^2).

        Ripped straight outta numpy for speed. Note: this function is written purely
        for speed, and is intended to *not* be highly generic. Don't replace this by
        statsmodels or even np.polyfit. They are significantly slower (>4x for
        statsmodels, 1.5x for polyfit). Unlike np.polyfit (whose weights are
        1/sigma), the basis and data are multiplied by ``sqrt(w)``, since ``w`` is
        1/sigma^2.
        """
        # set up the least squares matrices and apply weights.
        # Don't use inplace operations as they
        # can cause problems with NA.
        mask = w > 0
        sqrtw = np.sqrt(w[mask])

        lhs = van[:, mask] * sqrtw
        rhs = y[mask] * sqrtw

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
        """Residuals of data to model (NaN where the data is not finite)."""
        return self.ydata - self.evaluate()

    @cached_property
    def weighted_chi2(self) -> float:
        """The chi^2 of the weighted fit (over the finite data points)."""
        resid = self._apply_mask(self.residual)
        if self._has_weight_matrix:
            return resid @ self.weights @ resid
        return np.dot(resid.T, self._masked_weights * resid)

    @cached_property
    def reduced_weighted_chi2(self) -> float:
        """The weighted chi^2 divided by the degrees of freedom."""
        return (1 / self.degrees_of_freedom) * self.weighted_chi2

    @cached_property
    def weighted_rms(self) -> float:
        """The weighted root-mean-square of the residuals."""
        if self._has_weight_matrix:
            raise NotImplementedError(
                "weighted_rms is not defined for a full weight matrix; use "
                "weighted_chi2 or reduced_weighted_chi2."
            )
        return np.sqrt(self.weighted_chi2) / np.sum(self._masked_weights)

    @cached_property
    def hessian(self):
        """The Hessian matrix of the linear parameters (over the finite data)."""
        b = self._apply_mask(self.model.basis)
        w = self._masked_weights
        if self._has_weight_matrix:
            return b @ w @ b.T
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
