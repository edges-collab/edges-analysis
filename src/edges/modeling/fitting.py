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
    def n_used(self) -> int:
        """The number of data points that constrain the fit.

        These are the points with finite data and a positive weight. For a full
        weight matrix (where all data must be finite), it is the number of data points.
        """
        if self._has_weight_matrix:
            return int(np.sum(self._mask))
        return int(np.sum(self._mask & (np.asarray(self.weights) > 0)))

    @cached_property
    def degrees_of_freedom(self) -> int:
        """The number of degrees of freedom of the fit.

        This is the number of used data points (see :attr:`n_used`) minus the number
        of fitted parameters.
        """
        return self.n_used - self.model.model.n_terms

    @cached_property
    def fit(self) -> core.FixedLinearModel:
        """A model that has parameters set based on the best fit to this data."""
        mask = self._mask

        w = self._masked_weights

        if self.method in ("lstsq", "qr"):
            basis, ydata = self._solver_inputs()
            pars = _LinearSolver(basis, w, self.method).solve(ydata)
        elif self.method == "qrd-c":
            pars = self._qrd_c(self.model.basis[:, mask], self.ydata[mask], w=w)
        elif self.method == "alan-qrd":
            pars = self._alan_qrd(self.model.basis[:, mask], self.ydata[mask], w=w)

        # Create a new model with the same parameters but specific parameters and xdata.
        return self.model.with_params(parameters=pars)

    def _solver_inputs(self) -> tuple[np.ndarray, np.ndarray]:
        """The basis and data of the used points, as passed to :class:`_LinearSolver`.

        The data are masked to the finite points, except for the 'lstsq' method with
        a full weight matrix (for which all the data must be finite).
        """
        if self.method == "lstsq" and self._has_weight_matrix:
            return self.model.basis, self.ydata

        return _select_columns(self.model.basis, self._mask), self.ydata[self._mask]

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
        npar, _ = basis.shape

        if self._has_weight_matrix:
            wa = np.dot(basis, w)
        else:
            # A uniform weight does not change the solution, so use unit weights.
            # 1D weights scale the columns of the basis, which is identical to (but
            # O(n) rather than O(n^2) in memory and time) multiplying by diag(w).
            # The result is a new C-ordered array (as the matrix product was), so
            # that the dot products below sum in exactly the same order as before
            # (the C code is sensitive to the last bits of the normal equations).
            wa = np.multiply(basis, 1.0 if np.isscalar(w) else w, order="C")

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
        r"""The weighted root-mean-square of the residuals.

        This is :math:`\sqrt{\sum w r^2 / \sum w}` over the used data points (those
        with finite data). For uniform weights it is the plain RMS of the residuals.

        Raises
        ------
        NotImplementedError
            If the weights are a full (inverse-covariance) matrix.
        """
        if self._has_weight_matrix:
            raise NotImplementedError(
                "weighted_rms is not defined for a full weight matrix; use "
                "weighted_chi2 or reduced_weighted_chi2."
            )
        w = self._masked_weights
        sum_w = w * np.sum(self._mask) if np.isscalar(w) else np.sum(w)
        return np.sqrt(self.weighted_chi2 / sum_w)

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


class _LinearSolver:
    """The 'lstsq' and 'qr' weighted least-squares solves of :class:`ModelFit`.

    Everything that depends only on the basis and the weights (the weighted, scaled
    design matrix, or its QR decomposition) is computed on construction, so that
    :meth:`solve` can be called for many data vectors with fixed basis and weights.
    Each solve is bit-for-bit identical to a fresh solve.

    Parameters
    ----------
    basis
        The ``(n_terms, n)`` basis at the used data points.
    w
        The weights of the used data points: a scalar, a 1D array of inverse
        variances, or an ``(n, n)`` inverse-covariance matrix.
    method
        Either 'lstsq' or 'qr' (see :class:`ModelFit`).
    """

    def __init__(
        self,
        basis: np.ndarray,
        w: np.ndarray | float,
        method: Literal["lstsq", "qr"],
    ):
        self.method = method
        if method == "lstsq":
            self._init_lstsq(basis, w)
        elif method == "qr":
            self._init_qr(basis, w)
        else:
            raise ValueError(f"Unsupported method for _LinearSolver: {method}")

    def _init_lstsq(self, van: np.ndarray, w: np.ndarray | float):
        """Prepare the design matrix for the ``np.linalg.lstsq`` based solves.

        Ripped straight outta numpy for speed. Note: this is written purely for
        speed, and is intended to *not* be highly generic. Don't replace this by
        statsmodels or even np.polyfit. They are significantly slower (>4x for
        statsmodels, 1.5x for polyfit). Unlike np.polyfit (whose weights are
        1/sigma), the basis and data are multiplied by ``sqrt(w)``, since ``w`` is
        1/sigma^2. The columns of the design matrix are normalised before solving.

        A full weight matrix ``w = L L^T`` instead whitens the system with ``L^T``
        (generalised least squares), without normalising the columns.
        """
        self._wmask = None
        self._sqrtw = None
        self._lt = None
        if np.isscalar(w):
            # Determine the norms of the design matrix columns.
            self._scl = np.sqrt(np.square(van.T).sum(axis=0))
            self._lhs = van.T / self._scl
        elif np.ndim(w) == 2:
            self._lt = np.linalg.cholesky(w).T
            self._lhs = self._lt @ van.T
        else:
            # Set up the least squares matrices and apply weights. Don't use
            # in-place operations as they can cause problems with NA.
            # Select the points with positive weight (a slice, rather than a copy,
            # of the 1D arrays when all of them are, giving identical results).
            wmask = w > 0
            self._wmask = slice(None) if np.all(wmask) else wmask
            self._sqrtw = np.sqrt(w[self._wmask])
            lhs = _select_columns(van, wmask) * self._sqrtw

            # Determine the norms of the design matrix columns.
            scl = np.sqrt(np.square(lhs).sum(1))
            scl[scl == 0] = 1
            self._scl = scl
            self._lhs = lhs.T / scl

    def _init_qr(self, basis: np.ndarray, w: np.ndarray | float):
        """Prepare the QR decomposition of the whitened basis.

        See: http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/2804/pdf/imm2804.pdf
        """
        self._sqrtw = None
        self._sqrtw_matrix = None
        if np.ndim(w) == 2:
            # sqrt of weight matrix. For a full matrix w = L L^T, use L^T so that
            # |L^T r|^2 = r^T w r (an elementwise sqrt is only valid when w is
            # diagonal).
            self._sqrtw_matrix = np.linalg.cholesky(w).T

            # A "tilde"
            sqrt_wa = np.dot(self._sqrtw_matrix, basis.T)
        elif np.isscalar(w):
            # A uniform weight does not change the solution: use unit weights.
            sqrt_wa = basis.T
        else:
            # Diagonal weights: scale the rows rather than forming diag(sqrt(w)),
            # which is O(n^2) in memory and time (and gives identical results, since
            # the off-diagonal terms only ever add exact zeros).
            self._sqrtw = np.sqrt(w)
            sqrt_wa = self._sqrtw[:, None] * basis.T

        # solving system using 'short' QR decomposition (see R. Butt, Num. Anal.
        # Using MATLAB)
        self._q, self._r = sp.linalg.qr(sqrt_wa, mode="economic")

    def solve(self, y: np.ndarray) -> np.ndarray:
        """Get the best-fit parameters for the data ``y`` (at the used points)."""
        if self.method == "qr":
            if self._sqrtw_matrix is not None:
                w_ydata = np.dot(self._sqrtw_matrix, y)
            elif self._sqrtw is not None:
                w_ydata = self._sqrtw * y
            else:
                w_ydata = y
            return sp.linalg.solve(self._r, np.dot(self._q.T, w_ydata))

        if self._lt is not None:
            return np.linalg.lstsq(self._lhs, self._lt @ y, rcond=None)[0]

        rcond = y.size * np.finfo(y.dtype).eps
        if self._wmask is None:
            return np.linalg.lstsq(self._lhs, y.T, rcond)[0] / self._scl

        rhs = y[self._wmask] * self._sqrtw
        c, _resids, _rank, _s = np.linalg.lstsq(self._lhs, rhs.T, rcond)
        return (c.T / self._scl).T


class _RepeatedFit:
    """Fits of a fixed linear model, with fixed weights, to many data vectors.

    This is equivalent to calling ``model.fit(ydata, weights=weights,
    method=method)`` for each data vector, but the parts of the solve that depend
    only on the basis and the weights are computed once (see
    :class:`_LinearSolver`), and no :class:`ModelFit` is needed to get the best-fit
    parameters or residuals. The results are bit-for-bit identical to the full fits.

    The solve is prepared for the data mask (finite data points) of the first data
    vector; data with a different mask fall back to a full :class:`ModelFit`, as do
    the 'alan-qrd' and 'qrd-c' methods.

    Parameters
    ----------
    model
        The linear model, fixed at the data coordinates.
    weights
        The weights of the data (see :class:`ModelFit`).
    method
        The solving method (see :class:`ModelFit`).
    """

    def __init__(
        self,
        model: core.FixedLinearModel,
        weights: np.ndarray | float = 1.0,
        method: Literal["lstsq", "qr", "alan-qrd", "qrd-c"] = "lstsq",
    ):
        self.model = model
        self.weights = weights
        self.method = method
        self._mask: np.ndarray | None = None
        self._solver: _LinearSolver | None = None
        self._full_data = False

    def _transform(self, ydata: np.ndarray) -> np.ndarray:
        """Apply the model's data transform, as :meth:`FixedLinearModel.fit` does."""
        return self.model.model.data_transform.transform(self.model.x, ydata)

    def _cached_parameters(self, d: np.ndarray) -> np.ndarray | None:
        """Best-fit parameters for the transformed data ``d`` from the cached solve.

        Returns ``None`` if the cached solve does not apply to ``d``.
        """
        if self.method not in ("lstsq", "qr"):
            return None

        mask = np.isfinite(d)
        if self._mask is None:
            # A full fit validates the weights (and data), and gives the inputs of
            # the solve exactly as ModelFit would pass them.
            fit = ModelFit(
                self.model, ydata=d, weights=self.weights, method=self.method
            )
            basis, _ = fit._solver_inputs()
            self._solver = _LinearSolver(basis, fit._masked_weights, self.method)
            self._full_data = self.method == "lstsq" and fit._has_weight_matrix
            self._mask = mask
        elif not np.array_equal(mask, self._mask):
            return None

        return self._solver.solve(d if self._full_data else d[mask])

    def fit(self, ydata: np.ndarray) -> ModelFit:
        """Get the :class:`ModelFit` of the data (as ``model.fit(ydata, ...)``)."""
        d = self._transform(ydata)
        out = ModelFit(self.model, ydata=d, weights=self.weights, method=self.method)
        pars = self._cached_parameters(d)
        if pars is not None:
            # Pre-fill the (cached) best-fit model, so that it is not re-solved.
            out.__dict__["fit"] = self.model.with_params(parameters=pars)
        return out

    def parameters(self, ydata: np.ndarray) -> np.ndarray:
        """Get the best-fit parameters of the data, as an array."""
        pars = self._cached_parameters(self._transform(ydata))
        if pars is None:
            return np.asarray(self.fit(ydata).model_parameters)
        return pars

    def residual(self, ydata: np.ndarray) -> np.ndarray:
        """Get the residuals of the data to the best fit (see ModelFit.residual)."""
        d = self._transform(ydata)
        pars = self._cached_parameters(d)
        if pars is None:
            return self.fit(ydata).residual
        return d - self.model(parameters=pars)


def _select_columns(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Select the columns ``mask`` of a 2D array, i.e. ``arr[:, mask]``.

    The result is identical to ``arr[:, mask]``, including its (Fortran-ordered)
    memory layout, which matters for the summation order of later reductions; but
    when all columns are selected, the copy is made much faster.
    """
    if np.all(mask) and (arr.flags.c_contiguous or arr.flags.f_contiguous):
        return np.asfortranarray(arr)
    return arr[:, mask]


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
