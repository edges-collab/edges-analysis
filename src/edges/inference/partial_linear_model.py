"""A likelihood where some of the parameters are linear and pre-marginalized."""

import logging
from collections.abc import Callable
from functools import cached_property

import attrs
import numpy as np
from scipy import stats
from yabf import Likelihood
from yabf.chi2 import Chi2

from ..modeling import Model

logger = logging.getLogger(__name__)


@attrs.define(frozen=True, kw_only=True)
class PartialLinearModel(Chi2, Likelihood):
    r"""
    A likelihood where some of the parameters are linear and pre-marginalized.

    Parameters
    ----------
    linear_model
        A linear model containing all the terms that are linear.
    variance_func
        A callable function that takes two arguments: ``ctx`` and ``data``, and returns
        an array of model variance. If not provided, the input data must have a key
        called `"data_variance"` that provides static variance (i.e. the :math`\Sigma`
        in the derivation in the Notes).
    data_func
        A function that has the same signature as ``variance_func``, but returns data
        (i.e. the :math:`d` in the derivation). This might be dependent on non-linear
        parameters (not not the linear ones!). If not provided, the input data must have
        a key called ``"data"``.
    basis_func
        It is not recommended to provide this, but if provided it should be a function
        that takes the linear basis, context and data, and returns a new linear model,
        effectively altering the linear basis functions based on the nonlinear
        parameters.

    Notes
    -----
    The general idea is laid out in Monsalve et al. 2018
    (or https://arxiv.org/pdf/0809.0791.pdf). In this class, the variables are typically
    named the same as in Monsalve et al. 2018 (eg. Q, C, V, Sigma).
    However, note that M2018 misses some fairly significant simplifications.

    Eq. 15 of M18 is

    .. math:: d_\star^T \Sigma^{-1} d_\start - d_\star^T \Sigma^{-1} A
              (A^T \Sigma^{-1} A)^{-1} A^T \Sigma^{-1} d_\star

    where :math:`d_\star`  is the residual of the linear model:
    :math:`d_star = d - A\hat{\theta}`
    (note that we're omitting the nonlinear 21cm model from that paper here because
    it's just absorbed into :math:`d`). Note that part of the second term is just the
    "hat" matrix from weighted least-squares, i.e.

    .. math:: H = A (A^T \Sigma^{-1} A)^{-1} A^T \Sigma^{-1}

    which when applied to a data vector, returns the maximum-likelihood model of the
    data.

    Thus we can re-write

    .. math:: d_\star^T \Sigma^{-1} d_\start - d_\star^T\Sigma^{-1}H(d - A \hat{\theta})

    But :math:`A\hat{\theta}` is equivalent to `H d` (i.e. both produce the maximum
    likelihood model for the data), so we have

    .. math:: d_\star^T \Sigma^{-1} d_\start - d_\star^T \Sigma^{-1} H (d - Hd).

    But the `H` matrix is idempotent, so :math:`Hd - HHd = Hd - Hd = 0`. So we are left
    with the first term only.
    """

    linear_model: Model = attrs.field()
    variance_func: Callable | None = attrs.field(default=None)
    data_func: Callable | None = attrs.field(default=None)
    basis_func: Callable | None = attrs.field(default=None)
    subtract_fiducial: bool = attrs.field(default=False)
    verbose: bool = attrs.field(default=False)

    def _reduce(self, ctx, **params):
        if self.variance_func is None:
            var = self.data["data_variance"]
        else:
            var = self.variance_func(ctx, self.data)

        data = (
            self.data["data"]
            if self.data_func is None
            else self.data_func(ctx, self.data)
        )

        linear_model = (
            self.linear_model
            if self.basis_func is None
            else self.basis_func(self.linear_model, ctx, self.data)
        )

        wght = 1.0 if np.all(var == 0) else 1 / var

        linear_fit = linear_model.fit(ydata=data, weights=wght)
        return linear_fit, data, var

    @cached_property
    def Q(self):  # noqa: N802
        """The precision matrix of the data marginalized over the linear model."""
        if self.basis_func is not None or self.variance_func is not None:
            raise AttributeError("Q is not static in this instance!")
        return (self.linear_model.basis / self.data["data_variance"]).dot(
            self.linear_model.basis.T
        )

    @cached_property
    def logdetCinv(self) -> float | None:  # noqa: N802
        """A derived quantity, the log-determinant of the inverse of the covariance."""
        if np.all(self.data["data_variance"] == 0):
            return 0.0

        try:
            Cinv = self.Q
            return np.log(np.linalg.det(Cinv))
        except AttributeError:
            return None

    @cached_property
    def sigma_plus_v_inverse(self):
        """The inverse of the sum of the data variance and the foreground variance."""
        if self.basis_func is not None or self.variance_func is not None:
            raise AttributeError("V is not static in this instance!")
        A = self.linear_model.basis
        var = self.data["data_variance"]
        Sig = np.diag(var)
        SigInv = np.diag(1 / var)
        C = np.linalg.inv(self.Q)
        SigFG = A.T.dot(C.dot(A))
        V = np.linalg.inv(np.linalg.inv(SigFG) - SigInv)
        return np.linalg.inv(Sig + V)

    @cached_property
    def fiducial_lnl(self):
        """The log-likelihood at the fiducial parameters."""
        return attrs.evolve(self, subtract_fiducial=False, verbose=False)()[0]

    def logdet_cinv(self, model, ctx, **params) -> float:
        """A derived quantity, the log-determinant of the inverse of C."""
        if self.logdetCinv is None:
            fit = model[0]
            h = fit.hessian
            return np.log(np.linalg.det(h))
        return self.logdetCinv

    def logdet_sig(self, model, ctx, **params):
        """Compute the log-determinant of the covariance matrix."""
        var = model[-1]

        if not hasattr(var, "__len__"):
            var = var * np.ones(len(model[1]))
        elif np.all(var == 0):
            var = np.ones_like(var)

        var = var[~np.isinf(var)]

        return np.sum(np.log(var)) if self.variance_func is not None else 0

    def rms(self, model, ctx, **params):
        """Compute the RMS of the residuals weighted by the variance."""
        fit, data, var = model

        if not hasattr(var, "__len__"):
            var = var * np.ones(len(data))
        elif np.all(var == 0):
            var = np.ones_like(var)

        mask = ~np.isinf(var)
        data = data[mask]
        resid = fit.residual[mask]
        var = var[mask]

        return np.nansum(resid**2 / var)

    def lnl(self, model, **params):
        """Compute the log-likelihood value for a given model."""
        # Ensure we don't use flagged channels
        logdetSig = self.logdet_sig(model, None, **params)
        logdetCinv = self.logdet_cinv(model, None, **params)
        rms = self.rms(model, None, **params)

        lnl = -0.5 * (logdetSig + logdetCinv + rms)

        if np.isnan(lnl):
            lnl = -np.inf

        if self.subtract_fiducial:
            lnl -= self.fiducial_lnl

        if np.isnan(lnl) or np.isinf(lnl):
            logger.warning(f"Got bad log-likelihood: {lnl} for params: {params}")

        return lnl

    def get_unmarginalized_lnl(self, linear_params, nonlinear_params):
        """Compute the unmarginalized log-likelihood for given parameters."""
        ctx = self.get_ctx(params=nonlinear_params)

        var = (
            self.data["data_variance"]
            if self.variance_func is None
            else self.variance_func(ctx, self.data)
        )

        data = (
            self.data["data"]
            if self.data_func is None
            else self.data_func(ctx, self.data)
        )

        linear_model = (
            self.linear_model
            if self.basis_func is None
            else self.basis_func(self.linear_model, ctx, self.data)
        )

        linear = linear_model(parameters=linear_params)

        resid = data - linear
        nm = stats.norm(loc=0, scale=np.sqrt(var))
        return np.sum(nm.logpdf(resid))
