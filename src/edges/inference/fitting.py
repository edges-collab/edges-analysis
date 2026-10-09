"""Provides extra routines for fitting that are not in yabf."""

from functools import cached_property

import numpy as np
from scipy import linalg, stats
from scipy.optimize import dual_annealing, minimize
from yabf import Component

from edges.modeling import FixedLinearModel
from edges.modeling.data_transforms import IdentityTransform
from edges.modeling.fitting import _RepeatedFit


class SemiLinearFit:
    """A class for performing a fit to sky data composed of a 21cm and FG component.

    In this model, the FG component is assumed to be a linear model, while the
    21cm component is modeled as a non-linear function. For each set of non-linear
    parameters, the linear (FG) parameters are set to their (generalised) weighted
    least-squares best fit and the likelihood is evaluated there. Because the FG basis
    and the noise do not depend on the non-linear parameters, this is equivalent, up
    to a constant, to analytically marginalizing over the linear parameters with a
    flat prior: the likelihood surface (and its optimum) for the non-linear
    parameters is the same. The constant (needed e.g. for Bayesian evidences) is not
    computed; see :class:`~edges.inference.PartialLinearModel` for the full
    marginal likelihood.

    Parameters
    ----------
    fg
        The foreground model -- a linear model fixed to some set of frequency
        coordinates (only the coordinates are fixed, not the parameters).
    eor
        The 21cm model -- a non-linear model that is fit via non-linear optimization.
    spectrum
        The sky data to fit to.
    sigma
        The noise: either a float (a constant standard deviation for all
        frequencies), a 1D array of per-frequency standard deviations with the same
        shape as the spectrum, or a 2D covariance matrix of shape
        ``(nfreq, nfreq)``. With a covariance matrix, the FG fit is a generalised
        least-squares fit and the likelihood is a multivariate normal.

    Raises
    ------
    ValueError
        If ``sigma`` is a 2D array that is not a square, symmetric, positive-definite
        matrix matching the spectrum, or has more than two dimensions.
    NotImplementedError
        If ``sigma`` is a covariance matrix and the FG model has a non-identity data
        transform.

    Notes
    -----
    The parts of the FG fit that depend only on the FG basis and the noise are
    computed on the first fit and re-used, so ``fg`` and ``sigma`` should not be
    changed after the first fit.
    """

    def __init__(
        self,
        fg: FixedLinearModel,
        eor: Component,
        spectrum: np.ndarray,
        sigma: np.ndarray | float,
    ):
        """Perform a quick fit to data with a sum of linear and non-linear models.

        Useful for fitting foregrounds and EoR at the same time, where the EoR model is
        not linear, but the foreground model is.
        """
        self.fg = fg
        self.eor = eor
        self.spectrum = spectrum
        self.sigma = sigma

        ndim = np.ndim(sigma)
        if ndim > 2:
            raise ValueError(
                "sigma must be a 1D array, a float, or a 2D covariance matrix; got an "
                f"array with {ndim} dimensions."
            )
        self._is_cov = ndim == 2
        if self._is_cov:
            self._setup_covariance(np.asarray(sigma, dtype=float))

    def _setup_covariance(self, cov: np.ndarray):
        """Validate the covariance and precompute its inverse (the GLS weights)."""
        nfreq = np.shape(self.spectrum)[-1]
        if cov.shape != (nfreq, nfreq):
            raise ValueError(
                f"A covariance sigma must have shape ({nfreq}, {nfreq}) to match the "
                f"spectrum; got {cov.shape}."
            )
        if not np.allclose(cov, cov.T):
            raise ValueError("A covariance sigma must be symmetric.")
        if not isinstance(self.fg.model.data_transform, IdentityTransform):
            raise NotImplementedError(
                "A covariance sigma is only supported for FG models without a data "
                "transform."
            )
        try:
            cov_cho = linalg.cho_factor(cov, lower=True)
        except linalg.LinAlgError as e:
            raise ValueError("A covariance sigma must be positive definite.") from e

        # The FG fit is a generalised least-squares fit with weights = C^-1.
        cinv = linalg.cho_solve(cov_cho, np.eye(nfreq))
        self._cov_inverse = (cinv + cinv.T) / 2
        self._mvn = stats.multivariate_normal(mean=np.zeros(nfreq), cov=cov)

    def get_eor(self, p):
        """Compute the EOR model given EOR parameters p."""
        return self.eor(params=p)["eor_spectrum"]

    @cached_property
    def _fg_fitter(self) -> _RepeatedFit:
        """The FG fits, with the parts that depend only on the basis/noise cached."""
        if self._is_cov:
            return _RepeatedFit(self.fg, weights=self._cov_inverse, method="qr")
        return _RepeatedFit(
            self.fg,
            weights=1 / self.sigma**2 if hasattr(self.sigma, "__len__") else 1.0,
        )

    def fg_fit(self, p):
        """Compute the best FG fit, given EOR parameters p."""
        return self._fg_fitter.fit(self.spectrum - self.get_eor(p))

    def fg_params(self, p):
        """Compute the best-fit FG parameters, given EoR parameters p."""
        return tuple(self._fg_fitter.parameters(self.spectrum - self.get_eor(p)))

    def get_resid(self, p):
        """Comptue the residual for given parameters p."""
        return self._fg_fitter.residual(self.spectrum - self.get_eor(p))

    def neg_lk(self, p):
        """Comptue the negative log-likelihood given parameters p."""
        resid = self.get_resid(p)
        if self._is_cov:
            return -self._mvn.logpdf(resid)
        norm_obj = stats.norm(loc=0, scale=self.sigma)
        return -np.sum(norm_obj.logpdf(resid))

    def __call__(self, dual_annealing_kw=None, **kwargs):
        """Perform the fit to the data."""
        if dual_annealing_kw is None:
            return minimize(
                self.neg_lk,
                x0=np.array([apar.fiducial for apar in self.eor.child_active_params]),
                bounds=[(apar.min, apar.max) for apar in self.eor.child_active_params],
                **kwargs,
            )
        return dual_annealing(
            self.neg_lk,
            bounds=[(apar.min, apar.max) for apar in self.eor.child_active_params],
            x0=np.array([apar.fiducial for apar in self.eor.child_active_params]),
            local_search_options=kwargs,
            **dual_annealing_kw,
        )
