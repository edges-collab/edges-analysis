"""Provides extra routines for fitting that are not in yabf."""

import numpy as np
from scipy import stats
from scipy.optimize import dual_annealing, minimize
from yabf import Component

from edges.modeling import FixedLinearModel


class SemiLinearFit:
    """A class for performing a fit to sky data composed of a 21cm and FG component.

    In this model, the FG component is assumed to be a linear model, while the
    21cm component is modeled as a non-linear function. The linear component is
    fixed via analytic marginalization.

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
        Either a 1D array with the same shape as the spectrum, or a float indicating
        a constant noise level for all frequencies.

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

    def get_eor(self, p):
        """Compute the EOR model given EOR parameters p."""
        return self.eor(params=p)["eor_spectrum"]

    def fg_fit(self, p):
        """Compute the best FG fit, given EOR parameters p."""
        eor = self.get_eor(p)
        resid = self.spectrum - eor
        return self.fg.fit(
            ydata=resid,
            weights=1 / self.sigma**2 if hasattr(self.sigma, "__len__") else 1.0,
        )

    def fg_params(self, p):
        """Compute the best-fit FG parameters, given EoR parameters p."""
        return self.fg_fit(p).model_parameters

    def get_resid(self, p):
        """Comptue the residual for given parameters p."""
        return self.fg_fit(p).residual

    def neg_lk(self, p):
        """Comptue the negative log-likelihood given parameters p."""
        resid = self.get_resid(p)
        if hasattr(self.sigma, "ndim") and self.sigma.ndim == 2:
            norm_obj = stats.multivariate_normal(
                mean=np.zeros_like(resid), cov=self.sigma
            )
        else:
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
