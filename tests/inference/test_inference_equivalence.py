"""Equivalence of the cached likelihood code paths with straightforward ones.

Each test compares the library implementation with a simple reference
implementation (the original algorithm) on realistic random inputs.
"""

import numpy as np
import pytest
from scipy import stats

from edges import modeling as mdl
from edges.inference import FlattenedGaussian, LinearFG, SemiLinearFit


def _slf_setup(n: int, sigma_kind: str, nan: bool = False):
    rng = np.random.default_rng(11)
    freqs = np.linspace(50, 100, n)
    fg = mdl.LinLog(n_terms=5).at(x=freqs)
    eor = FlattenedGaussian(freqs=freqs, params=("amp", "w", "tau", "nu0"))
    data = (
        fg(parameters=[2000, 10, -10, 5, -5])
        + eor()["eor_spectrum"]
        + rng.normal(scale=0.01, size=n)
    )
    if nan:
        data[3::29] = np.nan
    sig = rng.uniform(0.005, 0.02, size=n)
    if sigma_kind == "scalar":
        sigma = 0.01
    elif sigma_kind == "1D":
        sigma = sig
    else:
        sigma = np.diag(sig**2) + 1e-6 * np.exp(
            -(np.subtract.outer(freqs, freqs) ** 2) / 4
        )
    p0 = np.array([a.fiducial for a in eor.child_active_params])
    return fg, eor, data, sigma, [p0, p0 * 1.1, p0 * 0.93]


def _ref_fg_fit(slf: SemiLinearFit, p):
    """The FG fit, done from scratch (the original algorithm)."""
    resid = slf.spectrum - slf.get_eor(p)
    if np.ndim(slf.sigma) == 2:
        return slf.fg.fit(ydata=resid, weights=slf._cov_inverse, method="qr")
    return slf.fg.fit(
        ydata=resid,
        weights=1 / slf.sigma**2 if hasattr(slf.sigma, "__len__") else 1.0,
    )


@pytest.mark.parametrize("sigma_kind", ["scalar", "1D", "cov"])
@pytest.mark.parametrize("nan", [False, True])
def test_semi_linear_fit_cached_fg_fit_is_bit_identical(sigma_kind: str, nan: bool):
    """The cached FG solve gives exactly the parameters/residuals of a full fit."""
    if nan and sigma_kind == "cov":
        pytest.skip("A covariance needs finite data.")
    fg, eor, data, sigma, ps = _slf_setup(150, sigma_kind, nan)
    slf = SemiLinearFit(fg=fg, eor=eor, spectrum=data, sigma=sigma)
    for p in ps:
        ref = _ref_fg_fit(slf, p)
        assert slf.fg_params(p) == ref.model_parameters
        np.testing.assert_array_equal(slf.get_resid(p), ref.residual)
        fit = slf.fg_fit(p)
        assert fit.model_parameters == ref.model_parameters
        np.testing.assert_array_equal(fit.residual, ref.residual)


@pytest.mark.parametrize("sigma_kind", ["scalar", "1D"])
def test_semi_linear_fit_neg_lk_matches_frozen_distribution(sigma_kind: str):
    """neg_lk equals the log-pdf of a frozen normal distribution, exactly."""
    fg, eor, data, sigma, ps = _slf_setup(150, sigma_kind)
    slf = SemiLinearFit(fg=fg, eor=eor, spectrum=data, sigma=sigma)
    for p in ps:
        resid = _ref_fg_fit(slf, p).residual
        ref = -np.sum(stats.norm(loc=0, scale=sigma).logpdf(resid))
        assert slf.neg_lk(p) == ref


def _plm_setup(n: int = 150, var_kind: str = "1D"):
    rng = np.random.default_rng(12)
    freqs = np.linspace(50, 100, n)
    fgm = mdl.LinLog(n_terms=5)
    eor = FlattenedGaussian(freqs=freqs, params=("amp", "w", "tau", "nu0"))
    var = rng.uniform(0.005, 0.02, size=n) ** 2
    if var_kind == "flagged":
        var[::23] = np.inf
    elif var_kind == "zero":
        var = np.zeros(n)
    t_sky = fgm.at(x=freqs)(parameters=[2000, 10, -10, 5, -5]) + rng.normal(
        scale=0.01, size=n
    )
    lfg = LinearFG(
        freqs=freqs, t_sky=t_sky, data_variance=var, fg=fgm, cosmic_signal=eor
    )
    p0 = np.array([a.fiducial for a in eor.child_active_params])
    names = ["amp", "w", "tau", "nu0"]
    return lfg.partial_linear_model, [
        dict(zip(names, p0 * f, strict=True)) for f in (1, 1.1, 0.9)
    ]


def test_unmarginalized_lnl_matches_frozen_distribution():
    """The unmarginalized log-likelihood equals that of a frozen normal, exactly."""
    plm, params = _plm_setup()
    lin = [2000, 10, -10, 5, -5]
    for p in params:
        ctx = plm.get_ctx(params=p)
        resid = (
            plm.data["t_sky"] - ctx["eor_spectrum"] - plm.linear_model(parameters=lin)
        )
        sig = np.sqrt(plm.data["data_variance"])
        ref = np.sum(stats.norm(loc=0, scale=sig).logpdf(resid))
        assert plm.get_unmarginalized_lnl(lin, p) == ref


@pytest.mark.parametrize("var_kind", ["1D", "flagged", "zero"])
def test_partial_linear_model_cached_fit_is_bit_identical(var_kind: str):
    """The static-basis linear fit equals a fresh fit, and so does the likelihood."""
    plm, params = _plm_setup(var_kind=var_kind)
    var = plm.data["data_variance"]
    wght = 1.0 if var_kind == "zero" else 1 / var
    for p in params:
        ctx = plm.get_ctx(params=p)
        fit, data, var_out = plm._reduce(ctx)
        ref = plm.linear_model.fit(ydata=data, weights=wght)
        assert var_out is var
        assert fit.model_parameters == ref.model_parameters
        np.testing.assert_array_equal(fit.residual, ref.residual)
        cached, fresh = (fit, data, var), (ref, data, var)
        assert plm.rms(cached, None) == plm.rms(fresh, None)
        assert plm.lnl(cached) == plm.lnl(fresh)
