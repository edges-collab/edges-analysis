"""End-to-end tests of the LinearFG likelihood."""

import numpy as np

from edges import modeling as mdl
from edges.inference import GaussianAbsorptionProfile, LinearFG
from edges.inference.eor_models import flattened_gaussian, simple_gaussian

TRUTH = {"amp": 0.5, "w": 20.0, "tau": 7.0, "nu0": 78.0}


def _mock(eor_func=flattened_gaussian, **eor_params):
    freqs = np.linspace(55, 100, 100)
    fg = mdl.LinLog(n_terms=4)
    t_sky = fg(x=freqs, parameters=[1750, -90, 30, -8]) + eor_func(freqs, **eor_params)
    return freqs, fg, t_sky, np.full(freqs.size, 1e-4)


def test_linear_fg_defaults_construct_and_evaluate():
    """LinearFG works with its default (flattened-Gaussian) cosmic signal."""
    freqs, fg, t_sky, var = _mock(**TRUTH)
    lfg = LinearFG(freqs=freqs, t_sky=t_sky, data_variance=var, fg=fg)
    plm = lfg.partial_linear_model

    assert [p.name for p in plm.child_active_params] == ["amp", "w", "tau", "nu0"]

    lnl_true = plm(params=TRUTH)[0]
    assert np.isfinite(lnl_true)
    assert np.isfinite(plm()[0])  # at the fiducial parameters

    # Noise-free data: at the truth the residual vanishes, so lnL = -0.5 log|Q|.
    q = plm.Q
    np.testing.assert_allclose(lnl_true, -0.5 * np.linalg.slogdet(q)[1], atol=1e-6)

    # The truth is preferred over a perturbed amplitude.
    assert plm(params={**TRUTH, "amp": 0.4})[0] < lnl_true - 10


def test_linear_fg_custom_cosmic_signal():
    # w is not fit, so the mock uses the component's fiducial w.
    freqs, fg, t_sky, var = _mock(simple_gaussian, amp=0.5, nu0=78.0, w=17.0)
    lfg = LinearFG(
        freqs=freqs,
        t_sky=t_sky,
        data_variance=var,
        fg=fg,
        cosmic_signal=GaussianAbsorptionProfile(freqs=freqs, params=("amp", "nu0")),
    )
    plm = lfg.partial_linear_model
    assert [p.name for p in plm.child_active_params] == ["amp", "nu0"]
    best = plm(params={"amp": 0.5, "nu0": 78.0})[0]
    assert plm(params={"amp": 0.5, "nu0": 76.0})[0] < best
