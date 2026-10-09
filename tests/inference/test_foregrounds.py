import numpy as np
import pytest
from yabf import chi2, run_map

from edges import inference as inf


def create_mock_data(fiducial_fg_logpoly):
    spec = fiducial_fg_logpoly()
    assert len(spec["LogPoly_spectrum"]) == 100
    return spec["LogPoly_spectrum"]


def test_retrieve_params(fiducial_fg_logpoly):
    spec = create_mock_data(fiducial_fg_logpoly)
    lk = chi2.MultiComponentChi2(
        kind="spectrum", components=[fiducial_fg_logpoly], data=spec
    )
    a = run_map(lk)
    assert a.success
    assert np.allclose(a.x, [2, -2.5, 50])
    assert len(a.x) == 3


def test_damped_oscillations(fiducial_dampedoscillations):
    freqs = fiducial_dampedoscillations.freqs
    spec = fiducial_dampedoscillations()["DampedOscillations_spectrum"]
    assert len(spec) == 100

    # Analytic form: (f/nuc)^b * (A_sin sin(2 pi f / P) + A_cos cos(2 pi f / P)).
    phase = 2 * np.pi * freqs / 15
    expected = (freqs / 75.0) * (0.3 * np.sin(phase) - 0.2 * np.cos(phase))
    np.testing.assert_allclose(spec, expected, rtol=1e-12, atol=0)
    assert np.ptp(spec) > 0.5

    # Zero amplitude gives exactly nothing.
    zero = fiducial_dampedoscillations(params={"amp_sin": 0.0, "amp_cos": 0.0})
    np.testing.assert_array_equal(zero["DampedOscillations_spectrum"], 0.0)


# Realistic (all non-zero) parameters for each foreground model, so that every
# basis term contributes to the spectrum.
REALISTIC_FG_PARAMS = {
    inf.PhysicalHills: {
        "b0": 1750.0,
        "b1": -0.05,
        "b2": 0.02,
        "b3": 0.01,
        "ion_spec_index": -2.0,
        "electron_temperature": 800.0,
    },
    inf.PhysicalSmallIonDepth: {
        "b0": 1750.0,
        "b1": -0.05,
        "b2": 0.02,
        "b3": 0.01,
        "ion_spec_index": -2.0,
        "b4": 8.0,
    },
    inf.PhysicalLin: {"p0": 1750.0, "p1": -87.5, "p2": 37.2, "p3": -17.5, "p4": 8.0},
    inf.LinLog: {"beta": -2.55, "p0": 1750.0, "p2": 40.0, "p3": -15.0, "p4": 6.0},
    inf.LinPoly: {
        "beta": -2.5,
        "p0": 1750.0,
        "p1": -60.0,
        "p2": 30.0,
        "p3": -10.0,
        "p4": 3.0,
    },
    inf.LogPoly: {"p0": 3.24, "p1": -2.5, "p2": 0.1, "p3": -0.05, "p4": 0.02},
}


@pytest.mark.parametrize("model", list(REALISTIC_FG_PARAMS), ids=lambda m: m.__name__)
def test_positivity_of_foregrounds(model):
    """Foreground models are positive over the band for realistic parameters."""
    params = REALISTIC_FG_PARAMS[model]
    comp = model(freqs=np.linspace(50, 100, 100))
    fgspec = next(iter(comp(params=params).values()))
    default = next(iter(comp().values()))

    assert np.all(fgspec > 0)
    # The non-trivial parameters really change the spectrum.
    assert not np.allclose(fgspec, default, rtol=1e-3)


def test_physical_lin_derived_b_recovers_small_ion_depth():
    """Hills+ Eq. 8 is the small-perturbation linearisation of Eq. 7.

    In that limit p0 = b0, p1 = b0 b1, p2 = b0 (b2 + b1^2/2), p3 = -b0 b3, p4 = b4
    (with ionospheric spectral index -2), and PhysicalLin's derived b's invert it.
    """
    freqs = np.linspace(50, 100, 101)
    small = inf.PhysicalSmallIonDepth(freqs=freqs)
    lin = inf.PhysicalLin(freqs=freqs)

    def check(eps):
        b = {
            "b0": 1750.0,
            "b1": -1.0 * eps,
            "b2": 0.5 * eps,
            "b3": 0.8 * eps,
            "b4": 5.0,
            "ion_spec_index": -2.0,
        }
        p = {
            "p0": b["b0"],
            "p1": b["b0"] * b["b1"],
            "p2": b["b0"] * (b["b2"] + b["b1"] ** 2 / 2),
            "p3": -b["b0"] * b["b3"],
            "p4": b["b4"],
        }
        for name in ("b0", "b1", "b2", "b3", "b4"):
            assert np.isclose(getattr(lin, name)(None, **p), b[name], rtol=1e-12)

        t_small = small(params=b)["PhysicalSmallIonDepth_spectrum"]
        t_lin = lin(params=p)["PhysicalLin_spectrum"]
        return np.max(np.abs(t_lin / t_small - 1))

    err_big = check(1e-2)
    err_small = check(1e-3)
    assert err_big < 1e-4
    assert err_small < 1e-6
    # The neglected terms are second order in the perturbation.
    assert 50 < err_big / err_small < 200


def test_ion_contrib_is_ionospheric_part_of_physical_lin():
    """IonContrib is exactly the p3 (absorption) and p4 (emission) terms of Eq. 8."""
    freqs = np.linspace(50, 100, 51)
    p = {"p0": 1750.0, "p1": -87.5, "p2": 37.2, "p3": -17.5, "p4": 8.0}
    lin = inf.PhysicalLin(freqs=freqs)
    full = lin(params=p)["PhysicalLin_spectrum"]
    no_ion = lin(params={**p, "p3": 0.0, "p4": 0.0})["PhysicalLin_spectrum"]
    ion = inf.IonContrib(freqs=freqs)(params={"absorption": -17.5, "emissivity": 8.0})
    np.testing.assert_allclose(
        ion["IonContrib_spectrum"], full - no_ion, rtol=1e-10, atol=1e-10
    )


def test_linpoly_fit_p1():
    """LinPoly uses p1 (unlike LinLog), so it may be fit."""
    freqs = np.linspace(50, 100, 20)
    lp = inf.LinPoly(freqs=freqs, params=["p1"])
    assert [p.name for p in lp.child_active_params] == ["p1"]
    out = lp(params=[-60.0])["LinPoly_spectrum"]
    f = freqs / 75.0
    np.testing.assert_allclose(out, f**-2.5 * (1750 - 60 * f), rtol=1e-12)


def test_linlog_fit_p1_without_use_p1_raises():
    with pytest.raises(ValueError, match="p1"):
        inf.LinLog(freqs=np.linspace(50, 100, 20), params=["p1"])
