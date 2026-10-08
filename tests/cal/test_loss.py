"""Physical-limit tests of the cable-loss (available gain) calculation."""

import numpy as np
import pytest
from astropy import units as un

from edges.cal import ReflectionCoefficient, SParams
from edges.cal.loss import (
    LossFunctionGivenSparams,
    compute_cable_loss_from_scattering_params,
)

N = 200
FREQS = np.linspace(50, 150, N) * un.MHz


def _random_load(rng, nmax: float = 0.95) -> np.ndarray:
    """Random passive reflection coefficients (|Γ| < nmax)."""
    return np.sqrt(rng.uniform(0, nmax**2, N)) * np.exp(
        2j * np.pi * rng.uniform(size=N)
    )


def _gamma_in(sparams: SParams, gamma_load: np.ndarray) -> np.ndarray:
    """Reflection coefficient looking into port 1 with gamma_load on port 2."""
    return sparams.s11 + sparams.s12 * sparams.s21 * gamma_load / (
        1 - sparams.s22 * gamma_load
    )


def _loss(sparams: SParams, gamma_load: np.ndarray) -> np.ndarray:
    gamma = ReflectionCoefficient(
        freqs=FREQS, reflection_coefficient=_gamma_in(sparams, gamma_load)
    )
    return compute_cable_loss_from_scattering_params(gamma, sparams)


def test_lossless_matched_line():
    rng = np.random.default_rng(0)
    phase = np.exp(-2j * np.pi * FREQS.to_value("MHz") / 37)
    sp = SParams(freqs=FREQS, s11=np.zeros(N), s12=phase, s22=np.zeros(N))

    loss = _loss(sp, _random_load(rng))
    np.testing.assert_allclose(loss, 1, rtol=0, atol=1e-12)


@pytest.mark.parametrize("rho", [0.1, 0.5, 0.9])
def test_unitary_mismatched_two_port(rho):
    """A lossless (unitary) but mismatched 2-port has unit available gain."""
    rng = np.random.default_rng(1)
    fq = FREQS.to_value("MHz")
    # Unitary reciprocal S = [[r, t], [t, r']] with |r|²+|t|² = 1 and
    # r t* + t r'* = 0, i.e. r' = -r* t / t*.
    r = rho * np.exp(1j * fq / 13)
    t = np.sqrt(1 - rho**2) * np.exp(-1j * fq / 20)
    rp = -np.conj(r) * t / np.conj(t)
    smat = np.array([[r, t], [t, rp]]).transpose(2, 0, 1)
    ident = smat @ np.conj(np.swapaxes(smat, -1, -2))
    np.testing.assert_allclose(
        ident, np.broadcast_to(np.eye(2), ident.shape), atol=1e-12
    )

    sp = SParams(freqs=FREQS, s11=r, s12=t, s22=rp)
    loss = _loss(sp, _random_load(rng))
    np.testing.assert_allclose(loss, 1, rtol=0, atol=1e-10)


@pytest.mark.parametrize("g", [0.5, 0.9, 0.99])
def test_matched_attenuator(g):
    """A matched attenuator of power gain g has G = g(1-|Γ|²)/(1-g²|Γ|²)."""
    rng = np.random.default_rng(2)
    sp = SParams(
        freqs=FREQS,
        s11=np.zeros(N),
        s12=np.sqrt(g) * np.ones(N),
        s22=np.zeros(N),
    )
    gamma_load = _random_load(rng)
    expected = g * (1 - np.abs(gamma_load) ** 2) / (1 - g**2 * np.abs(gamma_load) ** 2)
    np.testing.assert_allclose(_loss(sp, gamma_load), expected, rtol=1e-12)

    # A matched load gives exactly the attenuator's gain.
    np.testing.assert_allclose(_loss(sp, np.zeros(N)), g, rtol=1e-12)


def test_passive_reciprocal_network_gain_bounded():
    """Any passive reciprocal 2-port has 0 <= G <= 1."""
    rng = np.random.default_rng(3)
    for _ in range(10):
        a = rng.normal(size=(N, 2, 2)) + 1j * rng.normal(size=(N, 2, 2))
        s = a + np.swapaxes(a, -1, -2)  # reciprocal (symmetric)
        smax = np.linalg.svd(s, compute_uv=False)[:, 0]
        s *= (rng.uniform(0.2, 0.999, N) / smax)[:, None, None]

        sp = SParams(freqs=FREQS, s11=s[:, 0, 0], s12=s[:, 0, 1], s22=s[:, 1, 1])
        loss = _loss(sp, _random_load(rng))
        assert np.all(loss >= 0)
        assert np.all(loss <= 1 + 1e-10)


def test_loss_function_given_sparams():
    g = 0.8
    sp = SParams(
        freqs=FREQS, s11=np.zeros(N), s12=np.sqrt(g) * np.ones(N), s22=np.zeros(N)
    )
    func = LossFunctionGivenSparams(sp)
    gamma = ReflectionCoefficient(freqs=FREQS, reflection_coefficient=np.zeros(N))
    np.testing.assert_allclose(func(gamma), g, rtol=1e-12)

    bad = ReflectionCoefficient(
        freqs=FREQS[:-1], reflection_coefficient=np.zeros(N - 1)
    )
    with pytest.raises(ValueError, match="same length"):
        func(bad)
