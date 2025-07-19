"""Tests of the noise-wave fitting (iterative) procedure."""

from collections import deque

import numpy as np
import pytest

from edges import modelling as mdl
from edges.cal import noise_waves as nw

N = 501
FREQ = np.linspace(50, 100, N)


def gamma_zero(freq):
    return np.zeros(len(freq), dtype=complex)


def gamma_low(freq):
    return 1e-5 * np.exp(-1j * freq / 12)


def gamma_high(freq):
    return 1e-2 * np.exp(1j * freq / 6)


def gamma_decay(freq):
    return (freq / 75) ** -1 * np.exp(-1j * freq / 12)


def gamma_decay_flip(freq):
    return (freq / 75) ** -1 * np.exp(1j * freq / 12)


@pytest.mark.parametrize(
    ("true_sca", "true_off", "true_t_unc", "true_t_cos", "true_t_sin"),
    [
        (np.ones(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)),
        (np.linspace(3.5, 4.5, N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)),
        (np.linspace(3.5, 4.5, N), -np.ones(N), np.zeros(N), np.zeros(N), np.zeros(N)),
        (
            np.linspace(3.5, 4.5, N),
            -np.ones(N),
            np.linspace(19.0, 20.0, N),
            np.zeros(N),
            np.zeros(N),
        ),
        (
            np.linspace(3.5, 4.5, N),
            -np.ones(N),
            np.linspace(19.0, 20.0, N),
            np.linspace(4, 5, N),
            np.linspace(10, 15, N),
        ),
    ],
    ids=[
        "trivial",
        "non-zero scale",
        "non-zero scale and off",
        "non-zero tunc",
        "all non-zero",
    ],
)
@pytest.mark.parametrize(
    "gamma_rec",
    [gamma_zero, gamma_low, gamma_high],
    ids=["perfect_rx", "low-level-rx", "high-level-rx"],
)
@pytest.mark.parametrize(
    "gamma_amb",
    [gamma_zero, gamma_low],
    ids=["perfect_ra", "low-level-ra"],
)
def test_fit_perfect_receiver(
    true_sca, true_off, true_t_unc, true_t_cos, true_t_sin, gamma_rec, gamma_amb
):
    """Test that noise-wave fits work."""
    gamma_ant = {
        "ambient": gamma_amb,
        "hot_load": gamma_amb,
        "short": gamma_decay,
        "open": gamma_decay_flip,
    }

    temp = {"ambient": 300, "hot_load": 400, "short": 300, "open": 300}

    uncal_temp = {
        k: nw.decalibrate_antenna_temperature(
            temp=temp[k],
            gamma_ant=gamma_ant[k](FREQ),
            gamma_rec=gamma_rec(FREQ),
            sca=true_sca,
            off=true_off,
            t_unc=true_t_unc,
            t_cos=true_t_cos,
            t_sin=true_t_sin,
        )
        for k in temp
    }

    result = deque(
        nw.get_calibration_quantities_iterative(
            freq=FREQ,
            temp_raw=uncal_temp,
            gamma_rec=gamma_rec,
            gamma_ant=gamma_ant,
            temp_ant=temp,
            cterms=5,
            wterms=5,
        ),
        maxlen=1,
    )
    sca, off, nwv = result.pop()

    assert np.allclose(sca(FREQ), true_sca)
    assert np.allclose(off(FREQ), true_off)
    assert np.allclose(nwv.get_tunc(FREQ), true_t_unc)
    assert np.allclose(nwv.get_tcos(FREQ), true_t_cos)
    assert np.allclose(nwv.get_tsin(FREQ), true_t_sin)


def test_noise_waves(calobs):
    clb = calobs.clone(cterms=5, wterms=5)
    nwm = nw.NoiseWaves.from_calobs(clb)

    assert isinstance(nwm.linear_model, mdl.FixedLinearModel)
    assert isinstance(nwm.linear_model.model, mdl.CompositeModel)

    assert "ambient" in nwm.src_names
    assert len(nwm.get_noise_wave("tunc", src="ambient")) == clb.freq.size
    assert len(nwm.get_full_model("hot_load")) == calobs.freq.size
    assert nwm.with_params_from_calobs(clb) == nwm

    nok = nwm.get_linear_model(with_k=False)
    for k, m in nok.model.models.items():
        print(k, m.basis_scaler)
    assert all(m.basis_scaler is None for m in nok.model.models.values())
