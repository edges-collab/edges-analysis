"""Tests of the noise-wave fitting (iterative) procedure."""

from collections import deque

import numpy as np
import pytest
from astropy import units as un

from edges import modeling as mdl
from edges.cal import Calibrator
from edges.cal import noise_waves as nw

N = 500
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
        "ambient": gamma_amb(FREQ),
        "hot_load": gamma_amb(FREQ),
        "short": gamma_decay(FREQ),
        "open": gamma_decay_flip(FREQ),
    }

    temp = {
        "ambient": 300 * un.K,
        "hot_load": 1000 * un.K,
        "short": 300 * un.K,
        "open": 300 * un.K,
    }

    calibrator = Calibrator(
        freqs=FREQ,
        Tsca=true_sca * 1000,
        Toff=300 - true_off,
        Tcos=true_t_cos,
        Tsin=true_t_sin,
        Tunc=true_t_unc,
        receiver_s11=gamma_rec(FREQ),
    )

    Q = {
        k: calibrator.decalibrate(
            temp=temp[k],
            ant_s11=gamma_ant[k],
        )
        for k in temp
    }
    print({name: np.where(~np.isfinite(q)) for name, q in Q.items()})
    result = deque(
        nw.get_calibration_quantities_iterative(
            freqs=FREQ * un.MHz,
            source_q=Q,
            receiver_s11=gamma_rec(FREQ),
            source_s11s=gamma_ant,
            source_true_temps=temp,
            cterms=5,
            wterms=5,
        ),
        maxlen=1,
    )
    sca, off, nwv = result.pop()

    assert np.allclose(sca(FREQ), calibrator.Tsca)
    assert np.allclose(off(FREQ), calibrator.Toff)
    assert np.allclose(nwv.get_tunc(), calibrator.Tunc)
    assert np.allclose(nwv.get_tcos(), calibrator.Tcos)
    assert np.allclose(nwv.get_tsin(), calibrator.Tsin)


def test_noise_waves(calobs):
    print("RECEIVER S11:", calobs.receiver.s11)
    nwm = nw.NoiseWaves.from_calobs(calobs, cterms=5, wterms=5)

    assert isinstance(nwm.linear_model, mdl.FixedLinearModel)
    assert isinstance(nwm.linear_model.model, mdl.CompositeModel)

    assert "ambient" in nwm.src_names
    with pytest.raises(
        ValueError, match="Cannot evaluate a model without providing parameters"
    ):
        nwm.get_noise_wave("tunc", src="ambient")
    with pytest.raises(
        ValueError, match="You must supply parameters to evaluate the model"
    ):
        nwm.get_full_model("hot_load")

    nok = nwm.get_linear_model(with_k=False)
    assert all(m.basis_scaler is None for m in nok.model.models.values())
