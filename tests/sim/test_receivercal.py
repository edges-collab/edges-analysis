import numpy as np
import pytest
from astropy import units as un

from edges.cal import CalibrationObservation, Calibrator
from edges.sim.receivercal import (
    simulate_q,
    simulate_q_from_calibrator,
    simulate_qant_from_calibrator,
)


def test_simulate_q(calobs: CalibrationObservation, calibrator):
    q = simulate_q_from_calibrator(calobs.open, calibrator)
    qhot = simulate_q_from_calibrator(calobs.hot_load, calibrator)

    assert len(q) == calobs.freqs.size == len(qhot)
    assert not np.all(q == qhot)


def test_simulate_qant(calibrator: Calibrator):
    q = simulate_qant_from_calibrator(
        calibrator,
        ant_s11=np.zeros(calibrator.freqs.size),
        ant_temp=un.K * np.linspace(1, 100, calibrator.freqs.size) ** -2.5,
    )
    assert len(q) == calibrator.freqs.size


def _synthetic_calibrator() -> Calibrator:
    freqs = np.linspace(50, 100, 64) * un.MHz
    f = freqs.to_value("MHz") / 75
    return Calibrator(
        freqs=freqs,
        Tsca=1500 + 80 * f,
        Toff=300 - 5 * f,
        Tunc=20 + 3 * f,
        Tcos=5 - 2 * f,
        Tsin=-8 + f,
        receiver_s11=0.05 * np.exp(1j * freqs.to_value("MHz") / 9),
    )


@pytest.mark.parametrize("t_amb", [296 * un.K, 296.0, 20 * un.deg_C])
def test_simulate_qant_lossy_round_trip(t_amb):
    """Simulating through a lossy antenna and calibrating recovers the sky."""
    cal = _synthetic_calibrator()
    nf = cal.freqs.size
    fq = cal.freqs.to_value("MHz")
    ant_s11 = 0.2 * np.exp(-1j * fq / 7)
    ant_temp = 2000 * (fq / 75) ** -2.5 * un.K
    loss = np.linspace(0.97, 0.99, nf)

    q = simulate_qant_from_calibrator(
        cal, ant_s11=ant_s11, ant_temp=ant_temp, loss=loss, t_amb=t_amb
    )
    t_cal = cal.calibrate_q(q, ant_s11=ant_s11)

    t_amb_k = (
        t_amb.to(un.K, equivalencies=un.temperature())
        if isinstance(t_amb, un.Quantity)
        else t_amb * un.K
    )
    recovered = (t_cal - (1 - loss) * t_amb_k) / loss
    np.testing.assert_allclose(
        recovered.to_value("K"), ant_temp.to_value("K"), rtol=0, atol=1e-9
    )


def test_simulate_qant_no_loss_matches_simulate_q():
    cal = _synthetic_calibrator()
    fq = cal.freqs.to_value("MHz")
    ant_s11 = 0.2 * np.exp(-1j * fq / 7)
    ant_temp = 2000 * (fq / 75) ** -2.5 * un.K

    q = simulate_qant_from_calibrator(cal, ant_s11=ant_s11, ant_temp=ant_temp)
    expected = simulate_q(
        load_s11=ant_s11,
        receiver_s11=cal.receiver_s11,
        load_temp=ant_temp,
        t_sca=cal.Tsca,
        t_off=cal.Toff,
        t_unc=cal.Tunc,
        t_cos=cal.Tcos,
        t_sin=cal.Tsin,
    )
    np.testing.assert_allclose(q, expected, rtol=1e-14)
    np.testing.assert_allclose(
        cal.calibrate_q(q, ant_s11=ant_s11).to_value("K"),
        ant_temp.to_value("K"),
        rtol=0,
        atol=1e-9,
    )


def test_get_data_from_calobs_removed():
    from edges.sim import receivercal

    assert not hasattr(receivercal, "get_data_from_calobs")
