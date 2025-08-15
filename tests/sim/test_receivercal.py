import numpy as np
from astropy import units as un

from edges.cal import CalibrationObservation, Calibrator
from edges.sim.receivercal import (
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
