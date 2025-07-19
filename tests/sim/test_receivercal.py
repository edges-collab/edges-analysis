import numpy as np

from edges.cal import CalibrationObservation
from edges.sim.receivercal import simulate_q_from_calobs, simulate_qant_from_calobs


def test_simulate_q(calobs: CalibrationObservation):
    q = simulate_q_from_calobs(calobs, "open")
    qhot = simulate_q_from_calobs(calobs, "hot_load", freq=calobs.freq)

    assert len(q) == calobs.freq.size == len(qhot)
    assert not np.all(q == qhot)


def test_simulate_qant(calobs: CalibrationObservation):
    q = simulate_qant_from_calobs(
        calobs,
        ant_s11=np.zeros(calobs.freq.size),
        ant_temp=np.linspace(1, 100, calobs.freq.size) ** -2.5,
    )
    assert len(q) == calobs.freq.size
