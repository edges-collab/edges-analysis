"""Tests of the Dicke-switch calibration."""

import numpy as np
from astropy import units as un
from astropy.time import Time
from pygsdata import GSData

from edges import const
from edges.cal.dicke import dicke_calibration

LOADS = ("ant", "internal_load", "internal_load_plus_noise_source")


def _power_data(p_ant: np.ndarray, p_load: float, p_ns: float) -> GSData:
    nfreq = p_ant.shape[-1]
    ntime = 4
    data = np.zeros((3, 1, ntime, nfreq))
    data[0] = p_ant
    data[1] = p_load
    data[2] = p_ns
    times = np.linspace(2459856, 2459856.1, ntime)[:, None] + np.array([0, 1, 2]) / 1e5
    return GSData(
        data=data,
        freqs=np.linspace(50, 100, nfreq) * un.MHz,
        times=Time(times, format="jd", scale="utc"),
        telescope=const.KNOWN_TELESCOPES["edges-low"],
        loads=LOADS,
        data_unit="power",
    )


def test_ant_equals_load_gives_zero():
    p_load = 3.0
    q = dicke_calibration(_power_data(np.full(16, p_load), p_load, 5.0))
    assert q.loads == ("ant",)
    assert q.data_unit == "uncalibrated"
    np.testing.assert_allclose(q.data, 0, rtol=0, atol=0)


def test_ant_equals_noise_source_gives_one():
    p_ns = 5.0
    q = dicke_calibration(_power_data(np.full(16, p_ns), 3.0, p_ns))
    np.testing.assert_allclose(q.data, 1, rtol=1e-15)


def test_linear_in_antenna_power():
    p_ant = np.linspace(1, 10, 16)
    q = dicke_calibration(_power_data(p_ant, 3.0, 5.0))
    np.testing.assert_allclose(q.data[0, 0, 0], (p_ant - 3.0) / 2.0, rtol=1e-14)


def test_ns_equals_load_is_not_finite():
    """When P_NS = P_L the ratio is undefined; it must not raise."""
    q = dicke_calibration(_power_data(np.full(16, 4.0), 3.0, 3.0))
    assert not np.any(np.isfinite(q.data))
