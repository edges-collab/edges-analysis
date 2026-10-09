import numpy as np
import pytest
from pygsdata import GSData

from edges.cal import apply


def test_approximate_temperature(gsd_ones: GSData):
    # mock gsdata
    data = gsd_ones.update(data_unit="uncalibrated")

    new = apply.approximate_temperature(data, tload=300, tns=1000)
    assert new.data_unit == "uncalibrated_temp"
    assert not np.any(new.data == data.data)

    new = data.update(data_unit="temperature")
    with pytest.raises(ValueError, match="data_unit must be 'uncalibrated'"):
        apply.approximate_temperature(new, tload=300, tns=1000)


def test_approximate_temperature_round_trip(gsd_ones: GSData):
    rng = np.random.default_rng(0)
    q = rng.uniform(-0.5, 2, size=gsd_ones.data.shape)
    data = gsd_ones.update(data=q, data_unit="uncalibrated")

    tapprox = apply.approximate_temperature(data, tload=300.0, tns=1000.0)
    np.testing.assert_allclose(tapprox.data, q * 1000.0 + 300.0, rtol=1e-14)

    back = apply.approximate_temperature(tapprox, tload=300.0, tns=1000.0, reverse=True)
    assert back.data_unit == "uncalibrated"
    np.testing.assert_allclose(back.data, q, rtol=0, atol=1e-13)

    with pytest.raises(ValueError, match="data_unit must be 'uncalibrated_temp'"):
        apply.approximate_temperature(data, tload=300.0, tns=1000.0, reverse=True)
