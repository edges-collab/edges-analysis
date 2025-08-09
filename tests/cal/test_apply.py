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
