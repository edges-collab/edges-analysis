"""Tests of the load_data.Load class and its methods."""

import numpy as np
from astropy import units as un

from edges.cal.load_data import Load
from edges.io import CalObsDefEDGES2


class TestLoad:
    def test_load_from_caldef(self, caldef: CalObsDefEDGES2):
        load = Load.from_caldef(
            caldef, load_name="hot_load", ambient_temperature=300.0 * un.K
        )

        assert load.load_name == "hot_load"
        mask = ~np.isnan(load.averaged_q)
        assert np.all(load.averaged_q[mask] == load.spectrum.averaged_q[mask])
