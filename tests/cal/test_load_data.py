"""Tests of the load_data.Load class and its methods."""
from edges.cal.load_data import Load
from edges.io import CalObsDefEDGES2
import numpy as np
from astropy import units as un

class TestLoad:
    def test_load_from_caldef(self, caldef: CalObsDefEDGES2):
        load = Load.from_caldef(
            caldef, 
            load_name="hot_load", 
            ambient_temperature=300.0*un.K
        )

        assert load.load_name == "hot_load"
        mask = ~np.isnan(load.averaged_Q)
        assert np.all(load.averaged_Q[mask] == load.spectrum.averaged_Q[mask])
