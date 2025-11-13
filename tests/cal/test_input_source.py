"""Tests of the load_data.Load class and its methods."""

import attrs
import numpy as np
import pytest
from astropy import units as un

from edges.cal.input_sources import InputSource
from edges.io import CalObsDefEDGES2


class TestInputSource:
    def test_input_source_from_caldef(self, caldef: CalObsDefEDGES2):
        load = InputSource.from_caldef(
            caldef, load_name="hot_load", ambient_temperature=300.0 * un.K
        )

        assert load.name == "hot_load"
        mask = ~np.isnan(load.averaged_q)
        assert np.all(load.averaged_q[mask] == load.spectrum.averaged_q[mask])

    def test_bad_inputs(self, caldef: CalObsDefEDGES2):
        """Test that bad inputs raise appropriate errors."""
        load = InputSource.from_caldef(
            caldef, load_name="hot_load", ambient_temperature=300.0 * un.K
        )

        rc = attrs.evolve(
            load.reflection_coefficient,
            reflection_coefficient=load.reflection_coefficient.reflection_coefficient[
                :-1
            ],
            freqs=load.reflection_coefficient.freqs[:-1],
        )
        # Mismatched freq sizes
        with pytest.raises(
            ValueError,
            match="reflection_coefficient must have the same number of channels",
        ):
            attrs.evolve(load, reflection_coefficient=rc)

        loss = load.loss[:-1]
        # Mismatched freq sizes
        with pytest.raises(
            ValueError, match="loss must have the same number of channels"
        ):
            attrs.evolve(load, loss=loss)
