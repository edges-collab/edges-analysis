"""Tests of the load_data.Load class and its methods."""

import attrs
import numpy as np
import pytest
from astropy import units as un
from pygsdata import GSData

from edges.cal import LoadSpectrum, ReflectionCoefficient
from edges.cal import sparams as sp
from edges.cal.input_sources import InputSource
from edges.io import CalObsDefEDGES2


class TestInputSource:
    @pytest.mark.parametrize("load_name", ["hot_load", "HotLoad"])
    def test_input_source_from_caldef(self, caldef: CalObsDefEDGES2, load_name: str):
        load = InputSource.from_caldef(
            caldef, load_name=load_name, ambient_temperature=300.0 * un.K
        )

        assert load.name == "hot_load"
        mask = ~np.isnan(load.averaged_q)
        assert np.all(load.averaged_q[mask] == load.spectrum.averaged_q[mask])

    def test_unknown_load_name(self, caldef: CalObsDefEDGES2):
        with pytest.raises(ValueError, match="Unknown calibration load 'hotload'"):
            InputSource.from_caldef(caldef, load_name="hotload")

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

    @pytest.mark.parametrize("restrict", [True, False])
    def test_from_caldef_same_as_from_components(
        self, caldef: CalObsDefEDGES2, restrict: bool
    ):
        """from_caldef assembles the source with from_components."""
        f_low, f_high = 50 * un.MHz, 100 * un.MHz
        load = InputSource.from_caldef(
            caldef,
            load_name="open",
            f_low=f_low,
            f_high=f_high,
            ambient_temperature=300.0 * un.K,
            restrict_s11_freqs=restrict,
        )
        new = InputSource.from_components(
            spectrum=load.spectrum,
            raw_s11=load._raw_s11,
            s11_model_params=sp.input_source_model_params(name="open"),
            s11_freq_range=(f_low, f_high) if restrict else None,
            ambient_temperature=300.0 * un.K,
            name="open",
        )
        assert new.name == "open"
        assert new._raw_s11 is load._raw_s11
        np.testing.assert_array_equal(
            new.reflection_coefficient.s11, load.reflection_coefficient.s11
        )
        np.testing.assert_array_equal(new.loss, 1.0)
        assert new.ambient_temperature == 300.0 * un.K

    def test_from_components_s11_freq_range(self, caldef: CalObsDefEDGES2):
        """The S11 fit range is independent of the frequencies of the spectrum."""
        load = InputSource.from_caldef(
            caldef,
            load_name="short",
            f_low=50 * un.MHz,
            f_high=100 * un.MHz,
            ambient_temperature=300.0 * un.K,
        )
        params = sp.input_source_model_params(name="short")
        s11_range = (60 * un.MHz, 150 * un.MHz)
        new = InputSource.from_components(
            spectrum=load.spectrum,
            raw_s11=load._raw_s11,
            s11_model_params=params,
            s11_freq_range=s11_range,
        )
        expected = load._raw_s11.select_frequencies(*s11_range).smoothed(
            params, freqs=load.freqs
        )
        np.testing.assert_array_equal(new.reflection_coefficient.s11, expected.s11)
        assert new.reflection_coefficient.freqs.size == load.freqs.size
        assert new.name == ""

    def test_from_caldef_given_s11_model_and_spectrum_range(
        self, caldef: CalObsDefEDGES2
    ):
        """S11 model parameters and a spectrum range given in the kwargs are used."""
        params = sp.input_source_model_params(name="short")
        params = attrs.evolve(params, model=params.model.with_nterms(11))
        load = InputSource.from_caldef(
            caldef,
            load_name="short",
            ambient_temperature=300.0 * un.K,
            s11_kwargs={"model_params": params},
            spec_kwargs={"f_low": 60 * un.MHz, "f_high": 90 * un.MHz},
        )
        assert load.freqs.min() >= 60 * un.MHz
        assert load.freqs.max() <= 90 * un.MHz
        expected = load._raw_s11.smoothed(params, freqs=load.freqs)
        np.testing.assert_array_equal(load.reflection_coefficient.s11, expected.s11)


class TestTempWithLoss:
    """Limits of the loss-corrected source temperature."""

    def _source(self, gsd: GSData, loss: np.ndarray) -> InputSource:
        return InputSource(
            spectrum=LoadSpectrum(q=gsd, temp_ave=370.0 * un.K),
            reflection_coefficient=ReflectionCoefficient(
                freqs=gsd.freqs, reflection_coefficient=np.zeros(gsd.nfreqs)
            ),
            ambient_temperature=296.0 * un.K,
            loss=loss,
        )

    def test_no_loss_is_identity(self, gsd_averaged: GSData):
        src = self._source(gsd_averaged, np.ones(gsd_averaged.nfreqs))
        np.testing.assert_allclose(src.temp_ave.to_value("K"), 370.0, rtol=0, atol=0)

    def test_total_loss_is_ambient(self, gsd_averaged: GSData):
        src = self._source(gsd_averaged, np.zeros(gsd_averaged.nfreqs))
        np.testing.assert_allclose(src.temp_ave.to_value("K"), 296.0, rtol=0, atol=0)

    def test_partial_loss(self, gsd_averaged: GSData):
        loss = np.linspace(0.5, 1, gsd_averaged.nfreqs)
        src = self._source(gsd_averaged, loss)
        np.testing.assert_allclose(
            src.temp_ave.to_value("K"), loss * 370 + (1 - loss) * 296, rtol=1e-14
        )

    def test_temp_ave_updates_after_setting_loss(self, gsd_averaged: GSData):
        """Regression test: temp_ave was cached and went stale on mutation."""
        src = self._source(gsd_averaged, np.ones(gsd_averaged.nfreqs))
        np.testing.assert_allclose(src.temp_ave.to_value("K"), 370.0, rtol=0, atol=0)

        src.loss = np.zeros(gsd_averaged.nfreqs)
        np.testing.assert_allclose(src.temp_ave.to_value("K"), 296.0, rtol=0, atol=0)

    def test_temp_ave_updates_after_setting_ambient(self, gsd_averaged: GSData):
        src = self._source(gsd_averaged, np.zeros(gsd_averaged.nfreqs))
        np.testing.assert_allclose(src.temp_ave.to_value("K"), 296.0, rtol=0, atol=0)

        src.ambient_temperature = 300.0 * un.K
        np.testing.assert_allclose(src.temp_ave.to_value("K"), 300.0, rtol=0, atol=0)
