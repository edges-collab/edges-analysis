"""Test spectrum reading."""

import numpy as np
import pytest
from astropy import units as un

from edges.cal import LoadSpectrum
from edges.io.calobsdef import CalObsDefEDGES2


class TestLoadSpectrum:
    def test_read(self, caldef: CalObsDefEDGES2):
        for load in ["ambient", "hot_load", "open", "short"]:
            spec = LoadSpectrum.from_loaddef(
                getattr(caldef, load), f_high=100 * un.MHz, f_low=50 * un.MHz
            )

            print(spec.averaged_Q.shape)
            print(np.where(np.isnan(spec.averaged_Q)))

            mask = ~np.isnan(spec.averaged_Q)
            assert np.sum(~mask) < 100
            assert spec.averaged_Q.ndim == 1

            assert np.all(~np.isnan(spec.variance_Q[mask]))
            assert ~np.isinf(spec.temp_ave)
            assert ~np.isnan(spec.temp_ave)

    def test_gauss_smooth(self, caldef: CalObsDefEDGES2):
        spec = LoadSpectrum.from_loaddef(
            caldef.ambient,
            f_high=100 * un.MHz,
            f_low=50 * un.MHz,
            freq_bin_size=8,
            frequency_smoothing="gauss",
        )
        spec2 = LoadSpectrum.from_loaddef(
            caldef.ambient,
            f_high=100 * un.MHz,
            f_low=50 * un.MHz,
            freq_bin_size=8,
            frequency_smoothing="bin",
        )

        # gauss smooth has one more value because the way it bins is like Alan's code,
        # which starts at index 0, rather than in the middle of the bin.
        assert len(spec.averaged_Q) - 1 == len(spec2.averaged_Q)
        mask = (~np.isnan(spec.averaged_Q[:-1])) & (~np.isnan(spec2.averaged_Q))
        np.testing.assert_allclose(
            spec.averaged_Q[:-1][mask], spec2.averaged_Q[mask], atol=1e-2, rtol=0.1
        )

    def test_equality(self, caldef: CalObsDefEDGES2):
        spec1 = LoadSpectrum.from_loaddef(
            caldef.ambient, f_high=100 * un.MHz, f_low=50 * un.MHz
        )
        spec2 = LoadSpectrum.from_loaddef(
            caldef.ambient, f_high=100 * un.MHz, f_low=50 * un.MHz
        )

        assert spec1 == spec2
        assert spec1.metadata["hash"] == spec2.metadata["hash"]

        spec3 = LoadSpectrum.from_loaddef(
            caldef.ambient, f_high=100 * un.MHz, f_low=60 * un.MHz
        )
        assert spec3.metadata["hash"] != spec2.metadata["hash"]

    def test_temperature_range(self, caldef: CalObsDefEDGES2):
        with pytest.raises(RuntimeError, match="The temperature range has masked"):
            # Fails only because our test data is awful. spectra and thermistor measurements
            # don't overlap.
            LoadSpectrum.from_loaddef(caldef.ambient, temperature_range=0.5)

        with pytest.raises(RuntimeError, match="The temperature range has masked"):
            LoadSpectrum.from_loaddef(caldef.ambient, temperature_range=(20, 40))
