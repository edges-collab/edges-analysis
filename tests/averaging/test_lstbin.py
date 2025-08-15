"""Tests of the lstbin module."""

import attrs
import numpy as np
import pytest
from astropy import units as un
from pygsdata import GSData

from edges.averaging.lstbin import average_over_times, get_lst_bins, lst_bin
from edges.averaging.utils import NsamplesStrategy


class TestGetLSTBins:
    def test_bad_inputs(self):
        with pytest.raises(ValueError, match="Binsize must be"):
            get_lst_bins(binsize=35)

        with pytest.raises(ValueError, match="Binsize must be greater than 0"):
            get_lst_bins(binsize=-35)

    def test_simple_one_hour_bins(self):
        bins = get_lst_bins(binsize=1)
        assert np.all(bins == np.arange(25))

    def test_with_wrap(self):
        bins = get_lst_bins(binsize=1, first_edge=16, max_edge=4)
        assert np.all(bins == np.arange(16, 29))


class TestLSTBin:
    def test_simple_full_average(self, mock: GSData):
        with pytest.warns(UserWarning, match="Auxiliary measurements cannot be binned"):
            data = lst_bin(mock)
        manual = np.mean(mock.data, axis=2)
        np.testing.assert_array_almost_equal(data.data[:, :, 0], manual)

    def test_bad_inputs(self, mock: GSData):
        with pytest.raises(
            ValueError, match="Cannot bin with models without residuals"
        ):
            lst_bin(mock, use_model_residuals=True)

        rng = np.random.default_rng()
        mock2 = mock.update(
            effective_integration_time=rng.uniform(size=mock.data.shape[:-1]) * un.s,
            auxiliary_measurements=None,
        )
        with pytest.warns(
            UserWarning, match="lstbin does not yet support variable integration times"
        ):
            lst_bin(mock2)

    def test_with_model(self, mock_with_model: GSData):
        rng = np.random.default_rng()
        new = mock_with_model.update(
            nsamples=rng.uniform(size=mock_with_model.nsamples.shape)
        )
        with pytest.warns(UserWarning, match="Auxiliary measurements cannot be binned"):
            # mock_with_model has auxiliary measurements, check that we warn when lst
            # binning!
            data = lst_bin(new)
        manual = np.mean(new.data, axis=2)
        assert not np.allclose(data.data[:, :, 0], manual)


class TestAverageOverTimes:
    def test_all_ones(self, gsd_ones: GSData):
        """Testr that averaging over times doesn't error out."""
        new = average_over_times(gsd_ones)
        assert np.all(new.data == 1.0)

    @pytest.mark.parametrize(
        "nsamples_strategy",
        [
            NsamplesStrategy.FLAGGED_NSAMPLES,
            NsamplesStrategy.FLAGS_ONLY,
            NsamplesStrategy.FLAGGED_NSAMPLES_UNIFORM,
            NsamplesStrategy.NSAMPLES_ONLY,
        ],
    )
    def test_with_nans(self, gsd_ones: GSData, nsamples_strategy: NsamplesStrategy):
        """Test that averaging over times doesn't error out with nans."""
        new = attrs.evolve(gsd_ones, data=gsd_ones.data.copy())

        new.data[0, 0, 0] = np.nan
        new = average_over_times(new, nsamples_strategy=nsamples_strategy)
        assert np.all(new.data == 1.0)
