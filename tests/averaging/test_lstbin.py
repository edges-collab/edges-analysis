"""Tests of the lstbin module."""

import attrs
import numpy as np
import pytest
from astropy import units as un
from astropy.time import Time
from pygsdata import GSData, GSFlag

from edges.averaging.lstbin import average_over_times, get_lst_bins, lst_bin
from edges.averaging.utils import NsamplesStrategy
from edges.const import edges_location
from edges.testing import create_mock_edges_data


@pytest.fixture(scope="module")
def mock_midnight() -> GSData:
    """Mock data whose LSTs run from 23.8h through midnight to ~0.2h."""
    t0 = 2459900.27
    lst0 = Time(t0, format="jd").sidereal_time("apparent", edges_location.lon).hour
    # Convert the LST offset (sidereal hours) to solar days.
    time0 = t0 + ((23.8 - lst0) % 24) / 24 * 0.99726957
    data = create_mock_edges_data(time0=time0, ntime=36)
    rng = np.random.default_rng(9)
    return data.update(
        data=data.data * (1 + 0.01 * rng.normal(size=data.data.shape)),
        auxiliary_measurements=None,
    )


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

    def test_bad_reference_time(self, gsd_ones: GSData):
        with pytest.raises(ValueError, match="reference_time"):
            lst_bin(gsd_ones, reference_time="closest")

    @pytest.mark.parametrize("reference_time", ["min", "max", "mean"])
    def test_string_reference_times(self, gsd_ones: GSData, reference_time: str):
        out = lst_bin(gsd_ones, binsize=1.0, reference_time=reference_time)
        assert out.ntimes == 24

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


def _hours_from_midnight(lst_hours: np.ndarray) -> np.ndarray:
    lst_hours = np.asarray(lst_hours) % 24
    return np.minimum(lst_hours, 24 - lst_hours)


class TestAverageOverTimesFullyFlagged:
    def test_fully_flagged(self, gsd_ones: GSData):
        flags = GSFlag(flags=np.ones(gsd_ones.ntimes, dtype=bool), axes=("time",))
        data = gsd_ones.add_flags("all", flags)

        new = average_over_times(data)
        assert new.ntimes == 1
        assert np.all(np.isnan(new.data))
        assert np.all(new.nsamples == 0)
        # Metadata should still be sensible: the mean of all times.
        np.testing.assert_allclose(
            new.times.jd[0], np.mean(gsd_ones.times.jd, axis=0), rtol=0, atol=1e-9
        )

    def test_fully_flagged_with_resids(self, gsd_ones: GSData):
        flags = GSFlag(flags=np.ones(gsd_ones.ntimes, dtype=bool), axes=("time",))
        data = gsd_ones.update(residuals=np.zeros_like(gsd_ones.data)).add_flags(
            "all", flags
        )
        new = average_over_times(data)
        assert np.all(np.isnan(new.data))
        assert np.all(new.nsamples == 0)


class TestAcrossMidnight:
    def test_mock_spans_midnight(self, mock_midnight: GSData):
        lsts = mock_midnight.lsts.hour[:, 0]
        assert lsts[0] > 23.5
        assert lsts[-1] < 0.5

    def test_lst_bin_across_midnight(self, mock_midnight: GSData):
        new = lst_bin(mock_midnight, binsize=1.0, first_edge=23.5, max_edge=0.5)

        assert new.ntimes == 1
        np.testing.assert_allclose(
            new.data[:, :, 0], np.mean(mock_midnight.data, axis=2), rtol=1e-10
        )
        np.testing.assert_allclose(new.nsamples[:, :, 0], mock_midnight.ntimes)
        assert np.all(_hours_from_midnight(new.lsts.hour) < 1e-6)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "FLT-2: average_over_times wraps LSTs to within 12h of reference_lst and "
            "takes an arithmetic mean, so data spanning LST midnight gets a mean LST "
            "near 12h; fix pending (result-changing)"
        ),
    )
    def test_average_over_times_across_midnight(self, mock_midnight: GSData):
        new = average_over_times(mock_midnight)
        assert np.all(_hours_from_midnight(new.lsts.hour) < 0.5)
