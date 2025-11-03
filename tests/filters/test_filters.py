"""Test the filters module."""

import numpy as np
import pytest
from astropy import units as un
from pygsdata import GSData, GSFlag
from pygsdata.concat import concat
from pygsdata.select import select_freqs

from edges import modeling as mdl
from edges.averaging.utils import NsamplesStrategy
from edges.filters import filters
from edges.testing import create_mock_edges_data


def run_filter_check(data: GSData, fnc: callable, **kwargs):
    new_data = fnc(data, **kwargs)
    if isinstance(new_data, GSData):
        assert new_data.data.shape == data.data.shape
        assert fnc.__name__ in new_data.flags
        assert len(new_data.flags) - len(data.flags) == 1
    else:
        for nd, d in zip(new_data, data, strict=False):
            assert nd.data.shape == d.data.shape
            assert fnc.__name__ in nd.flags
            assert len(nd.flags) - len(d.flags) == 1


class TestAuxFilter:
    def test_basic_checks(self, mock):
        run_filter_check(
            mock,
            filters.aux_filter,
            maxima={"ambient_hum": 100, "receiver_temp": 100},
        )

    def test_aux_data_with_loads(self, mock_power: GSData):
        mock = mock_power.update(
            auxiliary_measurements={
                "ambient_hum": np.array([
                    np.linspace(10, 90, mock_power.ntimes),
                    np.linspace(50, 150, mock_power.ntimes),
                    np.linspace(10, 90, mock_power.ntimes),
                ]).T
            }
        )
        data = filters.aux_filter(mock, maxima={"ambient_hum": 100})
        assert np.sum(data.flags["aux_filter"].flags) == mock.ntimes // 2

    def test_aux_data_minima(self, mock):
        data = filters.aux_filter(mock, minima={"ambient_hum": 0})
        assert not np.any(data.flags["aux_filter"].flags)

    def test_bad_key(self, mock):
        with pytest.raises(ValueError, match="bad_key not in data"):
            filters.aux_filter(mock, minima={"bad_key": 0})

        with pytest.raises(ValueError, match="bad_key not in data"):
            filters.aux_filter(mock, maxima={"bad_key": 0})


def test_sun_filter(mock):
    run_filter_check(
        mock,
        filters.sun_filter,
        elevation_range=(-np.inf, 40),
    )


def test_moon_filter(mock):
    run_filter_check(
        mock,
        filters.moon_filter,
        elevation_range=(-np.inf, 40),
    )


class TestPeakPowerFilter:
    def test_basic(self, mock):
        run_filter_check(mock, filters.peak_power_filter)

    def test_bad_inputs(self, mock):
        with pytest.raises(
            ValueError, match="The frequency range of the peak must be non-zero"
        ):
            filters.peak_power_filter(data=mock, peak_freq_range=(60, 60))

        with pytest.raises(
            ValueError,
            match="The freq range of the mean must be a tuple with first less than",
        ):
            filters.peak_power_filter(data=mock, mean_freq_range=(61, 60))

    def test_passing_mean_freq_range(self, mock):
        data = filters.peak_power_filter(mock, mean_freq_range=(60, 80))
        assert "peak_power_filter" in data.flags

    def test_passing_mean_freq_range_bad(self, mock):
        data = filters.peak_power_filter(mock, mean_freq_range=(250, 270))
        assert not np.any(data.complete_flags)


def test_peak_orbcomm_filter(mock):
    run_filter_check(mock, filters.peak_orbcomm_filter)


class TestRMSF:
    def test_basic(self, mock):
        run_filter_check(mock, filters.rmsf_filter, threshold=100)

    def test_all_outside_range(self, mock):
        new = filters.rmsf_filter(mock, freq_range=(200, 250))
        assert not np.any(new.complete_flags)


class TestMaxFM:
    def test_basic_run(self, mock):
        run_filter_check(mock, filters.maxfm_filter, threshold=200)

    def test_no_fm(self, mock):
        new = select_freqs(mock, freq_range=(50 * un.MHz, 70 * un.MHz))
        out = filters.maxfm_filter(new)
        assert not np.any(out.complete_flags)


class Test150MHzFilter:
    def test_basic_no_freqs(self, mock):
        run_filter_check(mock, filters.filter_150mhz, threshold=100)

    def test_with_freqs(self):
        mock = create_mock_edges_data(fhigh=200 * un.MHz)
        data = filters.filter_150mhz(mock, threshold=100)
        assert not np.any(data.complete_flags)


class TestPowerPercentFilter:
    def test_basic(self, mock_power):
        run_filter_check(mock_power, filters.power_percent_filter)

    def test_on_non_power_data(self, mock):
        with pytest.raises(
            ValueError, match="Cannot perform power percent filter on non-power data"
        ):
            filters.power_percent_filter(mock)

    def test_total_freq_range(self, mock_power):
        """Test that if we sum over all frequencies, we get a ratio of unity."""
        data = filters.power_percent_filter(
            mock_power,
            freq_range=(
                mock_power.freqs.min().to_value("MHz"),
                mock_power.freqs.max().to_value("MHz"),
            ),
            min_threshold=0,
            max_threshold=101,  # should be 100%
        )
        assert not np.any(data.complete_flags)


class TestXRFI:
    def test_basic_run(self, mock: GSData):
        run_filter_check(mock, filters.rfi_iterative_filter, freq_range=(40, 100))

    @pytest.mark.parametrize(
        "strategy",
        [
            NsamplesStrategy.FLAGGED_NSAMPLES,
            NsamplesStrategy.NSAMPLES_ONLY,
            NsamplesStrategy.FLAGGED_NSAMPLES_UNIFORM,
            NsamplesStrategy.FLAGS_ONLY,
        ],
    )
    def test_nsamples_strategy(self, mock, strategy):
        data = filters.rfi_iterative_filter(
            mock, freq_range=(40, 100), nsamples_strategy=strategy
        )
        assert not np.any(data.complete_flags)

    def test_bad_strategy(self, mock):
        with pytest.raises(ValueError, match="Invalid nsamples_strategy"):
            filters.rfi_model_filter(
                mock, freq_range=(40, 100), nsamples_strategy="bad_strategy"
            )


class TestApplyFlags:
    def test_from_obj(self, mock):
        flags = GSFlag(flags=np.ones(mock.ntimes), axes=("time",))
        data = filters.apply_flags(mock, flags=flags)
        assert np.all(data.complete_flags)

    def test_from_file(self, mock, tmp_path):
        pth = tmp_path / "tmp.gsflag"
        flags = GSFlag(flags=np.ones(mock.ntimes), axes=("time",))
        flags.write_gsflag(pth)
        data = filters.apply_flags(mock, flags=pth)
        assert np.all(data.complete_flags)


class TestFlagFrequencyRanges:
    def test_single_range(self, mock):
        data = filters.flag_frequency_ranges(mock, freq_ranges=[(0, 200)])
        assert np.all(data.complete_flags)

    def test_single_range_inverted(self, mock):
        data = filters.flag_frequency_ranges(mock, invert=True, freq_ranges=[(0, 200)])
        assert not np.any(data.complete_flags)

    def test_multiple_non_overlapping(self, mock):
        data = filters.flag_frequency_ranges(mock, freq_ranges=[(50, 60), (70, 80)])
        fq = data.freqs.to_value("MHz")
        idx = ((fq >= 50) & (fq < 60)) | ((fq >= 70) & (fq < 80))
        assert np.all(data.complete_flags[..., idx])
        assert not np.any(data.complete_flags[..., ~idx])

    def test_multiple_non_overlapping_inverted(self, mock):
        data = filters.flag_frequency_ranges(
            mock, invert=True, freq_ranges=[(50, 60), (70, 80)]
        )
        fq = data.freqs.to_value("MHz")
        idx = ((fq >= 50) & (fq < 60)) | ((fq >= 70) & (fq < 80))

        assert not np.any(data.complete_flags[..., idx])
        assert np.all(data.complete_flags[..., ~idx])

    def test_multiple_overlapping(self, mock):
        data = filters.flag_frequency_ranges(mock, freq_ranges=[(50, 60), (55, 65)])
        fq = data.freqs.to_value("MHz")

        idx = (fq >= 50) & (fq < 65)

        assert np.all(data.complete_flags[..., idx])
        assert not np.any(data.complete_flags[..., ~idx])

    def test_multiple_overlapping_invert(self, mock):
        data = filters.flag_frequency_ranges(
            mock, invert=True, freq_ranges=[(50, 60), (55, 65)]
        )
        fq = data.freqs.to_value("MHz")

        idx = (fq >= 50) & (fq < 65)

        assert not np.any(data.complete_flags[..., idx])
        assert np.all(data.complete_flags[..., ~idx])


class TestNegativePowerFilter:
    def test_with_no_negatives(self, mock):
        data = filters.negative_power_filter(mock)
        assert not np.any(data.complete_flags)

    def test_with_negatives(self, mock):
        dd = mock.data.copy()
        dd[0, 0, 0, 0] = -1

        new = mock.update(data=dd)

        data = filters.negative_power_filter(new)
        assert np.all(data.complete_flags[:, :, 0])


def test_rmsf_filter(gsd_ones: GSData):
    with pytest.raises(ValueError, match="Unsupported data_unit for rmsf_filter"):
        filters.rmsf_filter(gsd_ones.update(data_unit="power"), threshold=100)

    # Let data be perfect:
    rng = np.random.default_rng()
    pl = np.outer(
        1 + rng.normal(scale=0.1, size=gsd_ones.ntimes * gsd_ones.npols),
        (gsd_ones.freqs / (75 * un.MHz)) ** -2.5,
    )
    pl.shape = gsd_ones.data.shape

    data = gsd_ones.update(data=pl, data_unit="uncalibrated_temp")

    print(gsd_ones.data.shape, data.data.shape)
    new = filters.rmsf_filter(data, threshold=1)

    data = gsd_ones.update(data=(pl - 300) / 1000, data_unit="uncalibrated")
    new2 = filters.rmsf_filter(data, threshold=1, tcal=300, tload=1000)

    assert np.all(new.complete_flags == new2.complete_flags)


class TestRMSFilter:
    def test_basic(self, mock_with_model: GSData):
        run_filter_check(mock_with_model, filters.rms_filter, threshold=10)

    @pytest.mark.parametrize(
        "strategy",
        [
            NsamplesStrategy.FLAGGED_NSAMPLES,
            NsamplesStrategy.NSAMPLES_ONLY,
            NsamplesStrategy.FLAGGED_NSAMPLES_UNIFORM,
            NsamplesStrategy.FLAGS_ONLY,
        ],
    )
    def test_nsamples_strategy(self, mock: GSData, strategy: str):
        # since there are no flags in the mock data, each of the strategies
        # should work the same
        data = filters.rms_filter(
            mock, threshold=10, nsamples_strategy=strategy, model=mdl.LinLog(n_terms=5)
        )
        assert not np.any(data.complete_flags)

    def test_invalid_strategy(self, mock: GSData):
        with pytest.raises(ValueError, match="Invalid nsamples_strategy"):
            filters.rms_filter(
                mock,
                threshold=10,
                nsamples_strategy="invalid",
                model=mdl.LinLog(n_terms=5),
            )

    def test_no_residuals_or_model(self, mock: GSData):
        with pytest.raises(
            ValueError, match="Cannot perform rms_filter without residuals"
        ):
            filters.rms_filter(mock, threshold=10)

    def test_modelled_data(self, mock_with_model: GSData):
        data = filters.rms_filter(mock_with_model, threshold=10)
        assert not np.any(data.complete_flags)


class TestExplicitDayFilter:
    @pytest.mark.parametrize("days", [(2022, 11, 17), (2022, 321)])
    def test_single_day_flagged(self, mock_season: list[GSData], days):
        # this data has days 245990{0,1,2}
        data = concat(mock_season[:2], axis="time")

        new1 = filters.explicit_day_filter(
            data,
            flag_days=[(2022, 11, 17)],
        )

        new2 = filters.explicit_day_filter(
            data,
            flag_days=[(2022, 321)],
        )
        assert np.any(new1.complete_flags)
        assert not np.all(new1.complete_flags)
        assert np.any(new2.complete_flags)
        assert not np.all(new2.complete_flags)
        assert np.all(new1.complete_flags == new2.complete_flags)

    def test_flag_with_jd(self, mock_season):
        data = concat(mock_season[:2], axis="time")
        new = filters.explicit_day_filter(
            data,
            flag_days=[2459901],
        )
        assert np.all(new.complete_flags[:, :, data.ntimes // 2 :])
        assert not np.any(new.complete_flags[:, :, : data.ntimes // 2])


class TestPruneFlaggedIntegrations:
    def test_pruning(self, mock: GSData):
        f = np.zeros(mock.ntimes, dtype=bool)
        f[0] = True
        flags = GSFlag(f, axes=("time",))
        new = mock.add_flags("explicit", flags)
        new = filters.prune_flagged_integrations(new)
        assert new.ntimes == mock.ntimes - 1
