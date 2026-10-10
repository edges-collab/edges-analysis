"""Test the filters module."""

import logging

import deprecation
import numpy as np
import pytest
from astropy import units as un
from astropy.time import Time
from pygsdata import GSData, GSFlag
from pygsdata.concat import concat
from pygsdata.select import select_freqs, select_loads

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

    def test_multi_load_power(self):
        """Multi-load (power) data used to crash with a broadcasting error."""
        power = create_mock_edges_data(add_noise=True, as_power=True, ntime=6)
        fq = power.freqs.to_value("MHz")
        assert power.nloads == 3
        assert np.any((fq > 80) & (fq <= 200))

        out = filters.peak_power_filter(power)
        flg = out.flags["peak_power_filter"]
        assert flg.flags.shape == (power.nloads, power.npols, power.ntimes)
        assert not np.any(flg.flags)

        # A huge spike in one load and integration flags only that one.
        data = power.data.copy()
        ichan = np.argmin(np.abs(fq - 90))
        data[1, 0, 2, ichan] = 1e6 * np.mean(data[1, 0, 2])
        out = filters.peak_power_filter(power.update(data=data))
        expected = np.zeros((power.nloads, power.npols, power.ntimes), dtype=bool)
        expected[1, 0, 2] = True
        np.testing.assert_array_equal(out.flags["peak_power_filter"].flags, expected)

    def test_multi_load_matches_single_load(self):
        """The fix for multi-load data must not change single-load results."""
        power = create_mock_edges_data(add_noise=True, as_power=True, ntime=6)
        data = power.data.copy()
        fq = power.freqs.to_value("MHz")
        data[:, 0, 3, np.argmin(np.abs(fq - 95))] *= 1e5
        power = power.update(data=data)

        multi = filters._peak_power_filter(data=power, threshold=40.0)
        for iload in range(power.nloads):
            single = filters._peak_power_filter(
                data=select_loads(power, loads=[power.loads[iload]]), threshold=40.0
            )
            np.testing.assert_array_equal(multi[iload], single[0])


def _flat(value: float = 1000.0, ntime: int = 4) -> GSData:
    """Flat spectra over 40-200 MHz, to be perturbed by the tests."""
    mock = create_mock_edges_data(flow=40 * un.MHz, fhigh=200 * un.MHz, ntime=ntime)
    return mock.update(data=np.full(mock.data.shape, value))


def _set(data: GSData, itime: int, freq: float, value: float) -> GSData:
    """Set the value of the channel nearest ``freq`` (MHz) in one integration."""
    arr = data.data.copy()
    arr[..., itime, np.argmin(np.abs(data.freqs.to_value("MHz") - freq))] = value
    return data.update(data=arr)


class TestPeakOrbcommFilter:
    """The ORBCOMM peak-power filter follows the C-code's -pkpwrm cut."""

    def test_basic(self, mock):
        run_filter_check(mock, filters.peak_orbcomm_filter)

    @staticmethod
    def _flags(data: GSData, **kwargs) -> np.ndarray:
        out = filters.peak_orbcomm_filter(data, **kwargs)
        return out.flags["peak_orbcomm_filter"].flags.reshape(-1, data.ntimes)[0]

    def test_band_is_137p5_plus_minus_1mhz(self):
        data = _set(_flat(), 2, 136.7, 1e9)
        np.testing.assert_array_equal(
            self._flags(data, threshold=40.0), [False, False, True, False]
        )

    def test_unity_mean_without_low_channels(self):
        """With no channel below a tenth of the peak, the mean is unity.

        So the statistic is just the peak in dB: 30 dB for a flat 1000 K spectrum.
        """
        assert np.all(self._flags(_flat(1000.0), threshold=29.0))
        assert not np.any(self._flags(_flat(1000.0), threshold=31.0))

    def test_mean_cut_relative_to_peak_above_80mhz(self):
        """Channels above a tenth of the peak *above 80 MHz* are left out of the mean.

        A spike at 90 MHz (outside the ORBCOMM band) sets that peak, so the flat
        spectrum is in the mean and the ORBCOMM peak is 0 dB above it. The other
        (flat) spectra have no channels in the mean, so they are at 30 dB.
        """
        data = _set(_flat(1000.0), 1, 90.0, 1e5)
        np.testing.assert_array_equal(
            self._flags(data, threshold=10.0), [True, False, True, True]
        )

    def test_negative_threshold_keeps_only_peaky_spectra(self):
        """As in the C-code, a negative threshold flags spectra below its magnitude."""
        data = _set(_flat(1000.0), 2, 137.5, 1e9)  # 60 dB above the flat mean
        np.testing.assert_array_equal(
            self._flags(data, threshold=-40.0), [True, True, False, True]
        )

    def test_nonfinite_channels_ignored(self):
        """Undefined channels (e.g. 0/0 in Dicke calibration) count as zero."""
        data = _set(_flat(1000.0), 1, 137.5, np.inf)
        data = _set(data, 2, 137.5, np.nan)
        assert not np.any(self._flags(data, threshold=40.0))


class TestMaxFM:
    def test_basic_run(self, mock):
        run_filter_check(mock, filters.maxfm_filter, threshold=200)

    def test_no_fm(self, mock):
        new = select_freqs(mock, freq_range=(50 * un.MHz, 70 * un.MHz))
        out = filters.maxfm_filter(new)
        assert not np.any(out.complete_flags)

    @staticmethod
    def _flags(data: GSData, fnc=filters.maxfm_filter, **kwargs) -> np.ndarray:
        out = fnc(data, **kwargs)
        return out.flags[fnc.__name__].flags.reshape(-1, data.ntimes)[0]

    def test_only_positive_spikes(self):
        """As in the C-code, a dip is not a spike (its neighbours stick out by half)."""
        data = _set(_flat(1000.0), 1, 100.0, 500.0)
        assert not np.any(self._flags(data, threshold=300.0))
        # The generic filter flags both by default.
        flags = self._flags(
            data, fnc=filters.single_channel_spike_filter, threshold=300.0
        )
        np.testing.assert_array_equal(flags, [False, True, False, False])

    def test_band_edge_channel(self):
        """The channels at the edges of the band use their neighbours outside it."""
        freqs = _flat().freqs.to_value("MHz")
        first = freqs[freqs >= 88.0][0]
        data = _set(_flat(1000.0), 3, first, 1500.0)
        np.testing.assert_array_equal(
            self._flags(data, threshold=300.0), [False, False, False, True]
        )


class Test150MHzFilter:
    def test_basic_no_freqs(self, mock):
        run_filter_check(mock, filters.filter_150mhz, threshold=100)

    def test_with_freqs(self):
        mock = create_mock_edges_data(fhigh=200 * un.MHz)
        data = filters.filter_150mhz(mock, threshold=100)
        assert not np.any(data.complete_flags)

    def test_single_bad_integration(self):
        mock = create_mock_edges_data(fhigh=200 * un.MHz, ntime=10)
        band = (mock.freqs >= 152.75 * un.MHz) & (mock.freqs <= 154.25 * un.MHz)
        data = mock.data.copy()
        rng = np.random.default_rng(1)
        # Very large excess noise in one integration only: its ratio is ~10x the
        # clean integrations' ratio, both with and without the sqrt in the statistic.
        data[..., 3, band] += rng.normal(scale=100.0, size=band.sum())

        out = filters.filter_150mhz(mock.update(data=data), threshold=3)
        expected = np.zeros(mock.ntimes, dtype=bool)
        expected[3] = True
        np.testing.assert_array_equal(out.complete_flags[0, 0, :, 0], expected)

    def test_statistic_per_integration(self):
        mock = create_mock_edges_data(fhigh=200 * un.MHz, ntime=4)
        rng = np.random.default_rng(2)
        data = mock.data * (1 + 0.01 * rng.normal(size=mock.data.shape))
        band = (mock.freqs >= 152.75 * un.MHz) & (mock.freqs <= 154.25 * un.MHz)
        band2 = (mock.freqs >= 156.25 * un.MHz) & (mock.freqs <= 157.75 * un.MHz)
        stat = (
            200
            * np.sqrt(np.std(data[..., band], axis=-1))
            / np.mean(data[..., band2], axis=-1)
        )
        # A threshold between the per-integration statistics splits them.
        threshold = np.median(stat)
        out = filters.filter_150mhz(mock.update(data=data), threshold=threshold)
        np.testing.assert_array_equal(out.complete_flags[..., 0], stat > threshold)


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
        ("fnc", "method"),
        [
            (filters.rfi_iterative_filter, "iterative"),
            (filters.rfi_watershed_filter, "watershed"),
            (filters.rfi_iterative_sliding_window, "iterative_sliding_window"),
        ],
    )
    def test_docstring(self, fnc, method):
        assert fnc.__doc__ is not None
        assert f"xrfi_{method}" in fnc.__doc__

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


@deprecation.fail_if_not_removed
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

    filters.rmsf_filter(data, threshold=1)


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

    def test_all_outside_range(self, mock):
        new = filters.rms_filter(
            mock, threshold=10, freq_range=(200 * un.MHz, 250 * un.MHz)
        )
        assert not np.any(new.complete_flags)


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

    @pytest.fixture(scope="class")
    def across_utc_midnight(self) -> GSData:
        """Data running from 2022-11-17 20:00 to 2022-11-18 ~04:00 UTC."""
        t0 = Time("2022-11-17 20:00:00", scale="utc").jd
        return create_mock_edges_data(time0=t0, ntime=48, dt=600 * un.s).update(
            auxiliary_measurements=None
        )

    @pytest.mark.parametrize(
        "day",
        [
            (2022, 11, 18),
            (2022, 322),
            Time("2022-11-18 13:00:00", scale="utc"),
            2459902,  # The Julian Day Number of 2022-11-18 (JD at noon).
            np.int64(2459902),
        ],
        ids=["ymd", "yday", "Time", "int", "np.int64"],
    )
    def test_flags_utc_calendar_day(self, across_utc_midnight: GSData, day):
        """A requested day flags 00:00 -> 24:00 UTC (not noon to noon)."""
        new = filters.explicit_day_filter(across_utc_midnight, flag_days=[day])

        after_midnight = across_utc_midnight.times.jd[:, 0] >= 2459901.5
        assert 0 < np.sum(after_midnight) < across_utc_midnight.ntimes
        np.testing.assert_array_equal(
            new.flags["explicit_day_filter"].flags, after_midnight
        )

    def test_multiple_days_and_time_array(self, across_utc_midnight: GSData):
        new = filters.explicit_day_filter(
            across_utc_midnight,
            flag_days=[Time(["2022-11-17 21:00", "2022-11-18 01:00"], scale="utc")],
        )
        assert np.all(new.flags["explicit_day_filter"].flags)

    def test_does_not_mutate_input(self, across_utc_midnight: GSData):
        flag_days = [(2022, 11, 18), Time("2022-11-17 21:00", scale="utc")]
        orig = list(flag_days)
        filters.explicit_day_filter(across_utc_midnight, flag_days=flag_days)
        assert len(flag_days) == len(orig)
        assert all(a is b for a, b in zip(flag_days, orig, strict=True))

    @pytest.mark.parametrize("day", [(2022, 11, 18, 1), 2459902.0, "2022-11-18"])
    def test_bad_entries(self, across_utc_midnight: GSData, day):
        with pytest.raises(ValueError, match="flag_days"):
            filters.explicit_day_filter(across_utc_midnight, flag_days=[day])


class TestPruneFlaggedIntegrations:
    def test_pruning(self, mock: GSData):
        f = np.zeros(mock.ntimes, dtype=bool)
        f[0] = True
        flags = GSFlag(f, axes=("time",))
        new = mock.add_flags("explicit", flags)
        new = filters.prune_flagged_integrations(new)
        assert new.ntimes == mock.ntimes - 1


class TestGalaxyFilter:
    def test_basic(self, mock):
        run_filter_check(
            mock,
            filters.galaxy_filter,
        )


class TestFilterLogging:
    def test_percentages_in_log(self, gsd_ones: GSData, caplog):
        """The summary log line reports every number as a percentage."""
        flags = np.zeros(gsd_ones.ntimes, dtype=bool)
        flags[0] = True  # 1 of 10 integrations, i.e. 10% of the data
        with caplog.at_level(logging.INFO, logger=filters.logger.name):
            filters.apply_flags(gsd_ones, flags=GSFlag(flags=flags, axes=("time",)))
        assert "0.00% + 10.00% → 10.00%" in caplog.text
        assert "<+10.00%>" in caplog.text
