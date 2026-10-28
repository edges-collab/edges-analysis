"""Tests for the combiners module."""

import numpy as np
import pytest
from astropy import units as un
from pygsdata import GSData, GSFlag

from edges import modeling as mdl
from edges.analysis.datamodel import add_model
from edges.averaging import NsamplesStrategy, combiners, lstbin
from edges.testing import create_mock_edges_data


@pytest.fixture(scope="module")
def small_season() -> list[GSData]:
    """A small three-day mock season (few times and channels, so IO is fast)."""
    return [
        create_mock_edges_data(
            add_noise=True, ntime=4, fhigh=55 * un.MHz, time0=2459900.27 + i
        ).update(auxiliary_measurements=None)
        for i in range(3)
    ]


def _write_all(objs: list[GSData], tmp_path) -> list:
    files = [tmp_path / f"tmp.{i}.gsh5" for i in range(len(objs))]
    for d, f in zip(objs, files, strict=True):
        d.write_gsh5(f)
    return files


class TestAverageMultipleObjects:
    @pytest.mark.parametrize(
        "strategy", list(lstbin.NsamplesStrategy.__members__.values())
    )
    def test_combine_season(
        self, mock_season: list[GSData], strategy: lstbin.NsamplesStrategy
    ):
        new = combiners.average_multiple_objects(
            *mock_season, nsamples_strategy=strategy
        )
        assert new.data.shape == mock_season[0].data.shape

    def test_use_resids(self, mock_season: list[GSData]):
        modelled = [d.update(residuals=np.zeros_like(d.data)) for d in mock_season]
        new = combiners.average_multiple_objects(*modelled, use_resids=True)
        new2 = combiners.average_multiple_objects(
            *modelled
        )  # default True when they exist

        assert new == new2

    def test_nan_model_raises(self, mock_season: list[GSData]):
        modelled = [d.update(model=d.data.copy()) for d in mock_season]
        data = modelled[0].data.copy()
        data[..., 0, 4] = np.nan
        # Residuals given for NaN data lose the model there.
        modelled[0] = modelled[0].update(data=data, residuals=modelled[0].residuals)
        with pytest.raises(ValueError, match="model of an object is NaN"):
            combiners.average_multiple_objects(*modelled, use_resids=True)

    def test_bad_inputs(self, mock_season, mock):
        with pytest.raises(ValueError, match="All objects must have the same shape"):
            combiners.average_multiple_objects(mock_season[0], mock)

        with pytest.raises(
            ValueError, match="One or more of the input objects has no residuals"
        ):
            combiners.average_multiple_objects(*mock_season, use_resids=True)


class TestAverageFilesPairwise:
    def test_simple(self, mock_season: list[GSData], tmp_path):
        for i, d in enumerate(mock_season):
            d.write_gsh5(tmp_path / f"tmp.{i}.gsh5")

        newread = combiners.average_files_pairwise(*[
            tmp_path / f"tmp.{i}.gsh5" for i in range(len(mock_season))
        ])
        new = combiners.average_multiple_objects(*mock_season)

        np.testing.assert_array_almost_equal(newread.nsamples, new.nsamples)
        np.testing.assert_array_almost_equal(newread.data, new.data)

    def test_with_residuals(self, small_season: list[GSData], tmp_path):
        modelled = [add_model(d, model=mdl.LinLog(n_terms=2)) for d in small_season]
        files = _write_all(modelled, tmp_path)

        pairwise = combiners.average_files_pairwise(*files)
        direct = combiners.average_multiple_objects(*modelled)

        np.testing.assert_allclose(pairwise.nsamples, direct.nsamples, rtol=1e-10)
        np.testing.assert_allclose(pairwise.data, direct.data, rtol=1e-10)

    @pytest.mark.parametrize(
        "strategy", list(lstbin.NsamplesStrategy.__members__.values())
    )
    def test_all_strategies(self, small_season: list[GSData], tmp_path, strategy):
        files = _write_all(small_season, tmp_path)

        pairwise = combiners.average_files_pairwise(*files, nsamples_strategy=strategy)
        direct = combiners.average_multiple_objects(
            *small_season, nsamples_strategy=strategy
        )
        np.testing.assert_allclose(pairwise.data, direct.data, rtol=1e-10)

    @pytest.mark.parametrize(
        "strategy", list(lstbin.NsamplesStrategy.__members__.values())
    )
    def test_with_residuals_all_strategies(
        self, small_season: list[GSData], tmp_path, strategy
    ):
        rng = np.random.default_rng(11)
        modelled = [
            add_model(
                d.update(nsamples=rng.uniform(0.5, 2, size=d.nsamples.shape)),
                model=mdl.LinLog(n_terms=2),
            )
            for d in small_season
        ]
        files = _write_all(modelled, tmp_path)

        pairwise = combiners.average_files_pairwise(*files, nsamples_strategy=strategy)
        direct = combiners.average_multiple_objects(
            *modelled, nsamples_strategy=strategy
        )
        np.testing.assert_allclose(pairwise.data, direct.data, rtol=1e-10)
        np.testing.assert_allclose(pairwise.residuals, direct.residuals, atol=1e-10)
        np.testing.assert_allclose(pairwise.nsamples, direct.nsamples, rtol=1e-10)

    def test_mixed_residuals_default(self, small_season: list[GSData], tmp_path):
        # If any file lacks residuals, the default is to average the data directly.
        modelled = [add_model(d, model=mdl.LinLog(n_terms=2)) for d in small_season[:2]]
        objs = [*modelled, small_season[2]]
        files = _write_all(objs, tmp_path)

        pairwise = combiners.average_files_pairwise(*files)
        direct = combiners.average_multiple_objects(*objs)
        assert pairwise.residuals is None
        np.testing.assert_allclose(pairwise.data, direct.data, rtol=1e-10)

    def test_single_file(self, small_season: list[GSData], tmp_path):
        files = _write_all(small_season[:1], tmp_path)
        pairwise = combiners.average_files_pairwise(*files)
        direct = combiners.average_multiple_objects(small_season[0])
        np.testing.assert_allclose(pairwise.data, direct.data, rtol=1e-10)

    def test_error_names_correct_file(
        self, small_season: list[GSData], mock: GSData, tmp_path
    ):
        files = _write_all(
            [*small_season[:2], mock.update(auxiliary_measurements=None)], tmp_path
        )
        with pytest.raises(ValueError, match=r"same shape.*File 2"):
            combiners.average_files_pairwise(*files)

    def test_missing_residuals_error(self, small_season: list[GSData], tmp_path):
        modelled = add_model(small_season[0], model=mdl.LinLog(n_terms=2))
        files = _write_all([modelled, small_season[1]], tmp_path)
        with pytest.raises(ValueError, match=r"has no residuals.*File 1"):
            combiners.average_files_pairwise(*files, use_resids=True)

    def test_no_files(self):
        with pytest.raises(ValueError, match="At least one file"):
            combiners.average_files_pairwise()


def _flag_channel(obj: GSData, ichan: int) -> GSData:
    """Flag one frequency channel of an object."""
    fl = np.zeros(obj.nfreqs, dtype=bool)
    fl[ichan] = True
    return obj.update(flags={**obj.flags, "chan": GSFlag(flags=fl, axes=("freq",))})


@pytest.fixture(scope="module")
def modelled_nights(small_season: list[GSData]) -> list[GSData]:
    """Three nights, each with a smooth model fitted, as in a field pipeline."""
    return [add_model(d, model=mdl.LinLog(n_terms=3)) for d in small_season]


class TestEmptyChannels:
    """Channels to which no object contributes weight (see PR #344)."""

    @pytest.mark.parametrize("use_resids", [True, False])
    def test_empty_channel_is_fill_value(self, modelled_nights, use_resids):
        nights = [_flag_channel(d, 2) for d in modelled_nights]
        avg = combiners.average_multiple_objects(*nights, use_resids=use_resids)
        assert np.all(np.isnan(avg.data[..., 2]))
        assert np.all(avg.nsamples[..., 2] == 0)
        assert np.all(np.isfinite(avg.data[..., 3]))

        avg0 = combiners.average_multiple_objects(
            *nights, use_resids=use_resids, fill_value=0.0
        )
        np.testing.assert_array_equal(avg0.data[..., 2], 0.0)

    def test_empty_channel_keeps_mean_model(self, modelled_nights):
        nights = [_flag_channel(d, 2) for d in modelled_nights]
        avg = combiners.average_multiple_objects(*nights, use_resids=True)
        expected = np.mean([d.model[..., 2] for d in nights], axis=0)
        np.testing.assert_allclose(avg.model[..., 2], expected)
        assert np.all(np.isnan(avg.residuals[..., 2]))

    def test_unknown_model_is_nan_not_zero(self, modelled_nights):
        """An empty channel where no object has a model gets a NaN model, not 0.

        The zero (an empty sum) looked like a real model of 0 K when the average was
        averaged again, and pulled down the mean model there ("icicles").
        """
        nights = []
        for d in modelled_nights:
            data = d.data.copy()
            data[..., 2] = np.nan  # e.g. an empty channel of a smoothed spectrum
            nsamples = d.nsamples.copy()
            nsamples[..., 2] = 0
            # residuals given for NaN data: the model is lost there (as in old files)
            nights.append(d.update(data=data, nsamples=nsamples, residuals=d.residuals))
        assert np.all(np.isnan(nights[0].model[..., 2]))
        avg = combiners.average_multiple_objects(*nights, use_resids=True)
        assert np.all(np.isnan(avg.model[..., 2]))
        assert np.all(np.isnan(avg.data[..., 2]))

        # Averaging it again with an object that has data there fails loudly, rather
        # than treating the unknown model as 0 K.
        with pytest.raises(ValueError, match="model of an object is NaN"):
            combiners.average_multiple_objects(avg, modelled_nights[0], use_resids=True)

    def test_model_mean_includes_flagged_objects(self, modelled_nights):
        """The B18 estimator (EDGES memo 183).

        The unweighted mean of all the objects' models plus the weighted mean of the
        residuals, including objects whose data are flagged in the channel.
        """
        a = _flag_channel(modelled_nights[0], 2)
        b = modelled_nights[1]
        avg = combiners.average_multiple_objects(
            a, b, use_resids=True, nsamples_strategy=NsamplesStrategy.FLAGGED_NSAMPLES
        )
        expected = 0.5 * (a.model[..., 2] + b.model[..., 2]) + b.residuals[..., 2]
        np.testing.assert_allclose(avg.data[..., 2], expected)

    @pytest.mark.parametrize("use_resids", [True, False])
    def test_pairwise_matches(self, modelled_nights, tmp_path, use_resids):
        nights = [_flag_channel(d, 2) for d in modelled_nights]
        files = _write_all(nights, tmp_path)
        a = combiners.average_files_pairwise(*files, use_resids=use_resids)
        b = combiners.average_multiple_objects(*nights, use_resids=use_resids)
        np.testing.assert_allclose(a.data, b.data, equal_nan=True)
        np.testing.assert_allclose(a.nsamples, b.nsamples)
