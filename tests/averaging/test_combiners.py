"""Tests for the combiners module."""

import numpy as np
import pytest
from astropy import units as un
from pygsdata import GSData

from edges import modeling as mdl
from edges.analysis.datamodel import add_model
from edges.averaging import combiners, lstbin
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
