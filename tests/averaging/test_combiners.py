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

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "FLT-4: the pairwise fold is only correct for additive weights; with "
            "residuals the model ends up weighted (mA+mB)/4 + mC/2; fix pending "
            "(result-changing)"
        ),
    )
    def test_with_residuals(self, small_season: list[GSData], tmp_path):
        modelled = [add_model(d, model=mdl.LinLog(n_terms=2)) for d in small_season]
        files = _write_all(modelled, tmp_path)

        pairwise = combiners.average_files_pairwise(*files)
        direct = combiners.average_multiple_objects(*modelled)

        np.testing.assert_allclose(pairwise.nsamples, direct.nsamples, rtol=1e-10)
        np.testing.assert_allclose(pairwise.data, direct.data, rtol=1e-10)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "FLT-4: with non-additive weight strategies (FLAGS_ONLY, uniform) the "
            "pairwise fold gives the first files too little weight; fix pending "
            "(result-changing)"
        ),
    )
    @pytest.mark.parametrize(
        "strategy",
        [
            lstbin.NsamplesStrategy.FLAGS_ONLY,
            lstbin.NsamplesStrategy.FLAGGED_NSAMPLES_UNIFORM,
            lstbin.NsamplesStrategy.UNFLAGGED_UNIFORM,
        ],
    )
    def test_non_additive_strategies(
        self, small_season: list[GSData], tmp_path, strategy
    ):
        files = _write_all(small_season, tmp_path)

        pairwise = combiners.average_files_pairwise(*files, nsamples_strategy=strategy)
        direct = combiners.average_multiple_objects(
            *small_season, nsamples_strategy=strategy
        )
        np.testing.assert_allclose(pairwise.data, direct.data, rtol=1e-10)
