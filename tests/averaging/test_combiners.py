"""Tests for the combiners module."""

import numpy as np
import pytest
from edges_cal import modelling as mdl
from pygsdata import GSData

from edges_analysis.averaging import combiners, lstbin
from edges_analysis.datamodel import add_model


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
        linlog = mdl.LinLog(n_terms=5)
        modelled = [add_model(d, model=linlog) for d in mock_season]

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
