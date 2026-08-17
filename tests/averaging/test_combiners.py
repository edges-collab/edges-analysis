"""Tests for the combiners module."""

import numpy as np
import pytest
from pygsdata import GSData, GSFlag

from edges.averaging import combiners, lstbin
from edges.averaging.utils import NsamplesStrategy


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

    def test_model_mean_uses_same_nights_as_residuals(self, mock_season: list[GSData]):
        """Flagged nights must not contribute their model at that channel."""
        day_a, day_b = mock_season[:2]
        model = np.full_like(day_a.data, 100.0)
        resid = np.zeros_like(day_a.data)

        model_a = model.copy()
        model_a[..., 1] = 200.0
        resid_a = resid.copy()
        resid_a[..., 1] = np.nan
        flags_a = np.zeros(day_a.data.shape[-1], dtype=bool)
        flags_a[1] = True
        day_a = day_a.update(
            data=model_a + np.where(np.isnan(resid_a), 0.0, resid_a),
            residuals=resid_a,
            flags={"chan": GSFlag(flags=flags_a, axes=("freq",))},
        )
        day_b = day_b.update(data=model.copy(), residuals=resid.copy(), flags={})

        avg = combiners.average_multiple_objects(
            day_a,
            day_b,
            use_resids=True,
            nsamples_strategy=NsamplesStrategy.FLAGGED_NSAMPLES_UNIFORM,
        )

        np.testing.assert_allclose(avg.data[..., 1], 100.0)
        np.testing.assert_allclose(avg.residuals[..., 1], 0.0)
        np.testing.assert_allclose(avg.data, avg.model + avg.residuals)

    def test_zero_weight_bins_are_fill_value(self, mock_season: list[GSData]):
        day = mock_season[0].update(
            data=np.full_like(mock_season[0].data, 50.0),
            residuals=np.full_like(mock_season[0].data, np.nan),
            flags={
                "all": GSFlag(
                    flags=np.ones(mock_season[0].ntimes, dtype=bool), axes=("time",)
                )
            },
        )
        avg = combiners.average_multiple_objects(
            day,
            use_resids=True,
            nsamples_strategy=NsamplesStrategy.FLAGGED_NSAMPLES_UNIFORM,
        )
        assert np.all(np.isnan(avg.data))
        assert np.all(np.isnan(avg.residuals))

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
