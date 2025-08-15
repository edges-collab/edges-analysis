"""Test the averaging module."""

import numpy as np
import pytest
from astropy import units as un

from edges.averaging import averaging
from edges.averaging.averaging import bin_data


class TestGetBinnedWeights:
    def test_empty_input(self):
        x = np.array([])
        bins = np.array([])

        with pytest.raises(ValueError, match="Bin edges must have at least 2 elements"):
            averaging.get_binned_weights(x=x, bins=bins)

    def test_single_bin(self):
        x = np.linspace(0, 1, 10)
        bins = np.array([0, 1])
        weights = np.ones_like(x)

        result = averaging.get_binned_weights(x, bins, weights)

        assert result.shape == (1,)
        assert np.isclose(result[0], 10)

    def test_default_weights(self):
        x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        bins = np.array([0, 0.5, 1])

        result = averaging.get_binned_weights(x=x, bins=bins)

        expected = np.array([2, 3])
        assert np.array_equal(result, expected)

    def test_outside_range(self):
        x = np.array([-1, 0, 1, 2, 3, 4, 5])
        bins = np.array([0, 2, 4])
        weights = np.ones_like(x)

        result = averaging.get_binned_weights(x=x, bins=bins, weights=weights)

        expected = np.array([3, 4])
        assert np.array_equal(result, expected)

    def test_non_monotonic_bins(self):
        x = np.array([1, 2, 3, 4, 5])
        bins = np.array([0, 3, 2, 5])  # Non-monotonic bin edges
        weights = np.array([1, 2, 3, 4, 5])

        with pytest.raises(
            ValueError, match="Bin edges must be monotonically increasing"
        ):
            averaging.get_binned_weights(x=x, bins=bins, weights=weights)

    def test_with_nan_and_inf(self):
        x = np.array([0.1, 0.3, np.nan, 0.7, np.inf])
        bins = np.array([0, 0.5, 1])
        weights = np.array([1, 2, 3, 4, 5])

        result = averaging.get_binned_weights(x=x, bins=bins, weights=weights)

        expected = np.array([3, 4])  # NaN and inf values should be excluded
        np.testing.assert_array_equal(result, expected)

    def test_negative_weights(self):
        x = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        bins = np.array([0, 1, 2, 3, 4, 5])
        weights = np.array([-1, 2, -3, 4, -5])

        result = averaging.get_binned_weights(x=x, bins=bins, weights=weights)

        expected = np.array([-1, 2, -3, 4, -5])
        assert np.array_equal(result, expected)

    def test_zero_width_bins(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        bins = np.array([0.1, 0.1, 0.3, 0.3, 0.5])
        weights = np.ones_like(x)

        with pytest.raises(
            ValueError, match="Bin edges must be monotonically increasing"
        ):
            averaging.get_binned_weights(x=x, bins=bins, weights=weights)

    def test_1d(self):
        w = np.ones(20)
        x = np.linspace(0, 1, 20)
        bins = [-0.1, 1.1]

        wghts = averaging.get_binned_weights(x=x, bins=bins, weights=w)
        assert wghts == 20

    def test_2d(self):
        w = np.ones((10, 20))
        x = np.linspace(0, 1, 20)
        bins = [-0.1, 1.1]

        wghts = averaging.get_binned_weights(x=x, bins=bins, weights=w)
        assert wghts.shape == (10, 1)
        assert all(w == 20 for w in wghts)

    def test_with_more_bins(self):
        w = np.ones((10, 20))
        x = np.linspace(0, 1, 20, endpoint=False)
        bins = np.linspace(0, 1, 11)

        wghts = averaging.get_binned_weights(x=x, bins=bins, weights=w)
        assert wghts.shape == (10, 10)
        assert np.allclose(wghts, 2)


class TestGetBinEdges:
    def test_get_bin_edges_with_astropy_quantity(self):
        coords = np.array([1, 2, 3, 4, 5]) * un.MHz
        bins = 1 * un.MHz
        expected = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]) * un.MHz

        result = averaging.get_bin_edges(coords, bins)

        assert isinstance(result, un.Quantity)
        assert result.unit == un.MHz
        np.testing.assert_allclose(result.value, expected.value)

    def test_get_bin_edges_with_quantity_start_stop(self):
        coords = np.array([1, 2, 3, 4, 5]) * un.MHz
        start = 0.5 * un.MHz
        stop = 5.5 * un.MHz
        bins = 2.0

        result = averaging.get_bin_edges(coords, bins=bins, start=start, stop=stop)

        expected = np.array([0.5, 2.5, 4.5, 6.5]) * un.MHz
        assert isinstance(result, un.Quantity)
        assert result.unit == un.MHz
        np.testing.assert_allclose(result.value, expected.value)

    def test_coords_not_monotonically_increasing(self):
        coords = np.array([0, 1, 2, 1.5, 3])
        bins = 2

        with pytest.raises(
            ValueError, match="coords must be monotonically increasing!"
        ):
            averaging.get_bin_edges(coords, bins)

    def test_not_regularly_spaced(self):
        coords = np.array([0, 1, 3, 6, 10])

        with pytest.raises(ValueError, match="coords must be regularly spaced!"):
            averaging.get_bin_edges(coords)

    def test_get_bin_edges_float_bins(self):
        coords = np.array([0, 1, 2, 3, 4, 5])
        bins = 0.5
        expected = np.array([-0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5])
        result = averaging.get_bin_edges(coords, bins)

        assert np.allclose(result, expected)

    def test_get_bin_edges_large_values(self):
        large_coords = np.arange(1000) * 1e7 + 1e15
        bins = 10

        result = averaging.get_bin_edges(large_coords, bins)

        assert len(result) == 101
        assert np.isclose(result[0], large_coords[0] - 5e6, rtol=1e-10)
        assert np.isclose(result[-1], large_coords[-1] + 5e6, rtol=1e-10)
        assert np.allclose(np.diff(result), 1e8, rtol=1e-10)

    def test_empty_coords(self):
        coords = np.array([])
        with pytest.raises(ValueError, match="coords must have at least 2 elements"):
            averaging.get_bin_edges(coords)

    def test_get_bin_edges_negative_coords(self):
        coords = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
        bins = 3
        expected = np.array([-5.5, -2.5, 0.5, 3.5])
        result = averaging.get_bin_edges(coords, bins)
        assert np.allclose(result, expected)

    def test_get_bin_edges_integer_bins_uneven(self):
        coords = np.linspace(0, 10, 11)  # 11 elements
        bins = 3  # This will result in len(coords) % bins != 0

        expected = np.array([0, 3, 6, 9]) - 0.5
        result = averaging.get_bin_edges(coords, bins)

        np.testing.assert_allclose(result, expected)


class TestWeightedSum:
    def test_weighted_sum_3d(self):
        data = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        weights = np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])

        result, sum_weights = averaging.weighted_sum(data, weights, axis=2)

        expected_result = np.array([[6, 30], [72, 132]])
        expected_sum_weights = np.array([[3, 6], [9, 12]])

        np.testing.assert_array_equal(result, expected_result)
        np.testing.assert_array_equal(sum_weights, expected_sum_weights)

    def test_weighted_sum_all_zero_weights(self):
        data = np.array([1, 2, 3, 4, 5])
        weights = np.zeros_like(data)
        expected_weights = 0

        result_sum, result_weights = averaging.weighted_sum(data, weights)

        assert np.isnan(result_sum)
        assert result_weights == expected_weights

    def test_weighted_sum_normalize(self):
        data = np.array([1, 2, 3, 4, 5])
        weights = np.array([2, 4, 6, 8, 10])

        result, sum_weights = averaging.weighted_sum(data, weights, normalize=True)

        expected_weights = weights / weights.max()
        expected_sum = np.sum(data * expected_weights)
        expected_sum_weights = np.sum(expected_weights)

        np.testing.assert_allclose(result, expected_sum)
        np.testing.assert_allclose(sum_weights, expected_sum_weights)

    def test_weighted_sum_non_default_axis(self):
        data = np.array([[1, 2, 3], [4, 5, 6]])
        weights = np.array([[1, 2, 3], [4, 5, 6]])

        result, sum_weights = averaging.weighted_sum(data, weights, axis=0)

        expected_result = np.array([17, 29, 45])
        expected_sum_weights = np.array([5, 7, 9])

        np.testing.assert_array_almost_equal(result, expected_result)
        np.testing.assert_array_almost_equal(sum_weights, expected_sum_weights)

    def test_weighted_sum_keepdims(self):
        data = np.array([[1, 2, 3], [4, 5, 6]])
        weights = np.array([[1, 1, 1], [2, 2, 2]])

        result_with_keepdims, weights_with_keepdims = averaging.weighted_sum(
            data, weights, axis=1, keepdims=True
        )
        result_without_keepdims, weights_without_keepdims = averaging.weighted_sum(
            data, weights, axis=1, keepdims=False
        )

        assert result_with_keepdims.shape == (2, 1)
        assert weights_with_keepdims.shape == (2, 1)
        assert result_without_keepdims.shape == (2,)
        assert weights_without_keepdims.shape == (2,)

        np.testing.assert_array_equal(
            result_with_keepdims, result_without_keepdims[:, np.newaxis]
        )
        np.testing.assert_array_equal(
            weights_with_keepdims, weights_without_keepdims[:, np.newaxis]
        )

    def test_weighted_sum_with_infinity(self):
        data = np.array([1, 2, np.inf, 4, 5])
        weights = np.array([1, 1, 1, 1, 1])

        result, sum_weights = averaging.weighted_sum(data, weights)

        assert np.isinf(result)
        assert np.isclose(sum_weights, 5)

        # Test with normalize=True
        result_normalized, sum_weights_normalized = averaging.weighted_sum(
            data, weights, normalize=True
        )

        assert np.isinf(result_normalized)
        assert np.isclose(sum_weights_normalized, 5)

    def test_weighted_sum_mismatched_shapes(self):
        data = np.array([[1, 2, 3], [4, 5, 6]])
        weights = np.array([1, 2, 3, 4])

        with pytest.raises(
            ValueError, match="data and weights must have the same shape"
        ):
            averaging.weighted_sum(data, weights)


class TestWeightedMean:
    def test_weighted_mean_all_zero_weights(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.zeros_like(data)

        result, out_weights = averaging.weighted_mean(data, weights)

        assert np.all(np.isnan(result))
        assert np.all(out_weights == 0)

    def test_weighted_mean_with_nan(self):
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        result, result_weights = averaging.weighted_mean(data, weights)

        expected_result = np.array(3.0)
        expected_weights = np.array(4.0)

        assert np.isclose(result, expected_result)
        assert np.isclose(result_weights, expected_weights)

    def test_weighted_mean_with_infinite_values(self):
        data = np.array([1.0, 2.0, np.inf, 4.0, 5.0])
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        result, result_weights = averaging.weighted_mean(data, weights)

        assert np.isinf(result)
        assert np.isclose(result_weights, 5.0)

        # Test with infinite weight
        weights[1] = np.inf
        with pytest.warns(
            RuntimeWarning, match="invalid value encountered in scalar divide"
        ):
            result, result_weights = averaging.weighted_mean(data, weights)

        assert np.isnan(result)
        assert np.isinf(result_weights)

    def test_weighted_mean_multi_dimensional(self):
        data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

        expected_avg = np.array([[3, 4], [5, 6]])
        expected_weights = np.array([[2, 2], [2, 2]])

        avg, total_weights = averaging.weighted_mean(data, axis=0)

        np.testing.assert_array_almost_equal(avg, expected_avg)
        np.testing.assert_array_almost_equal(total_weights, expected_weights)

    def test_with_axis_0(self):
        data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        weights = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

        avg, total_weights = averaging.weighted_mean(data, weights, axis=0)
        avg1, total_weights1 = averaging.weighted_mean(data.T, weights.T)

        np.testing.assert_almost_equal(avg1.T, avg)
        np.testing.assert_almost_equal(total_weights1.T, total_weights)

    def test_weighted_mean_keepdims(self):
        data = np.array([[1, 2, 3], [4, 5, 6]])
        weights = np.array([[1, 1, 1], [2, 2, 2]])

        # Test with keepdims=False
        avg_false, weights_false = averaging.weighted_mean(
            data, weights, axis=1, keepdims=False
        )
        assert avg_false.shape == (2,)
        assert weights_false.shape == (2,)
        assert np.allclose(avg_false, np.array([2, 5]))
        assert np.allclose(weights_false, np.array([3, 6]))

        # Test with keepdims=True
        avg_true, weights_true = averaging.weighted_mean(
            data, weights, axis=1, keepdims=True
        )
        assert avg_true.shape == (2, 1)
        assert weights_true.shape == (2, 1)
        assert np.allclose(avg_true, np.array([[2], [5]]))
        assert np.allclose(weights_true, np.array([[3], [6]]))

    def test_with_custom_fill_value(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.zeros_like(data)

        result, _ = averaging.weighted_mean(data, weights, fill_value=0)
        assert result == 0


class TestWeightedVariance:
    # Calculate weighted variance with provided data and nsamples
    def test_weighted_variance_with_samples(self):
        data = np.array([1.0, 2.0, 3.0, 4.0])
        nsamples = np.array([2, 1, 2, 1])

        variance = averaging.weighted_variance(data, nsamples)

        expected_variance = 1.177777777
        np.testing.assert_almost_equal(variance, expected_variance, decimal=5)

        # Handle empty arrays for data input

    def test_weighted_variance_empty_array(self):
        data = np.array([])
        nsamples = np.array([])

        variance = averaging.weighted_variance(data, nsamples)

        assert np.isnan(variance)

    def test_with_pregenerated_average(self):
        data = np.array([1.0, 2.0, 3.0, 4.0])
        nsamples = np.array([2, 1, 2, 1])
        avg, _ = averaging.weighted_mean(data, nsamples, keepdims=True)

        variance = averaging.weighted_variance(data, nsamples)
        print(data.shape, nsamples.shape, avg.shape)
        variance1 = averaging.weighted_variance(data, nsamples, avg=avg)
        np.testing.assert_equal(variance, variance1)


class TestBinData:
    @pytest.mark.parametrize("axis", [0, 1, 2, -1])
    def test_full_average_3d(self, axis):
        rng = np.random.default_rng()
        data = rng.random(size=(3, 4, 5))
        res, _, _ = bin_data(data, axis=axis)
        np.testing.assert_array_almost_equal(
            res, np.mean(data, axis=axis, keepdims=True)
        )

    def test_weights_uniform(self):
        rng = np.random.default_rng()
        data = rng.random(size=(3, 4, 5))
        weights = np.ones_like(data)
        res1, w1, _ = bin_data(data, weights=weights)
        res2, w2, _ = bin_data(data, weights=weights * 2)
        np.testing.assert_array_almost_equal(res1, res2)
        np.testing.assert_array_almost_equal(w1 * 2, w2)

    def test_weights_non_uniform(self):
        data = np.array([2, 2, 3, 4], dtype=float)
        weights = np.array([1, 1, 2, 2])
        res1, w1, _ = bin_data(data, weights=weights)
        assert res1 == 3
        assert w1 == 6

    def test_with_residuals_uniform_weights(self):
        data = np.array([2, 2, 3, 4], dtype=float)
        rng = np.random.default_rng()
        residuals = rng.normal(size=data.shape)
        res1, _, _ = bin_data(data, residuals=residuals)
        res2, _, _ = bin_data(data)

        np.testing.assert_allclose(res1, res2)

    def test_with_multiple_bins(self):
        data = np.array([1, 2, 3, 4], dtype=float)
        bins = [slice(0, 2), slice(2, 4)]
        res1, _, _ = bin_data(data, bins=bins)
        np.testing.assert_array_equal(res1, [1.5, 3.5])


class TestBinArrayUnweighted:
    def test_bin_array_simple_1d(self):
        x = np.array([1, 1, 2, 2, 3, 3])
        out = averaging.bin_array_unweighted(x, 2)
        assert np.all(out == np.array([1, 2, 3]))

    def test_bin_array_remainder(self):
        x = np.array([1, 1, 2, 2, 3, 3, 4])
        out = averaging.bin_array_unweighted(x, 2)
        assert np.all(out == np.array([1, 2, 3]))

    def test_bin_array_2d(self):
        x = np.array([
            [1, 1, 2, 2, 3, 3, 4],
            [4, 4, 5, 5, 6, 6, 7],
        ])
        out = averaging.bin_array_unweighted(x, 2)
        assert np.all(out == np.array([[1, 2, 3], [4, 5, 6]]))
