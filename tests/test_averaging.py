import numpy as np
from edges_cal import modelling as mdl
import pytest
from pytest_cases import fixture_ref as fxref
from pytest_cases import parametrize

from edges_analysis.analysis import averaging


def test_get_binned_weights_1d():
    w = np.ones(20)
    x = np.linspace(0, 1, 20)
    bins = [-0.1, 1.1]

    wghts = averaging.get_binned_weights(x=x, bins=bins, weights=w)
    assert wghts == 20


def test_get_binned_weights_2d():
    w = np.ones((10, 20))
    x = np.linspace(0, 1, 20)
    bins = [-0.1, 1.1]

    wghts = averaging.get_binned_weights(x=x, bins=bins, weights=w)
    assert wghts.shape == (10, 1)
    assert all(w == 20 for w in wghts)


def test_get_binned_weights_more_bins():
    w = np.ones((10, 20))
    x = np.linspace(0, 1, 20, endpoint=False)
    bins = np.linspace(0, 1, 11)

    wghts = averaging.get_binned_weights(x=x, bins=bins, weights=w)
    assert wghts.shape == (10, 10)
    assert np.allclose(wghts, 2)


FREQ = np.linspace(50, 100, 500)
LINLOG = mdl.LinLog(n_terms=5).at(x=FREQ)
N_GHA = 50
FID_PARAMS = np.array([2500, 0, 0, 5, 1])


@pytest.fixture(scope="module")
def fid_params():
    return np.array([FID_PARAMS] * N_GHA)


@pytest.fixture(scope="module")
def evolving_params():
    return np.array([[2500 + (i - N_GHA // 2) * 5, 0, 0, 5, 1] for i in range(N_GHA)])


def make_data(params, weights):
    model = np.array([LINLOG(parameters=p) for p in params])
    noise_level = model / 50
    data = model + np.random.normal(loc=0, scale=noise_level)
    return model, np.where(weights > 0, data, 1e6), noise_level


@pytest.fixture(scope="module")
def ideal_weights():
    return np.ones((N_GHA, len(FREQ)))


@pytest.fixture(scope="module")
def row_flags():
    weights = np.ones((N_GHA, len(FREQ)))
    weights[1::10] = 0
    return weights


@pytest.fixture(scope="module")
def bitsy_flags():
    np.random.seed(1234)
    return np.random.binomial(1, 0.6, size=(N_GHA, len(FREQ)))


@parametrize("params", [fxref(fid_params), fxref(evolving_params)])
@parametrize("weights", [fxref(ideal_weights), fxref(row_flags), fxref(bitsy_flags)])
@pytest.mark.parametrize("refit", [5, 3, False])
def test_model_bin_gha(params, weights, refit):

    model, data, sigma = make_data(params, weights)

    if refit:
        refit_mdl = LINLOG.with_nterms(refit)
        fits = [
            refit_mdl.fit(d, weights=ww) if np.any(ww > 0) else None
            for d, ww in zip(data, weights)
        ]
        fit_params = np.array(
            [
                fit.model_parameters
                if fit is not None
                else np.nan * np.ones(refit_mdl.n_terms)
                for fit in fits
            ]
        )
        resids = np.array(
            [
                fit.residual if fit is not None else np.nan * np.ones(len(FREQ))
                for fit in fits
            ]
        )
    else:
        refit_mdl = LINLOG
        fit_params = params
        resids = data - model

    p, r, w = averaging.bin_gha_unbiased_regular(
        params=fit_params,
        resids=resids,
        weights=weights,
        gha=np.linspace(0, 24, N_GHA),
        bins=[-12, 12.1],
    )

    # output should have one GHA bin
    assert len(p) == 1
    assert len(r) == 1
    assert len(w) == 1

    assert np.all(p[0] == np.nanmean(fit_params, axis=0))
    assert np.all(w[0] == np.sum(weights, axis=0))
    spec_out = refit_mdl(parameters=p[0]) + r[0]
    simple_spec_mean = np.mean(model, axis=0)
    print("Maximum deviation: ", np.abs(spec_out - simple_spec_mean).max())
    print(
        "Maximum dev (std): ",
        (np.abs(spec_out - simple_spec_mean) / (sigma / np.sqrt(N_GHA))).max(),
    )
    assert np.allclose(spec_out, simple_spec_mean, atol=6 * sigma / np.sqrt(N_GHA))


class TestBinArray:
    def test_out_shape_no_coords(self, fid_params, ideal_weights):
        model, corrupt, noise = make_data(fid_params, ideal_weights)

        coords, mean, wght = averaging.bin_array_unbiased_irregular(corrupt, axis=0)
        assert mean.shape == coords.shape == wght.shape == (1, 500)

        coords, mean, wght = averaging.bin_array_unbiased_irregular(corrupt, axis=1)
        assert mean.shape == coords.shape == wght.shape == (50, 1)

        coords, mean, wght = averaging.bin_array_unbiased_irregular(corrupt, axis=-1)
        assert mean.shape == coords.shape == wght.shape == (50, 1)

    def test_out_shape_with_coords(self, fid_params, ideal_weights):
        model, corrupt, noise = make_data(fid_params, ideal_weights)

        coords = FREQ
        with pytest.raises(ValueError):
            averaging.bin_array_unbiased_irregular(corrupt, coords=coords, axis=0)

        outc, mean, wght = averaging.bin_array_unbiased_irregular(
            corrupt, coords=coords, axis=-1
        )
        assert mean.shape == outc.shape == wght.shape == (50, 1)

        outc, mean, wght = averaging.bin_array_unbiased_irregular(
            corrupt, coords=coords, axis=-1, bins=5
        )
        assert mean.shape == outc.shape == wght.shape == (50, 100)

        outc, mean, wght = averaging.bin_array_unbiased_irregular(
            corrupt, coords=coords, axis=-1, bins=11
        )
        assert mean.shape == outc.shape == wght.shape == (50, 46)

        outc, mean, wght = averaging.bin_array_unbiased_irregular(
            corrupt, coords=coords, axis=-1, bins=1.0
        )
        assert mean.shape == outc.shape == wght.shape == (50, 50)

    def test_shape_3d_input(self, fid_params, ideal_weights):
        model, corrupt, noise = make_data(fid_params, ideal_weights)

        corrupt = corrupt.reshape((50, 20, 25))
        outc, mean, wght = averaging.bin_array_unbiased_irregular(
            corrupt, axis=0, bins=5
        )
        assert outc.shape == mean.shape == wght.shape == (10, 20, 25)

        outc, mean, wght = averaging.bin_array_unbiased_irregular(
            corrupt, axis=1, bins=5
        )
        assert outc.shape == mean.shape == wght.shape == (50, 4, 25)

        outc, mean, wght = averaging.bin_array_unbiased_irregular(
            corrupt, axis=2, bins=5
        )
        assert outc.shape == mean.shape == wght.shape == (50, 20, 5)

    def test_unity_input(self):
        data = np.ones((10, 100))

        outc, mean, wght = averaging.bin_array_unbiased_irregular(data, axis=0, bins=5)
        assert np.all(mean == 1)
        assert np.all(wght == 5)

    def test_unity_input_weights(self):
        data = np.ones((10, 100))
        weights = np.ones((10, 100))
        weights[:, ::5] = 0

        outc, mean, wght = averaging.bin_array_unbiased_irregular(
            data, axis=1, weights=weights, bins=5
        )
        assert np.all(mean == 1)
        assert np.all(wght == 4)
