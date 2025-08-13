"""Tests of the xrfi module."""

import itertools
from pathlib import Path

import numpy as np
import pytest
import yaml
from pytest_cases import fixture_ref as fxref
from pytest_cases import parametrize

from edges.filters import xrfi
from edges.modelling import EdgesPoly, ScaleTransform

NFREQ = 1000


@pytest.fixture(scope="module")
def freq():
    """Default frequencies."""
    return np.linspace(50, 150, NFREQ)


@pytest.fixture(scope="module")
def sky_pl_1d(freq):
    return 1750 * (freq / 75.0) ** -2.5


@pytest.fixture(scope="module")
def sky_flat_1d():
    return np.ones(NFREQ)


@pytest.fixture(scope="module")
def sky_linpoly_1d(freq):
    p = EdgesPoly(
        parameters=[1750, 0, 3, -2, 7, 5],
        transform=ScaleTransform(scale=75.0),
    )
    f = np.linspace(50, 100, len(freq))
    return p(x=f)


def thermal_noise(spec, scale=1, seed=None):
    rng = np.random.default_rng(seed)
    return rng.normal(0, spec / scale)


@pytest.fixture(scope="module")
def rfi_regular_1d():
    a = np.zeros(NFREQ)
    a[50::50] = 1
    return a


@pytest.fixture(scope="module")
def rfi_regular_leaky():
    """RFI that leaks into neighbouring bins."""
    a = np.zeros(NFREQ)
    a[50:-30:50] = 1
    a[49:-30:50] = (
        1.0 / 1000
    )  # needs to be smaller than 200 or else it will be flagged outright.
    a[51:-30:50] = 1.0 / 1000
    return a


@pytest.fixture(scope="module")
def rfi_random_1d():
    a = np.zeros(NFREQ)
    rng = np.random.default_rng(12345)
    rfipos = rng.integers(0, len(a), 40, dtype=int)
    a[rfipos] = 1
    return a


@pytest.fixture(scope="module")
def rfi_null_1d():
    return np.zeros(NFREQ)


def print_wrongness(
    wrong, std, info: xrfi.ModelFilterInfo, noise, true_flags, sky, rfi
):
    if len(wrong) > 0:
        print("Number of iterations: ", info.n_iters)

        print("Indices of WRONG flags:")
        print(100 + wrong)
        print("RFI false positive(0)/negative(1): ")
        print(true_flags[wrong])
        print("Corrupted sky at wrong flags: ")
        print(sky[wrong])
        print("Std. dev away from model at wrong flags: ")
        print((sky[wrong] - info.models[-1](info.x)[wrong]) / std[wrong])
        print("Zscore wrong flags: ")
        print((sky[wrong] - info.models[-1](info.x)[wrong]) / info.stds[-1][wrong])

        print("Std. dev of noise away from model at wrong flags: ")
        print(noise[wrong] / std[wrong])
        print("Std dev of RFI away from model at wrong flags: ")
        print(rfi[wrong] / std[wrong])
        print("Measured Std Dev: ")
        print(min(info.stds[-1]), max(info.stds[-1]))
        print("Actual Std Dev (for uniform):", np.std(noise))


class TestFlaggedFilter:
    def test_flagged_filter(self, sky_pl_1d, rfi_regular_1d):
        flags = rfi_regular_1d.astype("bool")
        in_data = sky_pl_1d.copy()
        detrended = xrfi.flagged_filter(
            in_data, size=5, flags=flags, interp_flagged=False
        )
        assert not np.any(np.isnan(detrended))
        assert np.all(in_data == sky_pl_1d)

        # Anything close to a flag will not be identical, as the
        # median of an even number of items is the average of the middle two (and with
        #  a flag the total number of items is reduced by one).
        assert np.all(detrended[flags] == sky_pl_1d[flags])

        padded_flags = np.zeros_like(flags)
        for index in np.where(flags)[0]:
            padded_flags[index - 2 : index + 3] = True
            padded_flags[index] = False

        # Ensure everything away from flags is exactly the same.
        assert np.all(detrended[~padded_flags] == sky_pl_1d[~padded_flags])

        # An unflagged filter should be an identity operation.
        unflagged = xrfi.flagged_filter(in_data, size=5)
        assert np.all(unflagged == sky_pl_1d)

        # But not quite, when mode = 'reflect':
        unflagged = xrfi.flagged_filter(in_data, size=5, mode="reflect")
        assert not np.all(unflagged[:2] == sky_pl_1d[:2])

        # An unflagged filter with RFI should be very close to the original
        sky = sky_pl_1d + 100000 * rfi_regular_1d
        detrended = xrfi.flagged_filter(sky, size=5)
        assert np.allclose(detrended, sky_pl_1d, rtol=1e-1)


class TestMedfilt:
    @parametrize(
        "sky_model",
        [
            fxref(sky_pl_1d),
            fxref(sky_linpoly_1d),
        ],  # [fxref(sky_flat_1d), fxref(sky_pl_1d), fxref(sky_linpoly_1d)]
    )
    @parametrize(
        "rfi_model", [fxref(rfi_null_1d), fxref(rfi_regular_1d), fxref(rfi_random_1d)]
    )
    @pytest.mark.parametrize(
        "scale",
        list(
            itertools.product((1000, 100))
        ),  # Note that realistic noise should be ~250.
    )
    def test_1d_medfilt(self, sky_model, rfi_model, scale):
        sky, std, noise, rfi = make_sky(sky_model, rfi_model, scale)

        true_flags = rfi_model > 0
        flags, _significance = xrfi.xrfi_medfilt(
            sky, max_iter=1, threshold=10, kf=5, use_meanfilt=True
        )

        wrong = np.where(true_flags != flags)[0]

        print_wrongness(wrong, std, {}, noise, true_flags, sky, rfi)

        assert len(wrong) == 0


class TestXRFIModel:
    @parametrize(
        "sky_model", [fxref(sky_flat_1d), fxref(sky_pl_1d), fxref(sky_linpoly_1d)]
    )
    @parametrize(
        "rfi_model", [fxref(rfi_null_1d), fxref(rfi_regular_1d), fxref(rfi_random_1d)]
    )
    @pytest.mark.parametrize("scale", [1000, 100])
    def test_xrfi_model(self, sky_model, rfi_model, scale, freq):
        sky, std, noise, rfi = make_sky(sky_model, rfi_model, scale)

        true_flags = rfi_model > 0
        flags, info = xrfi.xrfi_model(sky, freq=freq)

        wrong = np.where(true_flags != flags)[0]

        print_wrongness(wrong, std, info, noise, true_flags, sky, rfi)

        assert len(wrong) == 0

    @parametrize(
        "sky_model", [fxref(sky_flat_1d), fxref(sky_pl_1d), fxref(sky_linpoly_1d)]
    )
    @parametrize("rfi_model", [fxref(rfi_regular_leaky)])
    @pytest.mark.parametrize("scale", [1000, 100])
    def test_watershed_strict(self, sky_model, rfi_model, scale, freq):
        sky, std, noise, rfi = make_sky(sky_model, rfi_model, scale, rfi_amp=200)

        true_flags = rfi_model > 0
        flags, info = xrfi.xrfi_model(
            sky, freq=freq, watershed=1, threshold=5, min_threshold=4, max_iter=10
        )

        wrong = np.where(true_flags != flags)[0]

        print_wrongness(wrong, std, info, noise, true_flags, sky, rfi)

        assert len(wrong) == 0

    @parametrize(
        "sky_model", [fxref(sky_flat_1d), fxref(sky_pl_1d), fxref(sky_linpoly_1d)]
    )
    @parametrize("rfi_model", [fxref(rfi_regular_leaky)])
    @pytest.mark.parametrize("scale", [1000, 100])
    def test_watershed_relaxed(self, sky_model, rfi_model, scale, freq):
        sky, std, noise, rfi = make_sky(sky_model, rfi_model, scale, rfi_amp=500)

        true_flags = rfi_model > 0
        flags, info = xrfi.xrfi_model(sky, freq=freq, watershed=1, threshold=6)

        # here we just assert no *missed* RFI
        wrong = np.where(true_flags & ~flags)[0]

        print_wrongness(wrong, std, info, noise, true_flags, sky, rfi)

        assert len(wrong) == 0

    def test_init_flags(self, sky_pl_1d, rfi_null_1d, freq):
        # ensure init flags don't propagate through
        flags, _info = xrfi.xrfi_model(sky_pl_1d, freq=freq, init_flags=(90, 100))
        assert not np.any(flags)

    @parametrize("rfi_model", [fxref(rfi_random_1d), fxref(rfi_regular_1d)])
    @pytest.mark.parametrize("std_estimator", ["medfilt", "std", "mad", "sliding_rms"])
    def test_std_estimator(self, sky_flat_1d, rfi_model, std_estimator, freq):
        if std_estimator == "sliding_rms" and rfi_model[50] == 0:
            pytest.skip("sliding_rms doesn't work well for unrealistic random RFI")

        print("RFI MODEL: ", rfi_model)
        sky, std, noise, rfi = make_sky(sky_flat_1d, rfi_model, scale=1000, rfi_amp=300)

        true_flags = rfi_model > 0
        true_flags2 = rfi > 0
        assert np.all(true_flags == true_flags2)
        flags, info = xrfi.xrfi_model(
            sky, freq=freq, std_estimator=std_estimator, threshold=4
        )

        print("Number of flags: ", np.sum(flags))
        print("# False Positives: ", np.sum(flags & ~true_flags))
        print("# False Negatives: ", np.sum(~flags & true_flags))
        wrong = np.where(true_flags != flags)[0]

        print_wrongness(wrong, std, info, noise, true_flags, sky, rfi)

        assert len(wrong) == 0

    def test_bad_std_estimator(self, sky_flat_1d, rfi_random_1d, freq):
        sky, _std, _noise, _rfi = make_sky(sky_flat_1d, rfi_random_1d, scale=1000)

        with pytest.raises(ValueError):
            _flags, _info = xrfi.xrfi_model(
                sky, freq=freq, std_estimator="bad_estimator"
            )


class TestWatershed:
    def test_watershed(self):
        rfi = np.zeros((10, 10), dtype=bool)
        out, _ = xrfi.xrfi_watershed(flags=rfi)
        assert not np.any(out)

        rfi = np.ones((10, 10), dtype=bool)
        out, _ = xrfi.xrfi_watershed(flags=rfi)
        assert np.all(out)

        rfi = np.repeat([0, 1], 48).reshape((3, 32))
        out, _ = xrfi.xrfi_watershed(flags=rfi, tol=0.2)
        assert np.all(out)

    def test_pass_weights(self):
        out, _ = xrfi.xrfi_watershed(weights=np.zeros((10, 10)))
        assert np.all(out)

    def test_pass_no_flags(self):
        with pytest.raises(ValueError):
            xrfi.xrfi_watershed()


class TestModelSweep:
    @parametrize(
        "sky_model", [fxref(sky_flat_1d), fxref(sky_pl_1d), fxref(sky_linpoly_1d)]
    )
    @parametrize(
        "rfi_model", [fxref(rfi_null_1d), fxref(rfi_regular_1d), fxref(rfi_random_1d)]
    )
    @pytest.mark.parametrize("scale", [1000, 100])
    def test_xrfi_model_sweep(self, sky_model, rfi_model, scale):
        sky, std, noise, rfi = make_sky(sky_model, rfi_model, scale)

        true_flags = rfi_model > 0
        flags, info = xrfi.xrfi_model_sweep(
            sky,
            max_iter=10,
            threshold=5,
            use_median=True,
            which_bin="last",
        )

        # Only consider flags after bin 100 (since that's the bin width)
        wrong = np.where(true_flags[100:] != flags[100:])[0]

        print_wrongness(wrong, std, info, noise, true_flags, sky, rfi)
        assert len(wrong) == 0

    @parametrize(
        "sky_model", [fxref(sky_flat_1d), fxref(sky_pl_1d), fxref(sky_linpoly_1d)]
    )
    @parametrize(
        "rfi_model", [fxref(rfi_null_1d), fxref(rfi_regular_1d), fxref(rfi_random_1d)]
    )
    @pytest.mark.parametrize("scale", [1000, 100])
    def test_xrfi_model_sweep_all(self, sky_model, rfi_model, scale):
        sky, std, noise, rfi = make_sky(sky_model, rfi_model, scale)

        true_flags = rfi_model > 0
        flags, info = xrfi.xrfi_model_sweep(
            sky, max_iter=10, which_bin="all", threshold=5, use_median=True
        )

        # Only consider flags after bin 100 (since that's the bin width)
        wrong = np.where(true_flags[100:] != flags[100:])[0]

        print_wrongness(wrong, std, info, noise, true_flags, sky, rfi)
        assert len(wrong) == 0

    @parametrize(
        "sky_model", [fxref(sky_flat_1d), fxref(sky_pl_1d), fxref(sky_linpoly_1d)]
    )
    @parametrize(
        "rfi_model", [fxref(rfi_null_1d), fxref(rfi_regular_1d), fxref(rfi_random_1d)]
    )
    @pytest.mark.parametrize("scale", [1000, 100])
    def test_xrfi_model_sweep_watershed(self, sky_model, rfi_model, scale):
        sky, std, noise, rfi = make_sky(sky_model, rfi_model, scale)

        true_flags = rfi_model > 0
        flags, info = xrfi.xrfi_model_sweep(
            sky, max_iter=10, which_bin="all", threshold=5, use_median=True, watershed=3
        )

        # Only consider flags after bin 100 (since that's the bin width)
        wrong = np.where(true_flags[100:] & ~flags[100:])[0]

        print_wrongness(wrong, std, info, noise, true_flags, sky, rfi)
        assert len(wrong) == 0

    def test_too_many_flags(self):
        spec = np.ones(500)
        flags = np.ones(500, dtype=bool)
        weights = np.zeros(500)

        # We're testing where not *all* are flagged, just enough to be more than the
        # number of terms...
        flags[::100] = False
        weights[::100] = 1

        flags_, info = xrfi.xrfi_model_sweep(spectrum=np.where(flags, np.nan, spec))
        assert np.all(flags_)
        assert not info

        flags_, info = xrfi.xrfi_model_sweep(spectrum=spec, flags=flags)
        assert np.all(flags_)
        assert not info

        flags_, info = xrfi.xrfi_model_sweep(spectrum=spec, weights=weights)
        assert np.all(flags_)
        assert not info

    def test_all_flagged(self):
        spec = np.ones(500)
        flags = np.ones(500, dtype=bool)
        weights = np.zeros(500)

        flags_, info = xrfi.xrfi_model_sweep(spectrum=spec, flags=flags)
        assert np.all(flags_)
        assert not info

        flags_, info = xrfi.xrfi_model_sweep(spectrum=spec, weights=weights)
        assert np.all(flags_)
        assert not info

        flags_, info = xrfi.xrfi_model_sweep(spectrum=spec * np.nan)
        assert np.all(flags_)
        assert not info

    def test_no_data_error(self):
        # to raise no data error, there must be no data for a whole window
        spec = np.ones(500)
        spec[50:150] = np.nan

        flags, _info = xrfi.xrfi_model_sweep(spec)
        assert flags.shape == (500,)

    @parametrize(
        "rfi_model", [fxref(rfi_null_1d), fxref(rfi_regular_1d), fxref(rfi_random_1d)]
    )
    @pytest.mark.parametrize("scale", [1000, 100])
    def test_xrfi_model_sweep_median(self, sky_flat_1d, rfi_model, scale):
        rfi = rfi_model.copy()
        rfi[:100] = 0
        sky, std, noise, rfi = make_sky(sky_flat_1d, rfi_model, scale)

        true_flags = rfi_model > 0
        flags, info = xrfi.xrfi_model_sweep(
            sky, max_iter=10, threshold=5, use_median=False, which_bin="all"
        )

        # Only consider flags after bin 100 (since that's the bin width)
        wrong = np.where(true_flags[100:] != flags[100:])[0]

        print_wrongness(wrong, std, info, noise, true_flags, sky, rfi)

        assert len(wrong) == 0

    def test_watershed_last(self, sky_flat_1d):
        with pytest.raises(ValueError):
            xrfi.xrfi_model_sweep(sky_flat_1d, which_bin="last", watershed=4)

    def test_giving_weights(self, sky_flat_1d):
        sky, _std, _noise, _rfi = make_sky(sky_flat_1d)

        flags, _info = xrfi.xrfi_model_sweep(
            sky,
            weights=np.ones_like(sky),
            max_iter=10,
            which_bin="all",
            threshold=5,
            use_median=True,
        )

        flags2, _info2 = xrfi.xrfi_model_sweep(
            sky, max_iter=10, which_bin="all", threshold=5, use_median=True
        )

        assert np.all(flags == flags2)


def make_sky(sky_model, rfi_model=None, scale=1000, rfi_amp=200):
    if rfi_model is None:
        rfi_model = np.zeros_like(sky_model)
    std = sky_model / scale
    amp = std.max() * rfi_amp
    noise = thermal_noise(sky_model, scale=scale, seed=1010)
    rfi = rfi_model * amp
    return sky_model + noise + rfi, std, noise, rfi


class TestXRFIExplicit:
    def test_basic(self, freq, sky_flat_1d, rfi_regular_1d):
        flags = xrfi.xrfi_explicit(freq=freq, extra_rfi=[(60, 70), (80, 90)])
        assert flags[105]
        assert not flags[0]
        assert flags[350]

    def test_passing_spec(self, freq, sky_flat_1d, rfi_regular_1d):
        flags = xrfi.xrfi_explicit(
            spectrum=sky_flat_1d, freq=freq, extra_rfi=[(60, 70), (80, 90)]
        )
        assert flags[105]
        assert not flags[0]
        assert flags[350]

    def test_passing_file(self, freq, sky_flat_1d, rfi_regular_1d, tmpdir: Path):
        with open(tmpdir / "rfi_file.yml", "w") as fl:
            yaml.dump({"rfi_ranges": [(60, 70), (80, 90)]}, fl)

        flags = xrfi.xrfi_explicit(
            spectrum=sky_flat_1d, freq=freq, rfi_file=tmpdir / "rfi_file.yml"
        )
        assert flags[105]
        assert not flags[0]
        assert flags[350]


class TestXRFIModelNonlinearWindow:
    """Test the single-pass sliding RMS model.

    This is the algorithm most similar to Alan's C-code.
    """

    @parametrize(
        "sky_model", [fxref(sky_flat_1d), fxref(sky_pl_1d), fxref(sky_linpoly_1d)]
    )
    @parametrize(
        "rfi_model", [fxref(rfi_null_1d), fxref(rfi_regular_1d), fxref(rfi_random_1d)]
    )
    @pytest.mark.parametrize("scale", [1000, 100])
    def test_on_simple_data(self, sky_model, rfi_model, scale, freq):
        sky, *_ = make_sky(sky_model, rfi_model, scale)

        true_flags = rfi_model > 0
        flags, info = xrfi.xrfi_model_nonlinear_window(
            sky,
            freq=freq,
            model=EdgesPoly(n_terms=6),
            threshold=3.5,
            max_iter=15,
            reflag_thresh=1,
        )

        print("False Negatives: ", np.sum(true_flags & (~flags)))
        print("False Positives: ", np.sum(flags & (~true_flags)))

        wrong = np.where(true_flags != flags)[0]

        print(sky)

        assert len(wrong) == 0


@pytest.fixture(scope="module")
def model_info(sky_pl_1d, rfi_random_1d, freq):
    sky, _std, _noise, _rfi = make_sky(sky_pl_1d, rfi_random_1d)
    _flags, info = xrfi.xrfi_model(sky, freq=freq, max_iter=3)
    return info


@pytest.mark.skip("takes too long")
def test_visualisation(model_info: xrfi.ModelFilterInfo):
    xrfi.visualise_model_info(model_info)


def test_model_info_io(model_info: xrfi.ModelFilterInfo, tmpdir: Path):
    model_info.write(tmpdir / "model_info.h5")
    info2 = xrfi.ModelFilterInfo.from_file(tmpdir / "model_info.h5")
    assert all(
        model_info.n_flags_changed[i] == info2.n_flags_changed[i]
        for i in range(model_info.n_iters)
    )


def test_model_info_container(model_info: xrfi.ModelFilterInfo, tmpdir: Path):
    container = xrfi.ModelFilterInfoContainer([model_info])
    assert np.allclose(container.x, model_info.x)
    assert np.allclose(container.data, model_info.data)
    assert np.all(container.flags == model_info.flags)
    assert container.n_iters == model_info.n_iters
    assert np.all(container.total_flags == model_info.total_flags)
    assert np.allclose(container.get_model(), model_info.get_model())
    assert np.allclose(container.get_residual(), model_info.get_residual())
    assert np.allclose(container.get_absres_model(), model_info.get_absres_model())
    assert np.allclose(container.thresholds, model_info.thresholds)

    container.write(tmpdir / "model_info_container.h5")
    container2 = xrfi.ModelFilterInfoContainer.from_file(
        tmpdir / "model_info_container.h5"
    )
    assert np.all(container.total_flags == container2.total_flags)


class TestVisualiseModelInfo:
    def test_visualise_model_info(self, sky_linpoly_1d, rfi_random_1d, freq):
        sky, *_ = make_sky(sky_linpoly_1d, rfi_random_1d)

        _, info = xrfi.xrfi_model(sky, freq=freq, watershed=1, threshold=6)

        xrfi.visualise_model_info(info)
