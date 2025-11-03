"""Tests of the xrfi module."""

from pathlib import Path

import numpy as np
import pytest
import yaml
from pytest_cases import fixture_ref as fxref
from pytest_cases import parametrize

from edges import modeling as mdl
from edges.filters import xrfi
from edges.modeling import EdgesPoly, ScaleTransform

NFREQ = 1000


@pytest.fixture(scope="module")
def freq():
    """Default frequencies."""
    return np.linspace(50, 150, NFREQ)


@pytest.fixture(scope="module")
def sky_pl_1d(freq):
    return 1750 * (freq / 75.0) ** -2.5


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


def make_sky(sky_model, rfi_model=None, scale=1000, rfi_amp=200):
    if rfi_model is None:
        rfi_model = np.zeros_like(sky_model)
    std = sky_model / scale
    amp = std.max() * rfi_amp
    noise = thermal_noise(sky_model, scale=scale, seed=1010)
    rfi = rfi_model * amp
    return sky_model + noise + rfi, std, noise, rfi


def print_wrongness(
    wrong, std, info: xrfi.IterativeXRFIInfo, noise, true_flags, sky, rfi
):
    if len(wrong) > 0:
        print("Number of iterations: ", info.n_iters)

        print("Indices of WRONG flags:")
        print(wrong)
        print("RFI false positive(0)/negative(1): ")
        print(true_flags[wrong])
        print("Corrupted sky at wrong flags: ")
        print(sky[wrong])
        print("True Std. dev away from model at wrong flags: ")
        print((sky[wrong] - info.data_models[-1][wrong]) / std[wrong])
        print("Zscore wrong flags: ")
        print((sky[wrong] - info.data_models[-1][wrong]) / info.stds[-1][wrong])

        print("Std. dev of noise away from model at wrong flags: ")
        print(noise[wrong] / std[wrong])
        print("Std dev of RFI away from model at wrong flags: ")
        print(rfi[wrong] / std[wrong])
        print("Measured Std Dev: ")
        print(min(info.stds[-1]), max(info.stds[-1]))
        print("Actual Std Dev (for uniform):", np.std(noise))


class TestXRFIIterative:
    """Test the iterative xrfi method.

    Note that Gaussian and Mean filter kernels do not work well on non-flat
    spectra (they are biased high) and therefore we don't test them here.
    """

    def setup_class(self):
        self.polymod = xrfi.LinearModeler(
            mdl.EdgesPoly(n_terms=5), min_terms=5, max_terms=5
        )

    @parametrize("sky_model", [fxref(sky_pl_1d), fxref(sky_linpoly_1d)])
    @parametrize(
        "rfi_model", [fxref(rfi_null_1d), fxref(rfi_regular_1d), fxref(rfi_random_1d)]
    )
    @pytest.mark.parametrize("scale", [1000, 100])
    @pytest.mark.parametrize(
        "data_modeler",
        [
            xrfi.LinearModeler(
                model=mdl.EdgesPoly(n_terms=5), min_terms=5, max_terms=5
            ),
            xrfi.LinearModeler(
                model=mdl.EdgesPoly(n_terms=5), min_terms=3, max_terms=5
            ),
            xrfi.MedianFilterModeler(size=32),
        ],
        ids=["EdgesPolyConstant", "EdgesPolyVariable", "Median"],
    )
    @pytest.mark.parametrize(
        "std_modeler",
        [
            xrfi.LinearModeler(
                model=mdl.EdgesPoly(n_terms=5), min_terms=5, max_terms=5
            ),
            xrfi.LinearModeler(
                model=mdl.EdgesPoly(n_terms=5), min_terms=3, max_terms=5
            ),
            xrfi.MedianFilterModeler(size=32),
            xrfi.FilterModeler.gaussian(size=32),
            xrfi.FilterModeler.mean(size=32),
        ],
        ids=["EdgesPolyConstant", "EdgesPolyVariable", "Median", "Gaussian", "Mean"],
    )
    def test_easy_rfi_signatures(
        self, sky_model, rfi_model, scale, freq, data_modeler, std_modeler
    ):
        if (isinstance(data_modeler, xrfi.MedianFilterModeler)) and (
            isinstance(std_modeler, xrfi.MedianFilterModeler)
            or (
                isinstance(std_modeler, xrfi.LinearModeler)
                and std_modeler.min_terms == 3
            )  # variable poly
        ):
            pytest.skip(
                "Median for both data and std can produce zeros in the std model for "
                "steep power-law skies"
            )

        sky, std, noise, rfi = make_sky(sky_model, rfi_model, scale)

        true_flags = rfi_model > 0
        flags, info = xrfi.xrfi_iterative(
            data=sky,
            freqs=freq,
            data_modeler=data_modeler,
            std_modeler=std_modeler,
            threshold_setter=lambda i: 7.0,  # constant (high) threshold
        )

        wrong = np.where(true_flags != flags)[0]

        print_wrongness(wrong, std, info, noise, true_flags, sky, rfi)

        assert len(wrong) == 0

    @parametrize("sky_model", [fxref(sky_pl_1d), fxref(sky_linpoly_1d)])
    @parametrize("rfi_model", [fxref(rfi_regular_leaky)])
    @pytest.mark.parametrize("scale", [1000, 100])
    def test_watershed_strict(self, sky_model, rfi_model, scale, freq):
        sky, std, noise, rfi = make_sky(sky_model, rfi_model, scale, rfi_amp=200)

        true_flags = rfi_model > 0
        flags, info = xrfi.xrfi_iterative(
            sky,
            freqs=freq,
            data_modeler=self.polymod,
            std_modeler=self.polymod,
            watershed={1.0: 1},
            threshold_setter=lambda x: 5.0,
            max_iter=10,
        )

        wrong = np.where(true_flags != flags)[0]

        print_wrongness(wrong, std, info, noise, true_flags, sky, rfi)

        assert len(wrong) == 0

    @parametrize("sky_model", [fxref(sky_pl_1d), fxref(sky_linpoly_1d)])
    @parametrize("rfi_model", [fxref(rfi_regular_leaky)])
    @pytest.mark.parametrize("scale", [1000, 100])
    def test_watershed_relaxed(self, sky_model, rfi_model, scale, freq):
        sky, std, noise, rfi = make_sky(sky_model, rfi_model, scale, rfi_amp=500)

        true_flags = rfi_model > 0
        flags, info = xrfi.xrfi_iterative(
            sky,
            freqs=freq,
            data_modeler=self.polymod,
            std_modeler=self.polymod,
            watershed={1.0: 1},
            threshold_setter=lambda x: 6.0,
            max_iter=10,
        )

        # here we just assert no *missed* RFI
        wrong = np.where(true_flags & ~flags)[0]

        print_wrongness(wrong, std, info, noise, true_flags, sky, rfi)

        assert len(wrong) == 0

    def test_init_flags(self, sky_pl_1d, freq):
        # ensure init flags don't propagate through
        init_flags = np.where((freq > 90) & (freq < 100), True, False)
        flags, _info = xrfi.xrfi_iterative(
            sky_pl_1d,
            freqs=freq,
            data_modeler=self.polymod,
            std_modeler=self.polymod,
            init_flags=init_flags,
            threshold_setter=lambda x: 5.0,
            max_iter=5,
        )
        assert not np.any(flags)


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


class TestXRFIExplicit:
    def test_basic(self, freq):
        flags = xrfi.xrfi_explicit(freq=freq, extra_rfi=[(60, 70), (80, 90)])
        assert flags[105]
        assert not flags[0]
        assert flags[350]

    def test_passing_spec(self, freq, sky_pl_1d, rfi_regular_1d):
        flags = xrfi.xrfi_explicit(
            spectrum=sky_pl_1d, freq=freq, extra_rfi=[(60, 70), (80, 90)]
        )
        assert flags[105]
        assert not flags[0]
        assert flags[350]

    def test_passing_file(self, freq, sky_pl_1d, rfi_regular_1d, tmpdir: Path):
        with open(tmpdir / "rfi_file.yml", "w") as fl:
            yaml.dump({"rfi_ranges": [(60, 70), (80, 90)]}, fl)

        flags = xrfi.xrfi_explicit(
            spectrum=sky_pl_1d, freq=freq, rfi_file=tmpdir / "rfi_file.yml"
        )
        assert flags[105]
        assert not flags[0]
        assert flags[350]


class TestXRFIIterativeSlidingWindow:
    """Test the single-pass sliding RMS model.

    This is the algorithm most similar to Alan's C-code.
    """

    @parametrize("sky_model", [fxref(sky_pl_1d), fxref(sky_linpoly_1d)])
    @parametrize(
        "rfi_model", [fxref(rfi_null_1d), fxref(rfi_regular_1d), fxref(rfi_random_1d)]
    )
    @pytest.mark.parametrize("scale", [1000, 100])
    def test_on_simple_data(self, sky_model, rfi_model, scale, freq):
        sky, *_ = make_sky(sky_model, rfi_model, scale)

        true_flags = rfi_model > 0
        flags, _ = xrfi.xrfi_iterative_sliding_window(
            sky,
            freqs=freq,
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
