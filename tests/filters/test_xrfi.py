"""Tests of the xrfi module."""

from pathlib import Path

import numpy as np
import pytest
import yaml
from pytest_cases import fixture_ref as fxref
from pytest_cases import parametrize
from scipy.stats import norm

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


# ---------------------------------------------------------------------------
# Statistical tests of the xRFI noise model and flagging.
# ---------------------------------------------------------------------------
NSTAT = 10000


@pytest.fixture(scope="module")
def stat_freqs():
    return np.linspace(50, 150, NSTAT)


def _poly_modeler(n_terms: int = 3) -> xrfi.LinearModeler:
    return xrfi.LinearModeler(model=mdl.Polynomial(n_terms=n_terms))


class TestStdModelers:
    """Each std modeler should return ~1 when fed unit-variance Gaussian residuals."""

    @pytest.mark.parametrize(
        "modeler",
        [_poly_modeler(3), xrfi.FilterModeler.gaussian(size=64)],
        ids=["linear", "gaussian-filter"],
    )
    def test_base_get_std_unit_noise(self, modeler, stat_freqs):
        rng = np.random.default_rng(1234)
        resids = rng.normal(size=NSTAT)
        model = modeler.init_model(modeler.set_params(0), stat_freqs)
        std = modeler.get_std(model, resids, np.ones(NSTAT))
        assert np.mean(std) == pytest.approx(1.0, abs=0.05)

    def test_median_get_std_unit_noise(self):
        rng = np.random.default_rng(1234)
        resids = rng.normal(size=NSTAT)
        std = xrfi.MedianFilterModeler(size=64).get_std(None, resids, np.ones(NSTAT))
        assert np.mean(std) == pytest.approx(1.0, abs=0.05)


def _false_positive_check(std_modeler, freqs, threshold: float = 2.5):
    """Run xrfi_iterative on pure noise and check the false-positive fraction.

    xRFI flags only *positive* outliers, so the expected false-positive rate is the
    one-sided Gaussian tail probability, ``norm.sf(threshold)``.
    """
    rng = np.random.default_rng(2024)
    data = 10 + rng.normal(size=freqs.size)
    flags, info = xrfi.xrfi_iterative(
        data,
        freqs=freqs,
        data_modeler=_poly_modeler(3),
        std_modeler=std_modeler,
        threshold_setter=lambda i: threshold,
        max_iter=20,
    )
    assert info.n_iters < 20
    p = norm.sf(threshold)
    tol = 4 * np.sqrt(p * (1 - p) / freqs.size)
    assert abs(np.mean(flags) - p) < tol


class TestFalsePositiveRate:
    def test_linear_std_modeler(self, stat_freqs):
        _false_positive_check(_poly_modeler(3), stat_freqs)

    def test_median_std_modeler(self, stat_freqs):
        # A wide window is used: with narrow windows (e.g. 64 channels) the rolling
        # median std estimate is noisy enough to raise the false-positive rate.
        _false_positive_check(xrfi.MedianFilterModeler(size=1001), stat_freqs)


class TestSingleSpike:
    """A single huge spike in pure noise should be the only flagged channel."""

    def _run(self, spike: float):
        n = 1000
        freqs = np.linspace(50, 150, n)
        rng = np.random.default_rng(7)
        data = 10 + rng.normal(size=n)
        data[400] += spike
        return xrfi.xrfi_iterative(
            data,
            freqs=freqs,
            data_modeler=_poly_modeler(3),
            std_modeler=_poly_modeler(3),
            threshold_setter=lambda i: 5.0,
        )[0]

    def test_positive_spike_flagged(self):
        flags = self._run(100.0)
        assert flags[400]
        assert np.sum(flags) == 1

    def test_negative_spike_not_flagged(self):
        # xRFI is deliberately one-sided: only positive outliers are flagged.
        flags = self._run(-100.0)
        assert not np.any(flags)


def test_term_increase_reaches_max_terms():
    n = 500
    freqs = np.linspace(50, 150, n)
    rng = np.random.default_rng(0)
    data = 1000 * (freqs / 75) ** -2.5 + rng.normal(scale=0.01, size=n)
    _, info = xrfi.xrfi_iterative(
        data,
        freqs=freqs,
        data_modeler=xrfi.LinearModeler(
            model=mdl.LinLog(n_terms=5), min_terms=3, max_terms=5, term_increase=1
        ),
        std_modeler=_poly_modeler(2),
        threshold_setter=lambda i: 5.0,
    )
    assert info.model_params[-1]["nterms"] == 5
    assert [p["nterms"] for p in info.model_params[:3]] == [3, 4, 5]


def test_constant_params_converge_after_first_unchanged_iteration():
    """With fixed model params, an iteration with no flag changes is final."""
    n = 500
    freqs = np.linspace(50, 150, n)
    rng = np.random.default_rng(0)
    data = 10 + rng.normal(scale=0.01, size=n)
    flags, info = xrfi.xrfi_iterative(
        data,
        freqs=freqs,
        data_modeler=_poly_modeler(2),
        std_modeler=_poly_modeler(2),
        threshold_setter=lambda i: 10.0,
    )
    assert not np.any(flags)
    assert info.n_iters == 1


# ---------------------------------------------------------------------------
# Input mutation, signatures and info containers.
# ---------------------------------------------------------------------------
def _noisy_spectrum(n: int = 200, seed: int = 0):
    freqs = np.linspace(50, 150, n)
    rng = np.random.default_rng(seed)
    data = 100 + 3 * (freqs / 75) ** 2 + rng.normal(size=n)
    data[50] += 1000
    return freqs, data


class TestNoInputMutation:
    """xRFI functions must not modify the arrays they are given."""

    def _check(self, fnc, **kwargs):
        copies = {k: v.copy() for k, v in kwargs.items() if isinstance(v, np.ndarray)}
        fnc(**kwargs)
        for k, v in copies.items():
            np.testing.assert_array_equal(kwargs[k], v, err_msg=f"{k} was mutated")

    def test_iterative(self):
        freqs, data = _noisy_spectrum()
        flags = np.zeros(data.size, dtype=bool)
        flags[10] = True
        weights = np.ones(data.size)
        weights[20] = 0
        self._check(
            xrfi.xrfi_iterative,
            data=data,
            freqs=freqs,
            flags=flags,
            weights=weights,
            data_modeler=_poly_modeler(5),
            std_modeler=_poly_modeler(2),
        )

    def test_sliding_window(self):
        freqs, data = _noisy_spectrum()
        flags = np.zeros(data.size, dtype=bool)
        flags[10] = True
        weights = np.ones(data.size)
        weights[20] = 0
        self._check(
            xrfi.xrfi_iterative_sliding_window,
            spectrum=data,
            freqs=freqs,
            model=mdl.Polynomial(n_terms=5),
            flags=flags,
            weights=weights,
        )

    def test_explicit(self):
        freqs, data = _noisy_spectrum()
        flags = np.zeros(data.size, dtype=bool)
        self._check(
            xrfi.xrfi_explicit,
            spectrum=data,
            freq=freqs,
            flags=flags,
            extra_rfi=[(60, 70)],
        )

    def test_watershed(self):
        flags = np.zeros((4, 10), dtype=bool)
        flags[0, :8] = True
        weights = np.ones((4, 10))
        weights[1, 3] = 0
        self._check(xrfi.xrfi_watershed, flags=flags, weights=weights)


class TestWatershed1D:
    def test_mostly_flagged(self):
        flags = np.zeros(10, dtype=bool)
        flags[:6] = True
        out, _ = xrfi.xrfi_watershed(flags=flags, tol=0.5)
        assert out.shape == (10,)
        assert np.all(out)

    def test_few_flags(self):
        flags = np.zeros(10, dtype=bool)
        flags[:2] = True
        out, _ = xrfi.xrfi_watershed(flags=flags, tol=0.5)
        np.testing.assert_array_equal(out, flags)

    def test_weights_count_as_flags(self):
        weights = np.ones(10)
        weights[:6] = 0
        out, _ = xrfi.xrfi_watershed(flags=np.zeros(10, dtype=bool), weights=weights)
        assert np.all(out)


class TestXRFIExplicitSignature:
    def test_freqs_alias(self, freq):
        a = xrfi.xrfi_explicit(freq=freq, extra_rfi=[(60, 70)])
        b = xrfi.xrfi_explicit(freqs=freq, extra_rfi=[(60, 70)])
        np.testing.assert_array_equal(a, b)

    def test_both_freq_and_freqs(self, freq):
        with pytest.raises(ValueError, match="only one of"):
            xrfi.xrfi_explicit(freq=freq, freqs=freq, extra_rfi=[(60, 70)])

    def test_no_freqs(self):
        with pytest.raises(ValueError, match="must provide"):
            xrfi.xrfi_explicit(extra_rfi=[(60, 70)])

    def test_accepts_weights(self, freq, sky_pl_1d):
        flags = xrfi.xrfi_explicit(
            sky_pl_1d, freqs=freq, weights=np.ones_like(sky_pl_1d), extra_rfi=[(60, 70)]
        )
        assert flags.shape == sky_pl_1d.shape

    def test_range_ends_are_exclusive(self):
        freqs = np.arange(50.0, 80.0, 1.0)
        flags = xrfi.xrfi_explicit(freqs=freqs, extra_rfi=[(60, 70)])
        np.testing.assert_array_equal(freqs[flags], np.arange(61.0, 70.0, 1.0))

    def test_2d(self, freq):
        spec = np.ones((3, freq.size))
        flags = xrfi.xrfi_explicit(spec, freqs=freq, extra_rfi=[(60, 70)])
        assert flags.shape == spec.shape
        assert np.all(flags[:, (freq > 60) & (freq < 70)])


class TestInfoContainers:
    def _two_infos(self):
        out = []
        for seed in (1, 2):
            freqs, data = _noisy_spectrum(n=150, seed=seed)
            out.append(
                xrfi.xrfi_iterative(
                    data,
                    freqs=freqs,
                    data_modeler=_poly_modeler(5),
                    std_modeler=_poly_modeler(2),
                    threshold_setter=lambda i: 4.0,
                )[1]
            )
        return out

    def test_iterative_get_std_model(self):
        info = self._two_infos()[0]
        np.testing.assert_array_equal(info.get_std_model(), info.stds[-1])
        np.testing.assert_array_equal(info.get_std_model(0), info.stds[0])

    def test_container(self):
        infos = self._two_infos()
        cont = xrfi.ModelFilterInfoContainer()
        for info in infos:
            cont = cont.append(info)

        ntot = sum(info.data.size for info in infos)
        n_iters = max(info.n_iters for info in infos)
        assert cont.n_iters == n_iters
        assert cont.x.shape == (ntot,)

        assert len(cont.flags) == n_iters
        assert all(f.shape == (ntot,) for f in cont.flags)
        np.testing.assert_array_equal(
            cont.flags[-1], np.concatenate([info.flags[-1] for info in infos])
        )

        assert len(cont.stds) == n_iters
        assert all(s.shape == (ntot,) for s in cont.stds)

        std_model = cont.get_std_model()
        np.testing.assert_array_equal(
            std_model, np.concatenate([info.stds[-1] for info in infos])
        )
        np.testing.assert_array_equal(cont.get_absres_model(), std_model)

        assert cont.get_model().shape == (ntot,)
        assert cont.get_residual().shape == (ntot,)
        assert len(cont.thresholds) == n_iters

    def test_sliding_window_info_lengths(self):
        freqs, data = _noisy_spectrum()
        _, info = xrfi.xrfi_iterative_sliding_window(
            data, freqs=freqs, model=mdl.Polynomial(n_terms=5), threshold=3.0
        )
        assert info.n_iters == len(info.data_models)
        assert info.n_iters >= 1
        assert len(info.thresholds) == info.n_iters
        assert len(info.stds) == info.n_iters
        assert len(info.flags) == info.n_iters
