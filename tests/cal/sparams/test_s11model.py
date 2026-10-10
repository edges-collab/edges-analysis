from pathlib import Path

import numpy as np
import pytest
from astropy import units as un

from edges import modeling as mdl
from edges.cal.sparams import (
    AGILENT_85033E,
    CalkitReadings,
    ReflectionCoefficient,
    S11ModelParams,
    SParams,
    get_delay,
    smooth_sparams,
)
from edges.cal.sparams.core.s11model import _delay_objective

_DATA = Path(__file__).parents[2] / "data"
_REAL_S1P_FILES = sorted(
    list(_DATA.glob("alanmode/edges3-*-raw/*.s1p"))
    + list(_DATA.glob("cal/Receiver01_25C_2019_11_26_040_to_200MHz/S11/*/*.s1p"))
)
_DELAY_GRID = np.arange(-1e-3, 0.1, 1e-4)


def _delay_objective_reference(gamma: ReflectionCoefficient) -> np.ndarray:
    """The delay-search objective, evaluated one trial delay at a time."""
    return np.array([
        -np.abs(np.sum(gamma.remove_delay(d * un.microsecond).reflection_coefficient))
        for d in _DELAY_GRID
    ])


def _get_delay_reference(gamma: ReflectionCoefficient) -> un.Quantity:
    return _DELAY_GRID[np.argmin(_delay_objective_reference(gamma))] * un.microsecond


class TestGetDelay:
    """The vectorised delay search reproduces the delay-by-delay grid search."""

    def test_real_data_files_found(self):
        assert len(_REAL_S1P_FILES) >= 40

    @pytest.mark.parametrize(
        "path", _REAL_S1P_FILES, ids=lambda p: f"{p.parent.name}/{p.name}"
    )
    def test_real_data(self, path: Path):
        gamma = ReflectionCoefficient.from_s1p(path)
        assert get_delay(gamma) == _get_delay_reference(gamma)

    @pytest.mark.parametrize("seed", range(5))
    def test_random(self, seed: int):
        rng = np.random.default_rng(seed)
        nf = rng.integers(10, 3000)
        freqs = np.sort(rng.uniform(10, 300, nf)) * un.MHz
        delay = rng.uniform(-0.002, 0.11)
        gamma = ReflectionCoefficient(
            freqs=freqs,
            reflection_coefficient=np.exp(-2j * np.pi * freqs.to_value("MHz") * delay)
            + 0.3 * (rng.normal(size=nf) + 1j * rng.normal(size=nf)),
        )
        ref = _delay_objective_reference(gamma)
        # Small blocks exercise the chunking over trial delays.
        for block in (2**20, 1000, 1):
            obj = _delay_objective(
                gamma, _DELAY_GRID * un.microsecond, max_block_size=block
            )
            np.testing.assert_allclose(obj, ref, rtol=1e-14, atol=0)
        assert get_delay(gamma) == _get_delay_reference(gamma)


class TestS11ModelParams:
    def test_clone(self):
        params = S11ModelParams(set_transform_range=True)
        new_params = params.clone(set_transform_range=False)
        assert not new_params.set_transform_range
        assert params != new_params


class TestSmoothSparams:
    def setup_class(self):
        fq = np.linspace(50, 100, 50)
        self.sparams = SParams(
            freqs=fq * un.MHz,
            s11=fq + fq**2,
            s12=fq * 0.1,
            s21=1 + 0.05 * fq - 0.05 * fq * 1j,
            s22=0.5 * fq + 0.01 * fq**2 * 1j,
        )

        self.params = S11ModelParams(
            model=mdl.Polynomial(n_terms=5),
            complex_model_type=mdl.ComplexRealImagModel,
            set_transform_range=False,
            find_model_delay=False,
            combine_s12s21=False,
        )

    def test_trivial(self):
        smoothed = smooth_sparams(self.sparams, params=self.params)

        np.testing.assert_allclose(self.sparams.s11, smoothed.s11, atol=1e-6)
        np.testing.assert_allclose(self.sparams.s12, smoothed.s12, atol=1e-6)
        np.testing.assert_allclose(self.sparams.s21, smoothed.s21, atol=1e-6)
        np.testing.assert_allclose(self.sparams.s22, smoothed.s22, atol=1e-6)

    def test_passing_freqs_trivial(self):
        smoothed = smooth_sparams(
            self.sparams, params=self.params, freqs=self.sparams.freqs
        )

        np.testing.assert_allclose(self.sparams.s11, smoothed.s11, atol=1e-6)
        np.testing.assert_allclose(self.sparams.s12, smoothed.s12, atol=1e-6)
        np.testing.assert_allclose(self.sparams.s21, smoothed.s21, atol=1e-6)
        np.testing.assert_allclose(self.sparams.s22, smoothed.s22, atol=1e-6)

    def test_passing_individual_params(self):
        smoothed = smooth_sparams(
            self.sparams,
            params={
                "s11": self.params,
                "s12": self.params,
                "s21": self.params,
                "s22": self.params,
            },
        )

        np.testing.assert_allclose(self.sparams.s11, smoothed.s11, atol=1e-6)
        np.testing.assert_allclose(self.sparams.s12, smoothed.s12, atol=1e-6)
        np.testing.assert_allclose(self.sparams.s21, smoothed.s21, atol=1e-6)
        np.testing.assert_allclose(self.sparams.s22, smoothed.s22, atol=1e-6)


class TestSmoothSparamsPhaseWrap:
    """Smoothing S-parameters derived from OSL whose S12*S21 phase wraps past pi."""

    def setup_class(self):
        self.freqs = np.linspace(40, 200, 161) * un.MHz
        n = len(self.freqs)
        # A 5 ns error network: the phase of S12*S21 crosses -pi at 100 MHz.
        self.s12s21 = 0.9 * np.exp(-2j * np.pi * (self.freqs * 5 * un.ns).to_value(""))
        err = SParams(
            freqs=self.freqs,
            s11=np.full(n, 0.05 + 0.02j),
            s12=np.sqrt(self.s12s21),
            s21=self.s12s21 / np.sqrt(self.s12s21),
            s22=np.full(n, 0.1 - 0.05j),
        )
        kit = AGILENT_85033E.at_freqs(self.freqs)
        meas = CalkitReadings(
            open=kit.open.embed(err),
            short=kit.short.embed(err),
            match=kit.match.embed(err),
        )
        self.sparams = SParams.from_calkit_measurements(meas, AGILENT_85033E)
        np.testing.assert_allclose(
            self.sparams.s12 * self.sparams.s21, self.s12s21, atol=1e-12
        )

    def _params(self, combine: bool) -> S11ModelParams:
        return S11ModelParams(
            model=mdl.Polynomial(
                n_terms=14, transform=mdl.ZerotooneTransform(range=(0, 1))
            ),
            complex_model_type=mdl.ComplexRealImagModel,
            set_transform_range=True,
            combine_s12s21=combine,
        )

    def test_combined(self):
        smoothed = smooth_sparams(self.sparams, params=self._params(True))
        np.testing.assert_allclose(smoothed.s12 * smoothed.s21, self.s12s21, atol=1e-6)

    def test_separate(self):
        smoothed = smooth_sparams(self.sparams, params=self._params(False))
        np.testing.assert_allclose(smoothed.s12 * smoothed.s21, self.s12s21, atol=1e-6)

    def test_combined_s12_is_continuous(self):
        """The square root of the smoothed product follows a continuous branch."""
        smoothed = smooth_sparams(self.sparams, params=self._params(True))
        np.testing.assert_array_equal(smoothed.s12, smoothed.s21)
        assert np.all(np.real(smoothed.s12[1:] * np.conj(smoothed.s12[:-1])) > 0)
