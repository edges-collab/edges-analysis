import numpy as np
import pytest
from astropy import units as un

from edges import modeling as mdl
from edges.cal.sparams import (
    AGILENT_85033E,
    CalkitReadings,
    S11ModelParams,
    SParams,
    smooth_sparams,
)


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

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "S12 and S21 from OSL are the principal-branch sqrt of S12*S21, which "
            "flips sign where the phase crosses pi, so smoothing them separately "
            "(combine_s12s21=False) cannot fit them; fix pending (result-changing)"
        ),
    )
    def test_separate(self):
        smoothed = smooth_sparams(self.sparams, params=self._params(False))
        np.testing.assert_allclose(smoothed.s12 * smoothed.s21, self.s12s21, atol=1e-6)
