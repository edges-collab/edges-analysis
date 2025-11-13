import numpy as np
from astropy import units as un

from edges import modeling as mdl
from edges.cal.sparams import S11ModelParams, SParams, smooth_sparams


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
