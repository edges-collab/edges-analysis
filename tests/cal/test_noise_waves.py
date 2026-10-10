"""Tests of the noise-wave fitting (iterative) procedure."""

from collections import deque

import numpy as np
import pytest
from astropy import units as un

from edges import modeling as mdl
from edges.cal import Calibrator
from edges.cal import noise_waves as nw

N = 500
FREQ = np.linspace(50, 100, N)


def gamma_zero(freq):
    return np.zeros(len(freq), dtype=complex)


def gamma_low(freq):
    return 1e-5 * np.exp(-1j * freq / 12)


def gamma_high(freq):
    return 1e-2 * np.exp(1j * freq / 6)


def gamma_decay(freq):
    return (freq / 75) ** -1 * np.exp(-1j * freq / 12)


def gamma_decay_flip(freq):
    return (freq / 75) ** -1 * np.exp(1j * freq / 12)


@pytest.mark.parametrize(
    ("true_sca", "true_off", "true_t_unc", "true_t_cos", "true_t_sin"),
    [
        (np.ones(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)),
        (np.linspace(3.5, 4.5, N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)),
        (np.linspace(3.5, 4.5, N), -np.ones(N), np.zeros(N), np.zeros(N), np.zeros(N)),
        (
            np.linspace(3.5, 4.5, N),
            -np.ones(N),
            np.linspace(19.0, 20.0, N),
            np.zeros(N),
            np.zeros(N),
        ),
        (
            np.linspace(3.5, 4.5, N),
            -np.ones(N),
            np.linspace(19.0, 20.0, N),
            np.linspace(4, 5, N),
            np.linspace(10, 15, N),
        ),
    ],
    ids=[
        "trivial",
        "non-zero scale",
        "non-zero scale and off",
        "non-zero tunc",
        "all non-zero",
    ],
)
@pytest.mark.parametrize(
    "gamma_rec",
    [gamma_zero, gamma_low, gamma_high],
    ids=["perfect_rx", "low-level-rx", "high-level-rx"],
)
@pytest.mark.parametrize(
    "gamma_amb",
    [gamma_zero, gamma_low],
    ids=["perfect_ra", "low-level-ra"],
)
def test_fit_perfect_receiver(
    true_sca, true_off, true_t_unc, true_t_cos, true_t_sin, gamma_rec, gamma_amb
):
    """Test that noise-wave fits work."""
    gamma_ant = {
        "ambient": gamma_amb(FREQ),
        "hot_load": gamma_amb(FREQ),
        "short": gamma_decay(FREQ),
        "open": gamma_decay_flip(FREQ),
    }

    temp = {
        "ambient": 300 * un.K,
        "hot_load": 1000 * un.K,
        "short": 300 * un.K,
        "open": 300 * un.K,
    }

    calibrator = Calibrator(
        freqs=FREQ * un.MHz,
        Tsca=true_sca * 1000,
        Toff=300 - true_off,
        Tcos=true_t_cos,
        Tsin=true_t_sin,
        Tunc=true_t_unc,
        receiver_s11=gamma_rec(FREQ),
    )

    Q = {
        k: calibrator.decalibrate(
            temp=temp[k],
            ant_s11=gamma_ant[k],
        )
        for k in temp
    }
    print({name: np.where(~np.isfinite(q)) for name, q in Q.items()})
    result = deque(
        nw.get_calibration_quantities_iterative(
            freqs=FREQ * un.MHz,
            source_q=Q,
            receiver_s11=gamma_rec(FREQ),
            source_s11s=gamma_ant,
            source_true_temps=temp,
            cterms=5,
            wterms=5,
        ),
        maxlen=1,
    )
    sca, off, nwv = result.pop()

    assert np.allclose(sca(FREQ), calibrator.Tsca)
    assert np.allclose(off(FREQ), calibrator.Toff)
    assert np.allclose(nwv.get_tunc(), calibrator.Tunc)
    assert np.allclose(nwv.get_tcos(), calibrator.Tcos)
    assert np.allclose(nwv.get_tsin(), calibrator.Tsin)


def test_noise_waves(calobs):
    print("RECEIVER S11:", calobs.receiver.s11)
    nwm = nw.NoiseWaves.from_calobs(calobs, cterms=5, wterms=5)

    assert isinstance(nwm.linear_model, mdl.FixedLinearModel)
    assert isinstance(nwm.linear_model.model, mdl.CompositeModel)

    assert "ambient" in nwm.src_names
    with pytest.raises(
        ValueError, match="Cannot evaluate a model without providing parameters"
    ):
        nwm.get_noise_wave("tunc", src="ambient")
    with pytest.raises(
        ValueError, match="You must supply parameters to evaluate the model"
    ):
        nwm.get_full_model("hot_load")

    nok = nwm.get_linear_model(with_k=False)
    assert all(m.basis_scaler is None for m in nok.model.models.values())


class TestGetK:
    """Limits and passivity of the K-factors of Monsalve+17 Eq. 7."""

    def setup_class(self):
        rng = np.random.default_rng(1234)
        n = 1000
        # Passive reflection coefficients, |Γ| < 1.
        self.gamma_a = np.sqrt(rng.uniform(0, 0.99, n)) * np.exp(
            2j * np.pi * rng.uniform(size=n)
        )
        self.gamma_r = np.sqrt(rng.uniform(0, 0.99, n)) * np.exp(
            2j * np.pi * rng.uniform(size=n)
        )

    def test_matched_antenna(self):
        k = nw.get_K(gamma_rec=self.gamma_r, gamma_ant=np.zeros_like(self.gamma_a))
        np.testing.assert_allclose(k[0], 1, rtol=0, atol=1e-14)
        np.testing.assert_allclose(k[1], 0, rtol=0, atol=0)
        np.testing.assert_allclose(k[2], 0, rtol=0, atol=0)
        np.testing.assert_allclose(k[3], 0, rtol=0, atol=0)

    def test_matched_receiver(self):
        ga = self.gamma_a
        k = nw.get_K(gamma_rec=np.zeros_like(self.gamma_r), gamma_ant=ga)
        mag = np.abs(ga)
        phase = np.angle(ga)
        np.testing.assert_allclose(k[0], 1 - mag**2, rtol=0, atol=1e-14)
        np.testing.assert_allclose(k[1], mag**2, rtol=0, atol=1e-14)
        np.testing.assert_allclose(k[2], mag * np.cos(phase), rtol=0, atol=1e-14)
        np.testing.assert_allclose(k[3], mag * np.sin(phase), rtol=0, atol=1e-14)

    def test_passivity(self):
        """The mismatch factor K1*(1-|Γr|²) is the transducer gain, in [0, 1].

        Note that K1 itself is *not* bounded by 1: K1 = (1-|Γa|²)/|1-ΓaΓr|², which
        exceeds 1 when Γa and Γr are aligned in phase.
        """
        k = nw.get_K(gamma_rec=self.gamma_r, gamma_ant=self.gamma_a)
        gain = 1 - np.abs(self.gamma_r) ** 2
        mismatch = k[0] * gain
        assert np.all(mismatch >= 0)
        assert np.all(mismatch <= 1 + 1e-12)

        # K2 >= 0, and K3² + K4² = |Γa|²|F|²/(1-|Γr|²)² = K2/(1-|Γr|²).
        assert np.all(k[1] >= 0)
        np.testing.assert_allclose(
            k[2] ** 2 + k[3] ** 2, k[1] / gain, rtol=1e-10, atol=0
        )


def _poly(coeffs, x):
    return sum(c * x**i for i, c in enumerate(coeffs))


@pytest.mark.parametrize("use_loss", [False, True], ids=["no-loss", "hot-load-loss"])
def test_recover_polynomial_noise_waves(use_loss):
    """A synthetic receiver with polynomial noise waves is recovered exactly.

    This includes the ``hot_load_loss`` path, in which the "true" hot-load
    temperature is the physical temperature at the far end of a lossy cable.
    """
    x = FREQ / 75.0
    # Polynomials with fewer terms than cterms/wterms, so the models are exact.
    t_sca = _poly([1500, 60, -20], x)
    t_off = _poly([310, -12, 3], x)
    t_unc = _poly([25, -5, 2], x)
    t_cos = _poly([8, 4, -3], x)
    t_sin = _poly([-12, 3, 1], x)
    gamma_rec = 0.05 * np.exp(1j * FREQ / 9)

    calibrator = Calibrator(
        freqs=FREQ * un.MHz,
        Tsca=t_sca,
        Toff=t_off,
        Tunc=t_unc,
        Tcos=t_cos,
        Tsin=t_sin,
        receiver_s11=gamma_rec,
    )

    gamma_ant = {
        "ambient": 0.01 * np.exp(-1j * FREQ / 30),
        "hot_load": 0.02 * np.exp(1j * FREQ / 25),
        "short": gamma_decay(FREQ),
        "open": gamma_decay_flip(FREQ),
    }

    t_amb = 298.0 * un.K
    t_hot = 370.0 * un.K
    loss = 0.98 - 0.02 * x if use_loss else None

    # The temperature seen at the receiver input.
    t_hot_eff = loss * t_hot + (1 - loss) * t_amb if use_loss else t_hot
    temp_at_input = {
        "ambient": t_amb,
        "hot_load": t_hot_eff,
        "short": 300 * un.K,
        "open": 300 * un.K,
    }
    q = {
        k: calibrator.decalibrate(temp=temp_at_input[k], ant_s11=gamma_ant[k])
        for k in temp_at_input
    }

    true_temps = dict(temp_at_input)
    if use_loss:
        true_temps["hot_load"] = t_hot

    sca, off, nwv = deque(
        nw.get_calibration_quantities_iterative(
            freqs=FREQ * un.MHz,
            source_q=q,
            receiver_s11=gamma_rec,
            source_s11s=gamma_ant,
            source_true_temps=true_temps,
            cterms=5,
            wterms=5,
            niter=40,  # convergence is geometric; ~1e-10 K by 40 iterations
            hot_load_loss=loss,
        ),
        maxlen=1,
    ).pop()

    np.testing.assert_allclose(sca(FREQ), t_sca, rtol=0, atol=1e-7)
    np.testing.assert_allclose(off(FREQ), t_off, rtol=0, atol=1e-7)
    np.testing.assert_allclose(nwv.get_tunc(), t_unc, rtol=0, atol=1e-7)
    np.testing.assert_allclose(nwv.get_tcos(), t_cos, rtol=0, atol=1e-7)
    np.testing.assert_allclose(nwv.get_tsin(), t_sin, rtol=0, atol=1e-7)


def test_noise_waves_get_fitted_default_weights():
    """get_fitted works with its default weights (uniform weights)."""
    gamma_src = {
        "ambient": gamma_low(FREQ),
        "hot_load": gamma_low(FREQ),
        "open": gamma_decay_flip(FREQ),
        "short": gamma_decay(FREQ),
    }
    nwm = nw.NoiseWaves(
        freq=FREQ, gamma_src=gamma_src, gamma_rec=gamma_high(FREQ), c_terms=3, w_terms=3
    )
    rng = np.random.default_rng(42)
    params = rng.normal(size=nwm.linear_model.model.n_terms)
    data = nwm(parameters=params)

    fitted = nwm.get_fitted(data)
    np.testing.assert_allclose(fitted.parameters, params, rtol=0, atol=1e-8)

    fitted_none = nwm.get_fitted(data, weights=None)
    np.testing.assert_allclose(fitted_none.parameters, params, rtol=0, atol=1e-8)


class _UncachedNoiseWaveLinearModel(nw.NoiseWaveLinearModel):
    """Reference: the original k-factors, recomputing ``get_K`` on every call."""

    def cos_kfactor(self, freq):
        freq = freq[: len(freq) // len(self.gamma_src)]
        ph = np.exp(1j * 2 * np.pi * freq * self.delay * 1e6)
        out = []
        for gamma in self.gamma_src.values():
            K = nw.get_K(self.gamma_rec, gamma)
            out.append(K[2] * ph.real - K[3] * ph.imag)
        return np.concatenate(out)

    def sin_kfactor(self, freq):
        freq = freq[: len(freq) // len(self.gamma_src)]
        ph = np.exp(1j * 2 * np.pi * freq * self.delay * 1e6)
        out = []
        for gamma in self.gamma_src.values():
            K = nw.get_K(self.gamma_rec, gamma)
            out.append(K[2] * ph.imag + K[3] * ph.real)
        return np.concatenate(out)

    def unc_kfactor(self, freq):
        return np.concatenate([
            nw.get_K(self.gamma_rec, gamma)[1] for gamma in self.gamma_src.values()
        ])

    @property
    def _source_k(self):
        return {k: nw.get_K(self.gamma_rec, g) for k, g in self.gamma_src.items()}


def _random_s11(rng, n, scale):
    return scale * rng.uniform(0.2, 1, n) * np.exp(2j * np.pi * rng.uniform(size=n))


class TestCachedKFactors:
    """Caching the K-factors must not change any result (bit-for-bit)."""

    @pytest.mark.parametrize("delay", [0.0, -3.5e-9, 1e-8])
    def test_kfactors_equal_reference(self, delay):
        rng = np.random.default_rng(1)
        kw = {
            "freq": FREQ,
            "gamma_rec": _random_s11(rng, N, 0.1),
            "gamma_src": {
                "open": _random_s11(rng, N, 0.9),
                "short": -gamma_decay(FREQ),
            },
            "delay": delay,
        }
        new = nw.NoiseWaveLinearModel(**kw)
        ref = _UncachedNoiseWaveLinearModel(**kw)
        x = np.tile(FREQ, 2)
        for name in ("unc_kfactor", "cos_kfactor", "sin_kfactor"):
            # Call twice so that the second call uses the cache.
            for _ in range(2):
                np.testing.assert_array_equal(
                    getattr(new, name)(x), getattr(ref, name)(x)
                )

    def test_cache_cleared_on_reassignment(self):
        rng = np.random.default_rng(2)
        nwlm = nw.NoiseWaveLinearModel(
            freq=FREQ,
            gamma_rec=_random_s11(rng, N, 0.1),
            gamma_src={"open": _random_s11(rng, N, 0.9)},
        )
        first = nwlm.unc_kfactor(FREQ)
        nwlm.gamma_rec = _random_s11(rng, N, 0.1)
        second = nwlm.unc_kfactor(FREQ)
        assert not np.array_equal(first, second)
        np.testing.assert_array_equal(
            second, nw.get_K(nwlm.gamma_rec, nwlm.gamma_src["open"])[1]
        )
        nwlm.gamma_src = {"open": _random_s11(rng, N, 0.9)}
        np.testing.assert_array_equal(
            nwlm.unc_kfactor(FREQ),
            nw.get_K(nwlm.gamma_rec, nwlm.gamma_src["open"])[1],
        )

    def test_phase_follows_freq_and_delay(self):
        rng = np.random.default_rng(5)
        kw = {
            "freq": FREQ,
            "gamma_rec": _random_s11(rng, N, 0.1),
            "gamma_src": {"open": _random_s11(rng, N, 0.9)},
            "delay": -2e-9,
        }
        nwlm = nw.NoiseWaveLinearModel(**kw)
        other_freq = FREQ + 0.5
        for freq in (FREQ, other_freq, other_freq, FREQ):
            ref = _UncachedNoiseWaveLinearModel(**kw)
            np.testing.assert_array_equal(nwlm.cos_kfactor(freq), ref.cos_kfactor(freq))
            np.testing.assert_array_equal(nwlm.sin_kfactor(freq), ref.sin_kfactor(freq))

        # The memoised phase must not survive in-place changes to the input.
        freq = FREQ.copy()
        nwlm.cos_kfactor(freq)
        freq += 1.0
        np.testing.assert_array_equal(
            nwlm.cos_kfactor(freq),
            _UncachedNoiseWaveLinearModel(**kw).cos_kfactor(freq),
        )

        nwlm.delay = 5e-9
        kw["delay"] = 5e-9
        np.testing.assert_array_equal(
            nwlm.sin_kfactor(FREQ),
            _UncachedNoiseWaveLinearModel(**kw).sin_kfactor(FREQ),
        )

    def test_noise_waves_linear_model_equal_reference(self, monkeypatch):
        rng = np.random.default_rng(3)
        gamma_src = {
            "ambient": _random_s11(rng, N, 0.01),
            "hot_load": _random_s11(rng, N, 0.02),
            "open": gamma_decay_flip(FREQ),
            "short": gamma_decay(FREQ),
        }
        kw = {
            "freq": FREQ,
            "gamma_src": gamma_src,
            "gamma_rec": _random_s11(rng, N, 0.1),
            "c_terms": 4,
            "w_terms": 5,
        }
        basis = nw.NoiseWaves(**kw).linear_model.basis
        monkeypatch.setattr(nw, "NoiseWaveLinearModel", _UncachedNoiseWaveLinearModel)
        np.testing.assert_array_equal(basis, nw.NoiseWaves(**kw).linear_model.basis)

    @pytest.mark.parametrize("smooth", [True, False])
    def test_iterative_calibration_equal_reference(self, monkeypatch, smooth):
        rng = np.random.default_rng(4)
        gamma_rec = _random_s11(rng, N, 0.05)
        gamma_ant = {
            "ambient": _random_s11(rng, N, 0.01),
            "hot_load": _random_s11(rng, N, 0.02),
            "short": gamma_decay(FREQ),
            "open": gamma_decay_flip(FREQ),
        }
        x = FREQ / 75.0
        calibrator = Calibrator(
            freqs=FREQ * un.MHz,
            Tsca=_poly([1500, 60, -20], x),
            Toff=_poly([310, -12, 3], x),
            Tunc=_poly([25, -5, 2], x),
            Tcos=_poly([8, 4, -3], x),
            Tsin=_poly([-12, 3, 1], x),
            receiver_s11=gamma_rec,
        )
        temps = {
            "ambient": 298.0 * un.K,
            "hot_load": 370.0 * un.K,
            "short": 300 * un.K,
            "open": 300 * un.K,
        }
        q = {
            k: calibrator.decalibrate(temp=t, ant_s11=gamma_ant[k])
            * (1 + 1e-4 * rng.normal(size=N))
            for k, t in temps.items()
        }

        def run():
            return list(
                nw.get_calibration_quantities_iterative(
                    freqs=FREQ * un.MHz,
                    source_q=q,
                    receiver_s11=gamma_rec,
                    source_s11s=gamma_ant,
                    source_true_temps=temps,
                    cterms=4,
                    wterms=5,
                    niter=4,
                    smooth_scale_offset_within_loop=smooth,
                    delays_to_fit=np.array([0.0, -1e-9, -2e-9]),
                )
            )

        new = run()
        monkeypatch.setattr(nw, "NoiseWaveLinearModel", _UncachedNoiseWaveLinearModel)
        ref = run()

        assert len(new) == len(ref) == 4
        for (sca, off, nwv), (sca_r, off_r, nwv_r) in zip(new, ref, strict=True):
            np.testing.assert_array_equal(sca.parameters, sca_r.parameters)
            np.testing.assert_array_equal(off.parameters, off_r.parameters)
            assert nwv.delay == nwv_r.delay
            np.testing.assert_array_equal(nwv.get_tunc(), nwv_r.get_tunc())
            np.testing.assert_array_equal(nwv.get_tcos(), nwv_r.get_tcos())
            np.testing.assert_array_equal(nwv.get_tsin(), nwv_r.get_tsin())
            assert nwv.rms == nwv_r.rms
