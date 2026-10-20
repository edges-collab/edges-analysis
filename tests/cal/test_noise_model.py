"""Tests of the analytic noise model of Q."""

import numpy as np
import pytest
from astropy import units as un
from astropy.time import Time
from pygsdata import GSData

from edges import const
from edges.averaging import freqbin
from edges.cal import noise_model as nm
from edges.cal.spectra import get_ave_and_var_spec


def _gsdata(data, loads=("ant",), data_unit="uncalibrated"):
    nload, _, ntime, nfreq = data.shape
    times = np.linspace(2459856, 2459857, ntime + 1)[:-1, None]
    return GSData(
        data=data,
        freqs=np.linspace(50, 100, nfreq) * un.MHz,
        times=Time(np.hstack((times,) * nload), format="jd", scale="utc"),
        telescope=const.KNOWN_TELESCOPES["edges-low"],
        loads=loads,
        data_unit=data_unit,
    )


def test_var_q_limits():
    # Q = 0: n Var = R^2 + R^2 = 2 R^2
    assert np.isclose(nm.var_q(0.0, 2.0, 1.0), 8.0)
    # R = 0: (Q)^2 + Q^2 = 2 Q^2
    assert np.isclose(nm.var_q(3.0, 0.0, 10.0), 1.8)
    # next order vanishes as n_eff -> inf
    v1 = nm.var_q(0.5, 1.5, 1e9)
    assert np.isclose(nm.var_q(0.5, 1.5, 1e9, next_order=True), v1, rtol=1e-8)


def test_var_q_monte_carlo():
    rng = np.random.default_rng(1)
    n, nsim = 1000, 400_000
    t_l, t_ns, t_src = 1.0, 2.0, 1.5  # R = 0.5, Q = 0.25
    p = {
        k: rng.gamma(n, t / n, size=nsim)
        for k, t in (("src", t_src), ("l", t_l), ("lns", t_l + t_ns))
    }
    q = (p["src"] - p["l"]) / (p["lns"] - p["l"])
    pred = nm.var_q(0.25, 0.5, n, next_order=True)
    assert np.isclose(q.var(), pred, rtol=0.01)


def test_channel_correlation_bh():
    rho = nm.channel_power_correlation(nm.blackman_harris(4096), 5)
    assert rho[0] == pytest.approx(1)
    assert 0.6 < rho[1] < 0.7  # BH4 window
    assert rho[5] < 1e-4


def test_gauss_smoothing_factor_bh():
    # Variance of the 8-channel Gaussian smoothing relative to a raw channel, with the
    # channel correlations of the BH4 window (0.2232, Murray et al. in prep.).
    rho = nm.channel_power_correlation(nm.blackman_harris(65536), 8)
    mat = nm.smoothing_matrix(400, 8, kind="gauss", decimate_at=0)
    v = nm.propagate_channel_covariance(np.ones(400), mat, rho)
    assert v[25] == pytest.approx(0.2232, abs=2e-4)


@pytest.mark.parametrize("decimate_at", [0, 4])
def test_smoothing_matrix_matches_gauss_smooth(decimate_at):
    rng = np.random.default_rng(2)
    nf = 200
    x = rng.normal(size=nf)
    data = _gsdata(x[None, None, None])
    sm = freqbin.gauss_smooth(data, size=8, decimate_at=decimate_at)
    mat = nm.smoothing_matrix(nf, 8, kind="gauss", decimate_at=decimate_at)
    np.testing.assert_allclose(mat @ x, sm.data[0, 0, 0])


def test_smoothing_matrix_bin_matches_freq_bin():
    rng = np.random.default_rng(3)
    nf = 203
    x = rng.normal(size=nf)
    data = _gsdata(x[None, None, None])
    sm = freqbin.freq_bin(data, bins=8)
    mat = nm.smoothing_matrix(nf, 8, kind="bin")
    np.testing.assert_allclose(mat @ x, sm.data[0, 0, 0])


def test_propagate_independent_and_correlated():
    nf = 400
    mat = nm.smoothing_matrix(nf, 8, kind="gauss", decimate_at=0)
    v = np.ones(nf)
    # independent channels: sum of squared kernel weights
    out = nm.propagate_channel_covariance(v, mat, np.array([1.0]))
    np.testing.assert_allclose(out, (mat.multiply(mat)).sum(axis=1))
    # Monte Carlo with correlated channels (MA(1) process with rho_1 = 0.4)
    rng = np.random.default_rng(4)
    theta = 0.5  # rho_1 = theta / (1 + theta^2) = 0.4
    e = rng.normal(size=(20000, nf + 1))
    x = (e[:, 1:] + theta * e[:, :-1]) / np.sqrt(1 + theta**2)
    sm = (mat @ x.T).T
    pred = nm.propagate_channel_covariance(v, mat, np.array([1.0, 0.4]), full=True)
    np.testing.assert_allclose(sm[:, 10:20].var(0), np.diag(pred)[10:20], rtol=0.05)
    c = np.cov(sm[:, 10], sm[:, 11])[0, 1]
    assert c == pytest.approx(pred[10, 11], rel=0.1)


@pytest.mark.parametrize("freq_bin_size", [1, 4])
def test_analytic_variance_of_load_spectrum(freq_bin_size):
    """The analytic variance of Q matches simulated three-position spectra."""
    rng = np.random.default_rng(5)
    n_eff, ntime, nfreq = 20000, 3000, 64
    nu = np.linspace(50, 100, nfreq)
    t_l = 300 + 100 * (nu / 75) ** -2  # internal load + receiver
    t_ns = 1000 * np.ones(nfreq)
    t_src = 500 * (nu / 75) ** -2.5 + 200
    p = np.stack([
        rng.gamma(n_eff, t / n_eff, size=(1, ntime, nfreq))
        for t in (t_src, t_l, t_l + t_ns)
    ])
    data = _gsdata(
        p,
        loads=("ant", "internal_load", "internal_load_plus_noise_source"),
        data_unit="power",
    )
    kw = {"freq_bin_size": freq_bin_size, "frequency_smoothing": "bin"}
    _, var_emp = get_ave_and_var_spec(data, **kw)
    _, var_ana = get_ave_and_var_spec(
        data,
        variance_model="analytic",
        n_eff=n_eff,
        channel_correlation=np.array([1.0]),
        **kw,
    )
    ratio = var_emp.data[0, 0, 0] / var_ana.data[0, 0, 0]
    # each empirical variance has a relative error of ~ sqrt(2/ntime) = 2.6%
    assert np.abs(np.mean(ratio) - 1) < 3 * 0.026 / np.sqrt(ratio.size)
    assert np.all(np.abs(ratio - 1) < 5 * 0.026)


def test_smoothing_matrix_weighted_matches_gauss_smooth():
    rng = np.random.default_rng(6)
    nf = 200
    x = rng.normal(size=nf)
    n = rng.integers(0, 5, size=nf).astype(float)
    data = _gsdata(x[None, None, None]).update(nsamples=n[None, None, None])
    sm = freqbin.gauss_smooth(data, size=8, decimate_at=0, use_nsamples=True)
    mat = nm.smoothing_matrix(nf, 8, kind="gauss", decimate_at=0, mask=n)
    ok = sm.nsamples[0, 0, 0] > 0
    np.testing.assert_allclose((mat @ x)[ok], sm.data[0, 0, 0][ok])


def test_variance_nsamples_is_number_of_integrations():
    """After Gaussian smoothing, the variance carries the number of integrations."""
    rng = np.random.default_rng(8)
    ntime, nfreq = 50, 64
    p = (
        rng.gamma(1000, 1 / 1000, size=(3, 1, ntime, nfreq))
        * np.array([1.5, 1.0, 3.0])[:, None, None, None]
    )
    data = _gsdata(
        p,
        loads=("ant", "internal_load", "internal_load_plus_noise_source"),
        data_unit="power",
    )
    mean, var = get_ave_and_var_spec(data, freq_bin_size=8, frequency_smoothing="gauss")
    assert np.allclose(var.nsamples, ntime)
    assert not np.allclose(mean.nsamples, ntime)  # kernel-weighted sums


def test_n_eff_ratio_fastspec_settings():
    nfft = 1024
    # pxspec: one block, Blackman-Harris
    assert nm.n_eff_ratio(nm.blackman_harris(nfft), nfft) == pytest.approx(1.0)
    # fastspec, three taps, no window: r = (2/3, 1/3) -> 9/19
    assert nm.n_eff_ratio(np.ones(3 * nfft), nfft) == pytest.approx(9 / 19)
    # fastspec, three taps, Blackman-Harris over the three blocks
    assert nm.n_eff_ratio(nm.blackman_harris(3 * nfft), nfft) == pytest.approx(
        0.891, abs=2e-3
    )


def test_n_eff_ratio_monte_carlo():
    """Accumulated overlapping unwindowed 3-tap outputs of white noise."""
    rng = np.random.default_rng(9)
    nfft, taps, nacc, nsim = 64, 3, 200, 2000
    x = rng.normal(size=(nsim, (nacc + taps - 1) * nfft))
    blocks = x.reshape(nsim, -1, nfft)
    summed = sum(blocks[:, t : t + nacc] for t in range(taps))
    p = np.abs(np.fft.rfft(summed, axis=-1)[..., 5:25]) ** 2  # away from DC/Nyquist
    acc = p.sum(axis=1)
    neff = acc.mean(axis=0) ** 2 / acc.var(axis=0)
    assert np.mean(neff) / nacc == pytest.approx(9 / 19, rel=0.03)


def test_input_validation():
    with pytest.raises(ValueError, match="multiple of nfft"):
        nm.frame_correlation(np.ones(100), 64)
    with pytest.raises(ValueError, match="kind must be"):
        nm.smoothing_matrix(64, 8, kind="boxcar")


def test_smoothing_matrix_defaults_and_flagged_rows():
    # decimate_at defaults to size // 2, as in gauss_smooth
    mat = nm.smoothing_matrix(64, 8)
    assert mat.shape == (len(range(4, 64, 8)), 64)
    # an output whose whole kernel support is flagged has an empty row
    mask = np.ones(200, dtype=bool)
    mask[:60] = False
    mat = nm.smoothing_matrix(200, 8, kind="gauss", decimate_at=0, nsmooth=1, mask=mask)
    assert mat[[0], :].nnz == 0
    assert mat[[-1], :].nnz > 0


def _three_position_data(ntime=200, nfreq=64, n_eff=20000, seed=10):
    rng = np.random.default_rng(seed)
    t = (500.0, 300.0, 1300.0)
    p = np.stack([rng.gamma(n_eff, ti / n_eff, size=(1, ntime, nfreq)) for ti in t])
    return _gsdata(
        p,
        loads=("ant", "internal_load", "internal_load_plus_noise_source"),
        data_unit="power",
    )


def test_get_ave_and_var_spec_validation():
    data = _three_position_data()
    with pytest.raises(ValueError, match="n_eff is required"):
        get_ave_and_var_spec(data, variance_model="analytic")
    with pytest.raises(ValueError, match="variance_model must be"):
        get_ave_and_var_spec(data, variance_model="other")


def test_analytic_variance_gauss_default_correlation():
    """Gaussian smoothing with the default (Blackman-Harris) channel correlation."""
    data = _three_position_data()
    _, var = get_ave_and_var_spec(
        data,
        variance_model="analytic",
        n_eff=20000,
        freq_bin_size=8,
        frequency_smoothing="gauss",
    )
    raw = nm.var_q(500.0 / 1000 - 300.0 / 1000, 0.3, 20000)  # Q = 0.2, R = 0.3
    # interior channels: 0.2232 of the raw-channel variance
    np.testing.assert_allclose(var.data[0, 0, 0][2:-2] / raw, 0.2232, rtol=0.01)
    assert np.all(var.nsamples == 200)


def test_variance_of_averaged_q():
    from edges.cal.spectra import LoadSpectrum

    data = _three_position_data()
    mean, var = get_ave_and_var_spec(
        data, variance_model="analytic", n_eff=20000, channel_correlation=np.ones(1)
    )
    spec = LoadSpectrum(q=mean, variance=var, temp_ave=300 * un.K)
    np.testing.assert_allclose(spec.variance_of_averaged_q, spec.variance_q / 200)
