r"""Analytic model of the thermal noise of the three-position-switch ratio Q.

For a single integration of the three switch positions (source, internal load, and
internal load plus noise source), each a spectrum averaged over ``n_eff`` independent
power estimates, the variance of :math:`Q = (P_{\rm src} - P_{\rm L})/(P_{\rm LNS} -
P_{\rm L})` in a frequency channel is, to leading order in :math:`1/n_{\rm eff}`,

.. math:: n_{\rm eff} {\rm Var}(Q) = (Q + R)^2 + Q^2 (1 + R)^2 + R^2 (1 - Q)^2,

where :math:`R = \langle P_{\rm L}\rangle / \langle P_{\rm LNS} - P_{\rm L}\rangle`
(Murray et al., in prep.). For the EDGES-2 ``pxspec`` spectrometer, ``n_eff`` equals
the number of accumulated FFT blocks per switch position.

The powers of neighbouring channels of a windowed FFT are correlated, so averaging or
smoothing over channels reduces the variance by less than for independent channels.
:func:`smoothing_matrix` and :func:`propagate_channel_covariance` propagate the
variance of the raw channels through the frequency averaging of
:func:`edges.averaging.freqbin.gauss_smooth` and
:func:`edges.averaging.freqbin.freq_bin`, including these correlations.
"""

from typing import Literal

import numpy as np
from scipy import sparse

BLACKMAN_HARRIS_4 = (0.35875, 0.48829, 0.14128, 0.01168)


def var_q(
    q: np.ndarray, r: np.ndarray, n_eff: float, next_order: bool = False
) -> np.ndarray:
    """Variance of Q for a single integration, in each channel.

    Parameters
    ----------
    q
        The (mean) ratio Q in each channel.
    r
        The (mean) ratio R = P_L / (P_LNS - P_L) in each channel.
    n_eff
        The effective number of independent power estimates in each switch position.
    next_order
        Whether to include the term of order 1/n_eff^2.
    """
    v = (q + r) ** 2 + q**2 * (1 + r) ** 2 + r**2 * (1 - q) ** 2
    if next_order:
        v = (
            v
            + (
                q**2 * (32 * r**4 + 64 * r**3 + 58 * r**2 + 26 * r + 7)
                - q * (32 * r**4 + 28 * r**3 + 4 * r**2 - 6 * r)
                + 17 * r**4
                + 16 * r**3
                + 6 * r**2
            )
            / n_eff
        )
    return v / n_eff


def blackman_harris(n: int) -> np.ndarray:
    """The symmetric four-term Blackman-Harris window (as used by ``pxspec``)."""
    k = np.arange(n)
    return sum(
        (-1) ** i * a * np.cos(2 * np.pi * i * k / (n - 1))
        for i, a in enumerate(BLACKMAN_HARRIS_4)
    )


def channel_power_correlation(
    window: np.ndarray, max_lag: int = 8, nfft: int | None = None
) -> np.ndarray:
    r"""Correlation between the powers of channels ``lag = 0..max_lag`` apart.

    For a windowed FFT (or a polyphase filter bank with window ``w`` spanning several
    blocks of ``nfft`` samples) of Gaussian noise whose spectrum is flat over a few
    channels, the correlation is :math:`|\sum_n w_n^2 e^{2\pi i\,{\rm lag}\,n/N}|^2 /
    (\sum_n w_n^2)^2`, with :math:`N` the FFT length (``nfft``; by default the length
    of the window). The same correlation applies to the fluctuations of Q.
    """
    nfft = window.size if nfft is None else nfft
    n = np.arange(window.size)
    w2 = window**2
    return np.array([
        np.abs(np.sum(w2 * np.exp(2j * np.pi * k * n / nfft))) ** 2 / w2.sum() ** 2
        for k in range(max_lag + 1)
    ])


def frame_correlation(window: np.ndarray, nfft: int) -> np.ndarray:
    r"""Correlation of consecutive outputs of an overlapping (polyphase) channelizer.

    For a channelizer whose outputs are FFTs of the window-weighted sum of ``T``
    consecutive blocks of ``nfft`` samples (``window`` has length ``T * nfft``), with
    one output per block, outputs ``k`` blocks apart have, for noise that is white
    across a channel, the complex correlation
    :math:`r_k = \sum_n w(n)\,w(n + kN) / \sum_n w(n)^2` (``k = 0..T-1``).
    """
    if window.size % nfft:
        raise ValueError("the window length must be a multiple of nfft")
    taps = window.size // nfft
    s = np.sum(window**2)
    return np.array([
        np.sum(window[: window.size - k * nfft] * window[k * nfft :]) / s
        for k in range(taps)
    ])


def n_eff_ratio(window: np.ndarray, nfft: int) -> float:
    r"""Effective number of independent spectra per accumulated output.

    For :math:`N_{\rm acc}` accumulated outputs of an overlapping channelizer (see
    :func:`frame_correlation`), the radiometer relation holds with
    :math:`n_{\rm eff} = N_{\rm acc} / (1 + 2\sum_{k \geq 1} r_k^2)` (the limit of
    many outputs of the result of Welch 1967 for overlapping segments). Returns
    :math:`n_{\rm eff} / N_{\rm acc}`. For a single-block windowed FFT (EDGES-2
    ``pxspec``) it is 1; for ``fastspec`` with three taps and no window it is 9/19.
    """
    r = frame_correlation(window, nfft)
    return float(1 / (1 + 2 * np.sum(r[1:] ** 2)))


def smoothing_matrix(
    nchan: int,
    size: int,
    kind: Literal["gauss", "bin"] = "gauss",
    decimate_at: int | None = None,
    nsmooth: int = 4,
    mask: np.ndarray | None = None,
) -> sparse.csr_array:
    """The linear operator of the frequency averaging applied to a spectrum.

    Parameters
    ----------
    nchan
        The number of input channels.
    size
        The decimation factor (and, for ``kind='bin'``, the bin size).
    kind
        'gauss' for :func:`~edges.averaging.freqbin.gauss_smooth` (with
        ``decimate=True`` and the same ``decimate_at`` and ``nsmooth``), or 'bin'
        for contiguous bins of ``size`` channels as in
        :func:`~edges.averaging.freqbin.freq_bin` (which drops a partial last bin).
    mask
        Boolean array of input channels that are used (True) or flagged (False), or
        non-negative weights of the input channels (e.g. the nsamples used by
        :func:`~edges.averaging.freqbin.gauss_smooth` with ``use_nsamples=True``).
        Each output is normalized by the sum of the weighted kernel.

    Returns
    -------
    A sparse ``(nout, nchan)`` matrix ``A`` such that the averaged spectrum is
    ``A @ x``.
    """
    if mask is None:
        mask = np.ones(nchan, dtype=bool)
    mask = np.where(np.isfinite(mask), mask, 0).astype(float)

    if kind == "gauss":
        if decimate_at is None:
            decimate_at = size // 2
        y = np.arange(-size * nsmooth, size * nsmooth + 1) * 2 / size
        kernel = np.exp(-(y**2) * 0.69)
        offsets = np.arange(-size * nsmooth, size * nsmooth + 1)
        centres = np.arange(decimate_at, nchan, size)
    elif kind == "bin":
        kernel = np.ones(size)
        offsets = np.arange(size)
        centres = np.arange(0, nchan - size + 1, size)  # as freq_bin: no partial bin
    else:
        raise ValueError("kind must be 'gauss' or 'bin'")

    rows, cols, vals = [], [], []
    for i, c in enumerate(centres):
        j = c + offsets
        ok = (j >= 0) & (j < nchan)
        j, w = j[ok], kernel[ok] * mask[j[ok]]
        if w.sum() > 0:
            rows.append(np.full(j.size, i))
            cols.append(j)
            vals.append(w / w.sum())
    return sparse.csr_array(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(centres), nchan),
    )


def propagate_channel_covariance(
    variance: np.ndarray,
    matrix: sparse.csr_array,
    rho: np.ndarray,
    full: bool = False,
) -> np.ndarray:
    """Propagate per-channel variances, with channel correlations, through averaging.

    Parameters
    ----------
    variance
        The variance of each input channel (1D).
    matrix
        The averaging operator, e.g. from :func:`smoothing_matrix`.
    rho
        The correlation between channels ``0, 1, ..., len(rho)-1`` apart (``rho[0]=1``),
        e.g. from :func:`channel_power_correlation`.
    full
        Whether to return the full covariance of the outputs (otherwise its diagonal).

    Returns
    -------
    The variance (or covariance matrix) of the averaged channels. Non-finite input
    variances are treated as zero, so flagged channels must be masked in ``matrix``.
    """
    s = np.where(np.isfinite(variance), np.sqrt(np.abs(variance)), 0.0)
    n = s.size
    lags = np.arange(-(len(rho) - 1), len(rho))
    corr = sparse.diags_array(
        [np.full(n - abs(k), rho[abs(k)]) for k in lags], offsets=lags, shape=(n, n)
    )
    sd = sparse.diags_array(s)
    cov = matrix @ (sd @ corr @ sd) @ matrix.T
    if full:
        return cov.toarray()
    return cov.diagonal()
