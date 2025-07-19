"""Functions for calibrating the receiver."""

from __future__ import annotations

from collections.abc import Generator, Sequence
from functools import cached_property

import attrs
import numpy as np

from .. import modelling as mdl
from ..tools import ComplexSpline


def get_F(gamma_rec: np.ndarray, gamma_ant: np.ndarray) -> np.ndarray:  # noqa: N802
    """Get the F parameter for a given receiver and antenna.

    Parameters
    ----------
    gamma_rec : np.ndarray
        The reflection coefficient (S11) of the receiver.
    gamma_ant : np.ndarray
        The reflection coefficient (S11) of the antenna

    Returns
    -------
    F : np.ndarray
        The F parameter (see M17)
    """
    return np.sqrt(1 - np.abs(gamma_rec) ** 2) / (1 - gamma_ant * gamma_rec)


def get_alpha(gamma_rec: np.ndarray, gamma_ant: np.ndarray) -> np.ndarray:
    """Get the alpha parameter for a given receiver and antenna.

    Parameters
    ----------
    gamma_rec : np.ndarray
        The reflection coefficient of the receiver.
    gamma_ant : np.ndarray
        The reflection coefficient fo the antenna.
    """
    return np.angle(gamma_ant * get_F(gamma_rec, gamma_ant))


def get_K(gamma_rec, gamma_ant, f_ratio=None, alpha=None, gain=None):  # noqa: N802
    """
    Determine the S11-dependent factors for each term in Eq. 7 (Monsalve 2017).

    Parameters
    ----------
    gamma_rec : array_like
        Receiver S11
    gamma_ant : array_like
        Antenna (or load) S11.
    f_ratio : array_like, optional
        The F factor (Eq. 3 of Monsalve 2017). Computed if not given.
    alpha : array_like, optional
        The alpha factor (Eq. 4 of Monsalve, 2017). Computed if not given.
    gain : array_like, optional
        The transmission function, (1 - Gamma_rec^2). Computed if not given.

    Returns
    -------
    K0, K1, K2, K3: array_like
        Factors corresponding to T_ant, T_unc, T_cos, T_sin respectively.
    """
    # Get F and alpha for each load (Eqs. 3 and 4)
    if f_ratio is None:
        f_ratio = get_F(gamma_rec=gamma_rec, gamma_ant=gamma_ant)

    if alpha is None:
        alpha = get_alpha(gamma_rec=gamma_rec, gamma_ant=gamma_ant)

    # The denominator of each term in Eq. 7
    if gain is None:
        gain = 1 - np.abs(gamma_rec) ** 2

    f_ratio = np.abs(f_ratio)
    gant = np.abs(gamma_ant)
    fgant = gant * f_ratio / gain

    K2 = fgant**2 * gain
    K1 = f_ratio**2 / gain - K2
    K3 = fgant * np.cos(alpha)
    K4 = fgant * np.sin(alpha)

    return K1, K2, K3, K4


def power_ratio(
    temp_ant,
    gamma_ant,
    gamma_rec,
    scale,
    offset,
    temp_unc,
    temp_cos,
    temp_sin,
    temp_noise_source,
    temp_load,
    return_terms=False,
):
    """
    Compute the ratio of raw powers from the three-position switch.

    Parameters
    ----------
    temp_ant : array_like, shape (NFREQS,)
        Temperature of the antenna, or simulator.
    gamma_ant : array_like, shape (NFREQS,)
        S11 of the antenna (or simulator)
    gamma_rec : array_like, shape (NFREQS,)
        S11 of the receiver.
    scale : :class:`np.poly1d`
        A polynomial representing the C_1 term.
    offset : :class:`np.poly1d`
        A polynomial representing the C_2 term
    temp_unc : :class:`np.poly1d`
        A polynomial representing the uncorrelated noise-wave parameter
    temp_cos : :class:`np.poly1d`
        A polynomial representing the cosine noise-wave parameter
    temp_sin : :class:`np.poly1d`
        A polynomial representing the sine noise-wave parameter
    temp_noise_source : array_like, shape (NFREQS,)
        Temperature of the internal noise source.
    temp_load : array_like, shape (NFREQS,)
        Temperature of the internal load
    return_terms : bool, optional
        If True, return the terms of Qp, rather than the sum of them._

    Returns
    -------
    array_like : the quantity Q_P as a function of frequency.

    Notes
    -----
    Computes (as a model)

    .. math :: Q_P = (P_ant - P_L)/(P_NS - P_L)

    """
    K = get_K(gamma_rec, gamma_ant)

    terms = [
        t * k for t, k in zip([temp_ant, temp_unc, temp_cos, temp_sin], K, strict=False)
    ] + [
        (offset - temp_load),
        scale * temp_noise_source,
    ]

    return terms if return_terms else sum(terms[:5]) / terms[5]


@attrs.define(slots=False)
class NoiseWaveLinearModelFit:
    freq: np.ndarray = attrs.field()
    gamma_rec: callable = attrs.field()
    gamma_src: dict[str, callable] = attrs.field()
    modelfit: mdl.CompositeModel = attrs.field()
    delay: float = attrs.field(default=0.0)

    @property
    def model(self):
        """The model, with assigned parameters from best-fit."""
        return self.modelfit.fit

    def get_tunc(self, freq: np.ndarray | None, full_term=False):
        r"""Compute the model for Tunc(freq).

        If `full_term` is True, return the full term, i.e.

        .. math:: Tunc \frac{|Gamma_{\rm ant}|^2 |F|^2}{1 - |Gamma_{\rm rcv}|^2}

        otherwise just return Tunc itself. This is not implemented yet (it's a little
        trickier because it depends on which source you want it for).
        """
        if freq is None:
            freq = self.freq

        return self.model.model.models["unc"](x=freq, with_scaler=False)

    def get_tcos(self, freq: np.ndarray | None, full_term=False):
        r"""Compute the model for Tcos(freq).

        If `full_term` is True, return the full term, i.e.

        .. math:: Tcos \frac{|Gamma_{\rm ant}| |F| \cos\alpha}{1 - |Gamma_{\rm rcv}|^2}

        otherwise just return Tcos itself. This is not implemented yet (it's a little
        trickier because it depends on which source you want it for).
        """
        if freq is None:
            freq = self.freq

        ph = np.exp(1j * 2 * np.pi * freq * self.delay * 1e6)

        tcos = self.model.model.models["cos"](x=freq, with_scaler=False)
        tsin = self.model.model.models["sin"](x=freq, with_scaler=False)

        return tcos * ph.real + tsin * ph.imag

    def get_tsin(self, freq: np.ndarray | None, full_term=False):
        r"""Compute the model for Tcos(freq).

        If `full_term` is True, return the full term, i.e.

        .. math:: Tcos \frac{|Gamma_{\rm ant}| |F| \cos\alpha}{1 - |Gamma_{\rm rcv}|^2}

        otherwise just return Tcos itself. This is not implemented yet (it's a little
        trickier because it depends on which source you want it for).
        """
        if freq is None:
            freq = self.freq

        ph = np.exp(1j * 2 * np.pi * freq * self.delay * 1e6)

        tcos = self.model.model.models["cos"](x=freq, with_scaler=False)
        tsin = self.model.model.models["sin"](x=freq, with_scaler=False)

        return tsin * ph.real - tcos * ph.imag

    @cached_property
    def residual(self):
        """The residual of the fit."""
        return self.modelfit.residual

    @cached_property
    def rms(self):
        """The RMS of the residual of the fit."""
        return self.modelfit.weighted_rms


@attrs.define(slots=False)
class NoiseWaveLinearModel:
    """
    A linear model for the noise wave terms.

    Parameters
    ----------
    Kopen : tuple
        The K terms for the open load.
    Kshort : tuple
        The K terms for the short load.

    Returns
    -------
    model : :class:`np.poly1d`
        The linear model for the noise wave terms.
    """

    freq: np.ndarray = attrs.field()
    gamma_rec: callable = attrs.field()
    gamma_src: dict[str, callable] = attrs.field()
    model_type: mdl.Model = attrs.field(default=mdl.Polynomial)
    n_terms: int = attrs.field(default=5)
    delay: float = attrs.field(default=0.0)

    def cos_kfactor(self, freq):
        """Compute the scaler to the Tcos basis function."""
        freq = freq[: len(freq) // len(self.gamma_src)]

        ph = np.exp(1j * 2 * np.pi * freq * self.delay * 1e6)

        out = []
        for gamma in self.gamma_src.values():
            K = get_K(self.gamma_rec(freq), gamma(freq))
            out.append(K[2] * ph.real - K[3] * ph.imag)

        return np.concatenate(out)

    def sin_kfactor(self, freq):
        """Compute the scaler to the Tsin basis function."""
        freq = freq[: len(freq) // len(self.gamma_src)]
        ph = np.exp(1j * 2 * np.pi * freq * self.delay * 1e6)

        out = []
        for gamma in self.gamma_src.values():
            K = get_K(self.gamma_rec(freq), gamma(freq))

            out.append(K[2] * ph.imag + K[3] * ph.real)
        return np.concatenate(out)

    def unc_kfactor(self, freq):
        """Compute the scaler to the Tunc basis function."""
        freq = freq[: len(freq) // len(self.gamma_src)]

        return np.concatenate([
            get_K(self.gamma_rec(freq), gamma(freq))[1]
            for gamma in self.gamma_src.values()
        ])

    def fit(
        self,
        spectrum: dict[str, np.ndarray],
        temp_thermistor: dict[str, np.ndarray],
        method="lstsq",
    ):
        """Obtain a fit of the compositie model to given data."""
        x = np.concatenate([self.freq] * len(self.gamma_src))
        tr = mdl.ScaleTransform(scale=self.freq[len(self.freq) // 2])

        unc_model = self.model_type(
            n_terms=self.n_terms,
            xtransform=tr,
            basis_scaler=self.unc_kfactor,
        )

        cos_model = self.model_type(
            n_terms=self.n_terms,
            xtransform=tr,
            basis_scaler=self.cos_kfactor,
        )

        sin_model = self.model_type(
            n_terms=self.n_terms,
            xtransform=tr,
            basis_scaler=self.sin_kfactor,
        )

        model = mdl.CompositeModel(
            models={"unc": unc_model, "cos": cos_model, "sin": sin_model}
        )

        K0 = {
            k: get_K(self.gamma_rec(self.freq), self.gamma_src[k](self.freq))[0]
            for k in self.gamma_src
        }

        data = np.concatenate([
            spectrum[k] - temp_thermistor[k] * K0[k] for k in self.gamma_src
        ])

        return NoiseWaveLinearModelFit(
            freq=self.freq,
            gamma_rec=self.gamma_rec,
            gamma_src=self.gamma_src,
            modelfit=model.fit(xdata=x, ydata=data, method=method),
            delay=self.delay,
        )


def get_calibration_quantities_iterative(
    freq: np.ndarray,
    temp_raw: dict,
    gamma_rec: callable,
    gamma_ant: dict[str, callable],
    temp_ant: dict[str, np.ndarray | float],
    cterms: int,
    wterms: int,
    temp_amb_internal: float = 300,
    niter: int = 4,
    hot_load_loss: np.ndarray | None = None,
    smooth_scale_offset_within_loop: bool = True,
    delays_to_fit: np.ndarray = np.array([0.0]),
    fit_method="lstsq",
    poly_spacing: float = 1.0,
    return_early: bool = False,
) -> Generator[tuple[mdl.Polynomial, mdl.Polynomial, NoiseWaveLinearModelFit]]:
    """
    Derive calibration parameters using the scheme laid out in Monsalve (2017).

    All equation numbers and symbol names come from M17 (arxiv:1602.08065).

    Parameters
    ----------
    f_norm : array_like
        Normalized frequencies (arbitrarily normalised, but standard assumption is
        that the centre is zero, and the scale is such that the range is (-1, 1))
    temp_raw : dict
        Dictionary of antenna uncalibrated temperatures, with keys
        'ambient', 'hot_load, 'short' and 'open'. Each value is an array with the same
        length as f_norm.
    gamma_rec : float array
        Receiver S11 as a function of frequency.
    gamma_ant : dict
        Dictionary of antenna S11, with keys 'ambient', 'hot_load, 'short'
        and 'open'. Each value is an array with the same length as f_norm.
    temp_ant : dict
        Dictionary like `gamma_ant`, except that the values are modelled/smoothed
        thermistor temperatures for each source load.
    cterms : int
        Number of polynomial terms for the C_i
    wterms : int
        Number of polynonmial temrs for the T_i
    temp_amb_internal : float
        The ambient internal temperature, interpreted as T_L.
        Note: this must be the same as the T_L used to generate T*.
    niter : int
        The number of iterations to perform.
    hot_load_loss : array_like, optional
        The loss of the hot load. If None, then either no loss is assumed, or the loss
        is already assumed to be applied to the "true" temperature of the hot load.
    smooth_scale_offset_within_loop : bool
        If True, then the scale and offset are smoothed within the loop. If False, then
        the scale and offset are only smoothed at the end of the loop.
    delays_to_fit : array_like
        The delays to sweep over when fitting noise-wave parameters. The delay resulting
        in the lowest residuals will be used.
    poly_spacing
        Spacing between polynomial term powers for scale and offset. Default of 1.0 is
        for a standard polynomial basis set. Alan's C code use 0.5 for the scale/offset.

    Returns
    -------
    sca, off, tu, tc, ts
        Fitted Models for each of the Scale (C_1), Offset (C_2), and noise-wave
        temperatures for uncorrelated, cos and sin components.

    Notes
    -----
    To achieve the same results as the legacy C pipeline, the `hot_load_loss` parameter
    should be given, and not applied to the "true" temperature. There is a small
    mathematical difference that arises if you do it the other way. Furthermore, the
    `smooth_scale_offset_within_loop` parameter should be set to False.
    """
    mask = np.all([np.isfinite(temp) for temp in temp_raw.values()], axis=0)

    fmask = freq[mask]
    temp_raw = {key: value[mask] for key, value in temp_raw.items()}
    temp_ant = {
        key: (value[mask] if hasattr(value, "__len__") else value)
        for key, value in temp_ant.items()
    }
    temp_ant_hot = temp_ant["hot_load"]

    # The denominator of each term in Eq. 7
    G = 1 - np.abs(gamma_rec(fmask)) ** 2

    K1, K2, K3, K4 = {}, {}, {}, {}
    for k, gamma_a in gamma_ant.items():
        K1[k], K2[k], K3[k], K4[k] = get_K(gamma_rec(fmask), gamma_a(fmask), gain=G)

    # Initialize arrays
    nf = len(fmask)
    tamb_iter = np.zeros(nf)
    thot_iter = np.zeros(nf)

    sca, off, tunc, tcos, tsin = (
        np.ones(nf),
        np.zeros(nf),
        np.zeros(nf),
        np.zeros(nf),
        np.zeros(nf),
    )

    tr = mdl.ScaleTransform(scale=freq[len(freq) // 2])
    sca_mdl = mdl.Polynomial(n_terms=cterms, transform=tr, spacing=poly_spacing).at(
        x=fmask
    )
    off_mdl = mdl.Polynomial(n_terms=cterms, transform=tr, spacing=poly_spacing).at(
        x=fmask
    )

    temp_cal_iter = dict(temp_raw)  # copy

    # Initial values for breaking early.
    sca_off_chisq = np.inf
    cable_chisq = np.inf
    best = None

    # Calibration loop
    for _ in range(niter):
        # Step 1: approximate physical temperature
        nwp = tunc * K2["ambient"] + tcos * K3["ambient"] + tsin * K4["ambient"]
        tamb_iter = (temp_cal_iter["ambient"] - nwp) / K1["ambient"]

        nwp = tunc * K2["hot_load"] + tcos * K3["hot_load"] + tsin * K4["hot_load"]
        thot_iter = (temp_cal_iter["hot_load"] - nwp) / K1["hot_load"]

        # Step 2: scale and offset
        if hot_load_loss is not None:
            thot_iter = (
                thot_iter - temp_ant["ambient"] * (1 - hot_load_loss)
            ) / hot_load_loss

        # Updating scale and offset
        sca_new = (temp_ant_hot - temp_ant["ambient"]) / (thot_iter - tamb_iter)
        off_new = tamb_iter - temp_ant["ambient"]

        sca *= sca_new
        off += off_new

        # Model scale and offset
        p_sca = sca_mdl.fit(ydata=sca, method=fit_method).fit
        p_off = off_mdl.fit(ydata=off, method=fit_method).fit

        if smooth_scale_offset_within_loop:
            sca = p_sca(fmask)
            off = p_off(fmask)

        # Step 3: corrected "uncalibrated spectrum" of cable
        temp_cal_iter = {
            k: (v - temp_amb_internal) * sca + temp_amb_internal - off
            for k, v in temp_raw.items()
        }

        new_sca_off_chisq = (
            (temp_cal_iter["hot_load"] - temp_ant_hot) ** 2
            + (temp_cal_iter["ambient"] - temp_ant["ambient"]) ** 2
        ).sum()

        # Return early if the chi^2 is not improving.
        if new_sca_off_chisq >= sca_off_chisq and return_early:
            return p_sca, p_off, best

        sca_off_chisq = new_sca_off_chisq

        # Step 4: computing NWP
        best = None
        for delay in delays_to_fit:
            srcs = ["open", "short"]
            nwm = NoiseWaveLinearModel(
                freq=fmask,
                gamma_rec=gamma_rec,
                gamma_src={k: gamma_ant[k] for k in srcs},
                model_type=mdl.Polynomial,
                n_terms=wterms,
                delay=delay,
            )
            nwmfit = nwm.fit(
                {k: temp_cal_iter[k] for k in srcs},
                {k: temp_ant[k] for k in srcs},
                method=fit_method,
            )
            if best is None or nwmfit.rms < best.rms:
                best = nwmfit

        tunc = best.get_tunc(fmask)
        tcos = best.get_tcos(fmask)
        tsin = best.get_tsin(fmask)

        # Return early if the chi^2 is not improving.
        if best.rms >= cable_chisq and return_early:
            return p_sca, p_off, best

        cable_chisq = best.rms

        yield (p_sca, p_off, best)


def get_linear_coefficients(
    gamma_ant, gamma_rec, sca, off, t_unc, t_cos, t_sin, t_load=300
):
    """
    Use Monsalve (2017) Eq. 7 to determine a and b, such that T = aT* + b.

    Parameters
    ----------
    gamma_ant : array_like
        S11 of the antenna/load.
    gamma_rec : array_like
        S11 of the receiver.
    sca,off : array_like
        Scale and offset calibration parameters (i.e. C1 and C2). These are in the form
        of arrays over frequency (i.e. it is not the polynomial coefficients).
    t_unc, t_cos, t_sin : array_like
        Noise-wave calibration parameters (uncorrelated, cosine, sine). These are in the
        form of arrays over frequency (i.e. not the polynomial coefficients).
    t_load : float, optional
        The nominal temperature of the internal ambient load. This *must match* the
        value used to derive the calibration parameters in the first place.
    """
    K = get_K(gamma_rec, gamma_ant)

    return get_linear_coefficients_from_K(K, sca, off, t_unc, t_cos, t_sin, t_load)


def get_linear_coefficients_from_K(  # noqa: N802
    k, sca, off, t_unc, t_cos, t_sin, t_load=300
):
    """Calculate linear coefficients a and b from noise-wave parameters K0-4.

    Parameters
    ----------
    k : np.ndarray
        Shape (4, nfreq) array with each of the K-coefficients.
    sca,off : array_like
        Scale and offset calibration parameters (i.e. C1 and C2). These are in the form
        of arrays over frequency (i.e. it is not the polynomial coefficients).
    t_unc, t_cos, t_sin : array_like
        Noise-wave calibration parameters (uncorrelated, cosine, sine). These are in the
        form of arrays over frequency (i.e. not the polynomial coefficients).
    t_load : float, optional
        The nominal temperature of the internal ambient load. This *must match* the
        value used to derive the calibration parameters in the first place.
    """
    # Noise wave contribution
    noise_wave_terms = t_unc * k[1] + t_cos * k[2] + t_sin * k[3]
    return sca / k[0], (t_load - off - noise_wave_terms - t_load * sca) / k[0]


def calibrated_antenna_temperature(
    temp_raw, gamma_ant, gamma_rec, sca, off, t_unc, t_cos, t_sin, t_load=300
):
    """
    Use M17 Eq. 7 to determine calibrated temperature from an uncalibrated temperature.

    Parameters
    ----------
    temp_raw : array_like
        The raw (uncalibrated) temperature spectrum, T*.
    gamma_ant : array_like
        S11 of the antenna/load.
    gamma_rec : array_like
        S11 of the receiver.
    sca,off : array_like
        Scale and offset calibration parameters (i.e. C1 and C2). These are in the form
        of arrays over frequency (i.e. it is not the polynomial coefficients).
    t_unc, t_cos, t_sin : array_like
        Noise-wave calibration parameters (uncorrelated, cosine, sine). These are in the
        form of arrays over frequency (i.e. not the polynomial coefficients).
    t_load : float, optional
        The nominal temperature of the internal ambient load. This *must match* the
        value used to derive the calibration parameters in the first place.
    """
    a, b = get_linear_coefficients(
        gamma_ant, gamma_rec, sca, off, t_unc, t_cos, t_sin, t_load
    )

    return temp_raw * a + b


def decalibrate_antenna_temperature(
    temp, gamma_ant, gamma_rec, sca, off, t_unc, t_cos, t_sin, t_load=300
):
    """
    Use M17 Eq. 7 to determine uncalibrated temperature from a calibrated temperature.

    Parameters
    ----------
    temp : array_like
        The true (or calibrated) temperature spectrum.
    gamma_ant : array_like
        S11 of the antenna/load.
    gamma_rec : array_like
        S11 of the receiver.
    sca,off : array_like
        Scale and offset calibration parameters (i.e. C1 and C2). These are in the form
        of arrays over frequency (i.e. it is not the polynomial coefficients).
    t_unc, t_cos, t_sin : array_like
        Noise-wave calibration parameters (uncorrelated, cosine, sine). These are in the
        form of arrays over frequency (i.e. not the polynomial coefficients).
    t_load : float, optional
        The nominal temperature of the internal ambient load. This *must match* the
        value used to derive the calibration parameters in the first place.
    """
    a, b = get_linear_coefficients(
        gamma_ant, gamma_rec, sca, off, t_unc, t_cos, t_sin, t_load
    )
    return (temp - b) / a


@attrs.define(frozen=True, kw_only=True, slots=False)
class NoiseWaves:
    """A class to manage linear models for fitting noise-wave parameters.

    This is different from :class:`NoiseWaveLinearModel` in several ways (ultimately
    they may be merged). The main way they are different is that this class allows for
    fitting not only the three noise-wave models (Tunc, Tcos, Tsin) but also the
    offset Tload -- which is related to the known temperatures via a linear model as
    well. Also, this class is not restricted to accepting only the long-cable data
    (short and open), but can also accept the ambient and hot load calibration sources
    as input data, to which the noise-wave models are fitted simultaneously.
    """

    freq: np.ndarray = attrs.field(eq=attrs.cmp_using(eq=np.array_equal))
    gamma_src: dict[str, np.ndarray] = attrs.field()
    gamma_rec: np.ndarray = attrs.field(eq=attrs.cmp_using(eq=np.array_equal))
    c_terms: int = attrs.field(default=5)
    w_terms: int = attrs.field(default=6)
    parameters: Sequence | None = attrs.field(
        default=None, eq=attrs.cmp_using(eq=np.array_equal)
    )
    with_tload: bool = attrs.field(default=True)

    @cached_property
    def src_names(self) -> tuple[str]:
        """List of names of inputs sources (eg. ambient, hot_load, open, short)."""
        return tuple(self.gamma_src.keys())

    def get_linear_model(self, with_k: bool = True) -> mdl.CompositeModel:
        """Define and return a Model.

        Parameters
        ----------
        with_k
            Whether to use the K matrix as an "extra basis" in the linear model.
        """
        gamma_rec_func = ComplexSpline(x=self.freq, y=self.gamma_rec)
        gamma_src_func = {
            k: ComplexSpline(x=self.freq, y=v) for k, v in self.gamma_src.items()
        }
        nwlm = NoiseWaveLinearModel(
            freq=self.freq,
            gamma_rec=gamma_rec_func,
            gamma_src=gamma_src_func,
            model_type=mdl.Polynomial,
            n_terms=self.w_terms,
            delay=0.0,  # TODO: make this a parameter?
        )

        # x is the frequencies repeated for every input source
        x = np.tile(self.freq, len(self.gamma_src))
        tr = mdl.UnitTransform(range=(x.min(), x.max()))

        models = {
            "tunc": mdl.Polynomial(
                n_terms=self.w_terms,
                parameters=self.parameters[: self.w_terms]
                if self.parameters is not None
                else None,
                transform=tr,
                basis_scaler=nwlm.unc_kfactor if with_k else None,
            ),
            "tcos": mdl.Polynomial(
                n_terms=self.w_terms,
                parameters=self.parameters[self.w_terms : 2 * self.w_terms]
                if self.parameters is not None
                else None,
                transform=tr,
                basis_scaler=nwlm.cos_kfactor if with_k else None,
            ),
            "tsin": mdl.Polynomial(
                n_terms=self.w_terms,
                parameters=self.parameters[2 * self.w_terms : 3 * self.w_terms]
                if self.parameters is not None
                else None,
                transform=tr,
                basis_scaler=nwlm.sin_kfactor if with_k else None,
            ),
        }

        if self.with_tload:
            models["tload"] = mdl.Polynomial(
                n_terms=self.c_terms,
                parameters=self.parameters[3 * self.w_terms :]
                if self.parameters is not None
                else None,
                transform=tr,
                basis_scaler=(lambda x: -np.ones(len(x))) if with_k else None,
            )

        return mdl.CompositeModel(models=models).at(x=x)

    @cached_property
    def linear_model(self) -> mdl.CompositeModel:
        """The actual composite linear model object associated with the noise waves."""
        return self.get_linear_model()

    def get_noise_wave(
        self,
        noise_wave: str,
        parameters: Sequence | None = None,
        src: str | None = None,
    ) -> np.ndarray:
        """Get the model for a particular noise-wave term."""
        out = self.linear_model.model.get_model(
            noise_wave,
            parameters=parameters,
            x=self.linear_model.x,
            with_scaler=bool(src),
        )
        if src:
            indx = self.src_names.index(src)
            return out[indx * len(self.freq) : (indx + 1) * len(self.freq)]
        return out[: len(self.freq)]

    def get_full_model(
        self, src: str, parameters: Sequence | None = None
    ) -> np.ndarray:
        """Get the full model (all noise-waves) for a particular input source."""
        out = self.linear_model(parameters=parameters)
        indx = self.src_names.index(src)
        return out[indx * len(self.freq) : (indx + 1) * len(self.freq)]

    def get_fitted(
        self, data: np.ndarray, weights: np.ndarray | None = None, **kwargs
    ) -> NoiseWaves:
        """Get a new noise wave model with fitted parameters."""
        fit = self.linear_model.fit(ydata=data, weights=weights, **kwargs)
        return attrs.evolve(self, parameters=fit.model_parameters)

    def with_params_from_calobs(self, calobs, cterms=None, wterms=None) -> NoiseWaves:
        """Get a new noise wave model with parameters fitted using standard methods."""
        cterms = cterms or calobs.cterms
        wterms = wterms or calobs.wterms

        def modify(thing, n):
            if isinstance(thing, np.ndarray):
                thing = thing.tolist()
            elif isinstance(thing, tuple):
                thing = list(thing)

            if len(thing) < n:
                return thing + [0] * (n - len(thing))
            if len(thing) > n:
                return thing[:n]
            return thing

        tu = modify(calobs.cal_coefficient_models["NW"].model.parameters, wterms)
        tc = modify(calobs.cal_coefficient_models["NW"].model.parameters, wterms)
        ts = modify(calobs.cal_coefficient_models["NW"].model.parameters, wterms)

        if self.with_tload:
            c2 = -np.asarray(calobs.cal_coefficient_models["C2"].parameters)
            c2[0] += calobs.t_load
            c2 = modify(c2, cterms)

        return attrs.evolve(self, parameters=tu + tc + ts + c2)

    @classmethod
    def from_calobs(
        cls,
        calobs,
        cterms=None,
        wterms=None,
        sources=None,
        with_tload: bool = True,
        loads: dict | None = None,
    ) -> NoiseWaves:
        """Initialize a noise wave model from a calibration observation."""
        if loads is None:
            if sources is None:
                sources = calobs.load_names

            loads = {src: load for src, load in calobs.loads.items() if src in sources}

        freq = calobs.freq.to_value("MHz")

        gamma_src = {name: source.s11_model(freq) for name, source in loads.items()}

        try:
            lna_s11 = calobs.receiver.s11_model(freq)
        except AttributeError:
            lna_s11 = calobs.receiver_s11(freq)

        nw_model = cls(
            freq=freq,
            gamma_src=gamma_src,
            gamma_rec=lna_s11,
            c_terms=cterms or calobs.cterms,
            w_terms=wterms or calobs.wterms,
            with_tload=with_tload,
        )
        return nw_model.with_params_from_calobs(calobs, cterms=cterms, wterms=wterms)

    def __call__(self, **kwargs) -> np.ndarray:
        """Call the underlying linear model."""
        return self.linear_model(**kwargs)
