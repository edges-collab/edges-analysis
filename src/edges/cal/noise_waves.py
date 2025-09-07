"""Functions for calibrating the receiver."""

from collections.abc import Generator, Sequence
from functools import cached_property
from typing import Self

import attrs
import numpy as np
from astropy import units as un

from .. import modeling as mdl
from .. import types as tp


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
    """A class representing a fit of the :class:NoiseWaveLinearModel` to data.

    Parameters
    ----------
    freq
        The frequencies at which the fit was performed.
    gamma_rec
        The S11 of the receiver used in the fit.
    gamma_src
        The S11s of the calibration sources used in the fit.
    modelfit
        The composite model defining the model that was fit to the data.
    delay
        The delay corrected for in the fit.
    """

    freq: np.ndarray = attrs.field()
    gamma_rec: np.ndarray = attrs.field()
    gamma_src: dict[str, np.ndarray] = attrs.field()
    modelfit: mdl.CompositeModel = attrs.field()
    delay: float = attrs.field(default=0.0)

    @property
    def model(self):
        """The model, with assigned parameters from best-fit."""
        return self.modelfit.fit

    def get_tunc(self, freq: tp.FreqType | None = None, full_term=False):
        r"""Compute the model for Tunc(freq).

        If `full_term` is True, return the full term, i.e.

        .. math:: Tunc \frac{|Gamma_{\rm ant}|^2 |F|^2}{1 - |Gamma_{\rm rcv}|^2}

        otherwise just return Tunc itself. This is not implemented yet (it's a little
        trickier because it depends on which source you want it for).
        """
        if freq is None:
            freq = self.freq
        return self.model.model.models["unc"](x=freq, with_scaler=False)

    def get_tcos(self, freq: tp.FreqType | None = None, full_term=False):
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

    def get_tsin(self, freq: tp.FreqType | None = None, full_term=False):
        r"""Compute the model for Tsin(freq).

        If `full_term` is True, return the full term, i.e.

        .. math:: Tsin \frac{|Gamma_{\rm ant}| |F| \sin\alpha}{1 - |Gamma_{\rm rcv}|^2}

        otherwise just return Tsin itself. This is not implemented yet (it's a little
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
    gamma_rec: np.ndarray = attrs.field()
    gamma_src: dict[str, np.ndarray] = attrs.field()
    model_type: mdl.Model = attrs.field(default=mdl.Polynomial)
    n_terms: int = attrs.field(default=5)
    delay: float = attrs.field(default=0.0)

    def cos_kfactor(self, freq):
        """Compute the scaler to the Tcos basis function."""
        freq = freq[: len(freq) // len(self.gamma_src)]

        ph = np.exp(1j * 2 * np.pi * freq * self.delay * 1e6)

        out = []
        for gamma in self.gamma_src.values():
            K = get_K(self.gamma_rec, gamma)
            out.append(K[2] * ph.real - K[3] * ph.imag)

        return np.concatenate(out)

    def sin_kfactor(self, freq):
        """Compute the scaler to the Tsin basis function."""
        freq = freq[: len(freq) // len(self.gamma_src)]
        ph = np.exp(1j * 2 * np.pi * freq * self.delay * 1e6)

        out = []
        for gamma in self.gamma_src.values():
            K = get_K(self.gamma_rec, gamma)

            out.append(K[2] * ph.imag + K[3] * ph.real)
        return np.concatenate(out)

    def unc_kfactor(self, freq):
        """Compute the scaler to the Tunc basis function."""
        return np.concatenate([
            get_K(self.gamma_rec, gamma)[1] for gamma in self.gamma_src.values()
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

        K0 = {k: get_K(self.gamma_rec, self.gamma_src[k])[0] for k in self.gamma_src}

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
    freqs: tp.FreqType,
    source_q: dict,
    source_s11s: dict[str, np.ndarray],
    source_true_temps: dict[str, tp.TemperatureType],
    receiver_s11: np.ndarray,
    cterms: int,
    wterms: int,
    t_load_guess: tp.TemperatureType = 300 * un.K,
    t_load_ns_guess: tp.TemperatureType = 1000.0 * un.K,
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
    freqs
        The frequencies at which the data is specified.
    source_q
        The PSD ratios, Q, of the calibration sources, as a dictionary with the source
        names as keys.
    source_s11s
        A dictionary with the same keys as `source_q`, containing callable S11 models
        for each calibration source.
    source_true_temps
        A dictionary with the same keys as `source_q`, containing the true temperature
        of each calibration source, to which to calibrate the `source_q`. These must
        be astropy temperature Quantities, but may be scalar or arrays. If an array,
        must be the same length as freqs.
    receiver_s11
        Receiver S11 as a function of frequency (callable).
    cterms : int
        Number of polynomial terms for the C_i
    wterms : int
        Number of polynonmial temrs for the T_i
    t_load_guess
        An initial guess for the internal dicke-switch load temperature, used to
        initialize the iterative algorithm.
    t_load_ns_guess
        An initial guess for the internal dicke-switch load+noise-sourc temperature,
        used to initialize the iterative algorithm.
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
    return_early
        Whether to return from the iterative loop early if the RMS of the residuals
        stops decreasing.

    Yields
    ------
    t_sca_model
        A fitted polynomial model for the temperature scale (approximately, the
        true internal load+noise-source temperature, but with corrections from
        internal path-lengths).
    t_off_model
        A fitted polynomial model for the temperature offset (approximately, the
        true internal load temperature, but with corrections from internal path-lengths)
    nwp
        A fitted model for the noise-wave parameters. The values of each of the
        noise-wave temperatures can be accessed via e.g. `nwp.get_tunc()`.

    Notes
    -----
    To achieve the same results as the legacy C pipeline, the `hot_load_loss` parameter
    should be given, and not applied to the "true" temperature. There is a small
    mathematical difference that arises if you do it the other way. Furthermore, the
    `smooth_scale_offset_within_loop` parameter should be set to False.
    """
    # Convert all temperatures to K and remove the units to make the iterative
    # algorithm fast and not have to worry about units.
    t_load_guess = t_load_guess.to_value("K")
    t_load_ns_guess = t_load_ns_guess.to_value("K")
    source_true_temps = {k: v.to_value("K") for k, v in source_true_temps.items()}
    freqs = freqs.to_value("MHz")

    # Get approximate temperatures

    mask = np.all([np.isfinite(temp) for temp in source_q.values()], axis=0)

    fmask = freqs[mask]
    source_q = {key: value[mask] for key, value in source_q.items()}
    source_true_temps = {
        key: (value[mask] if hasattr(value, "__len__") else value)
        for key, value in source_true_temps.items()
    }
    temp_ant_hot = source_true_temps["hot_load"]
    receiver_s11 = receiver_s11[mask]
    source_s11s = {k: v[mask] for k, v in source_s11s.items()}

    # The denominator of each term in Eq. 7
    G = 1 - np.abs(receiver_s11) ** 2

    K1, K2, K3, K4 = {}, {}, {}, {}
    for k, gamma_a in source_s11s.items():
        K1[k], K2[k], K3[k], K4[k] = get_K(receiver_s11, gamma_a, gain=G)

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

    tr = mdl.ScaleTransform(scale=freqs[len(freqs) // 2])
    sca_mdl = mdl.Polynomial(n_terms=cterms, transform=tr, spacing=poly_spacing).at(
        x=fmask
    )
    off_mdl = mdl.Polynomial(n_terms=cterms, transform=tr, spacing=poly_spacing).at(
        x=fmask
    )

    # Get initial guess of the calibrated temperatures.
    temp_cal_iter = {k: v * t_load_ns_guess + t_load_guess for k, v in source_q.items()}

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
                thot_iter - source_true_temps["ambient"] * (1 - hot_load_loss)
            ) / hot_load_loss

        # Updating scale and offset
        sca_new = (temp_ant_hot - source_true_temps["ambient"]) / (
            thot_iter - tamb_iter
        )
        off_new = tamb_iter - source_true_temps["ambient"]

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
            k: v * t_load_ns_guess * sca + t_load_guess - off
            for k, v in source_q.items()
        }

        new_sca_off_chisq = (
            (temp_cal_iter["hot_load"] - temp_ant_hot) ** 2
            + (temp_cal_iter["ambient"] - source_true_temps["ambient"]) ** 2
        ).sum()

        # Return early if the chi^2 is not improving.
        if new_sca_off_chisq >= sca_off_chisq and return_early:
            return

        sca_off_chisq = new_sca_off_chisq

        # Step 4: computing NWP
        best = None
        for delay in delays_to_fit:
            srcs = ["open", "short"]
            nwm = NoiseWaveLinearModel(
                freq=fmask,
                gamma_rec=receiver_s11,
                gamma_src={k: source_s11s[k] for k in srcs},
                model_type=mdl.Polynomial,
                n_terms=wterms,
                delay=delay,
            )
            nwmfit = nwm.fit(
                {k: temp_cal_iter[k] for k in srcs},
                {k: source_true_temps[k] for k in srcs},
                method=fit_method,
            )
            if best is None or nwmfit.rms < best.rms:
                best = nwmfit

        tunc = best.get_tunc(fmask)
        tcos = best.get_tcos(fmask)
        tsin = best.get_tsin(fmask)

        # Return early if the chi^2 is not improving.
        if best.rms >= cable_chisq and return_early:
            return

        cable_chisq = best.rms

        t_sca_model = p_sca.with_params(np.array(p_sca.parameters) * t_load_ns_guess)
        t_off_model = p_off.with_params(
            np.array([t_load_guess] + [0] * (p_off.n_terms - 1))
            - np.array(p_off.parameters)
        )

        yield (t_sca_model, t_off_model, best)


def get_linear_coefficients(gamma_ant, gamma_rec, t_sca, t_off, t_unc, t_cos, t_sin):
    """
    Use Monsalve (2017) Eq. 7 to determine a and b, such that T = aT* + b.

    Parameters
    ----------
    gamma_ant : array_like
        S11 of the antenna/load.
    gamma_rec : array_like
        S11 of the receiver.
    sca,off : array_like
        Scale and offset calibration parameters (i.e. Tsca and Toff). These are in the
        form of arrays over frequency (i.e. it is not the polynomial coefficients).
    t_unc, t_cos, t_sin : array_like
        Noise-wave calibration parameters (uncorrelated, cosine, sine). These are in the
        form of arrays over frequency (i.e. not the polynomial coefficients).
    """
    K = get_K(gamma_rec, gamma_ant)
    return get_linear_coefficients_from_K(K, t_sca, t_off, t_unc, t_cos, t_sin)


def get_linear_coefficients_from_K(  # noqa: N802
    k, t_sca, t_off, t_unc, t_cos, t_sin
):
    """Calculate linear coefficients a and b from noise-wave parameters K0-4.

    Parameters
    ----------
    k : np.ndarray
        Shape (4, nfreq) array with each of the K-coefficients.
    sca,off : array_like
        Scale and offset calibration temperatures. These are in the form
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
    return t_sca / k[0], (t_off - noise_wave_terms) / k[0]


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
        nwlm = NoiseWaveLinearModel(
            freq=self.freq,
            gamma_rec=self.gamma_rec,
            gamma_src=self.gamma_src,
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
    ) -> Self:
        """Get a new noise wave model with fitted parameters."""
        fit = self.linear_model.fit(ydata=data, weights=weights, **kwargs)
        return attrs.evolve(self, parameters=fit.model_parameters)

    @classmethod
    def from_calobs(
        cls,
        calobs,
        cterms: int,
        wterms: int,
        sources=None,
        with_tload: bool = True,
        loads: dict | None = None,
    ) -> Self:
        """Initialize a noise wave model from a calibration observation."""
        if loads is None:
            if sources is None:
                sources = calobs.load_names

            loads = {src: load for src, load in calobs.loads.items() if src in sources}

        freq = calobs.freqs.to_value("MHz")

        gamma_src = {name: source.s11.s11 for name, source in loads.items()}

        lna_s11 = calobs.receiver.s11

        return cls(
            freq=freq,
            gamma_src=gamma_src,
            gamma_rec=lna_s11,
            c_terms=cterms,
            w_terms=wterms,
            with_tload=with_tload,
        )

    def __call__(self, **kwargs) -> np.ndarray:
        """Call the underlying linear model."""
        return self.linear_model(**kwargs)
