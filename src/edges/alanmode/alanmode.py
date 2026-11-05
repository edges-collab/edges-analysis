"""Functions that run the calibration in a style similar to the C-code."""

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated, Self

import attrs
import numpy as np
from astropy import units as un
from astropy.time import Time
from cyclopts import Parameter
from pygsdata import GSData
from pygsdata.select import select_freqs, select_times
from read_acq.gsdata import read_acq_to_gsdata

from edges.cal import sparams as sp
from edges.frequencies import get_mask
from edges.io.calobsdef3 import CalObsDefEDGES3

from .. import modeling as mdl
from .. import types as tp
from ..averaging.freqbin import gauss_smooth
from ..averaging.lstbin import average_over_times
from ..cal.apply import approximate_temperature
from ..cal.calobs import CalibrationObservation, Calibrator
from ..cal.dicke import dicke_calibration
from ..cal.loss import (
    LossFunctionGivenSparams,
    get_cable_loss_model,
    get_loss_model_from_file,
)
from ..cal.receiver_cal import get_noise_wave_calibration_iterative
from ..cal.sparams import (
    CalkitReadings,
    ReflectionCoefficient,
    S11ModelParams,
    SParams,
    correct_receiver_for_extra_cable,
)
from ..cal.spectra import LoadSpectrum
from ..config import config
from ..filters.filters import (
    maxfm_filter,
    peak_orbcomm_filter,
    power_percent_filter,
    rms_filter,
)
from . import alanio

logger = logging.getLogger(__name__)


def reads1p1(
    res: float,
    s11_file_open: str | Path,
    s11_file_short: str | Path,
    s11_file_load: str | Path,
    s11_file_ant: str | Path,
    loadps: float = 33.0,
    openps: float = 33.0,
    shortps: float = 33.0,
) -> sp.ReflectionCoefficient:
    """Read and calibrate an S11 measured with the VNA, as the C function reads1p1.

    The S11 is calibrated with the internal open/short/load measurements, using
    :func:`edges.cal.sparams.calibrate_gamma_src` with the ``AGILENT_ALAN`` calkit,
    updated with the given match resistance and delays of the standards.

    Parameters
    ----------
    res
        The resistance of the match (load) standard, in Ohms.
    s11_file_open, s11_file_short, s11_file_load
        The ``.s1p`` files of the measurements of the open, short and match (load)
        standards.
    s11_file_ant
        The ``.s1p`` file of the measurement of the device (load, antenna or LNA).
    loadps, openps, shortps
        The offset delays of the match (load), open and short standards, in ps.

    Returns
    -------
    ReflectionCoefficient
        The calibrated reflection coefficient of the device.
    """
    calkit = sp.get_calkit(
        sp.AGILENT_ALAN,
        resistance_of_match=res * un.ohm,
        short={"offset_delay": shortps * un.ps},
        open={"offset_delay": openps * un.ps},
        match={"offset_delay": loadps * un.ps},
    )
    return sp.calibrate_gamma_src(
        gamma_src=sp.ReflectionCoefficient.from_s1p(s11_file_ant),
        internal_osl=CalkitReadings(
            open=sp.ReflectionCoefficient.from_s1p(s11_file_open),
            short=sp.ReflectionCoefficient.from_s1p(s11_file_short),
            match=sp.ReflectionCoefficient.from_s1p(s11_file_load),
        ),
        internal_calkit=calkit,
    )


def corrcsv(
    s11: sp.ReflectionCoefficient,
    cablen: float,
    cabdiel: float,
    cabloss: float,
) -> sp.ReflectionCoefficient:
    """Correct the receiver (LNA) S11 for the cable to the VNA, as the C corrcsv.

    This calls :func:`edges.cal.sparams.correct_receiver_for_extra_cable`. Unlike
    :func:`edges.cal.sparams.calibrate_gamma_receiver`, the correction is applied even
    if the cable length is zero, as in the C code.

    Parameters
    ----------
    s11
        The S11 of the receiver.
    cablen
        The cable length, in inches.
    cabdiel
        The cable dielectric constant, as a percent.
    cabloss
        The cable loss, as a percent.

    Returns
    -------
    ReflectionCoefficient
        The corrected S11 of the receiver.
    """
    return correct_receiver_for_extra_cable(
        gamma=s11,
        cable_length=(cablen * un.imperial.inch).to("m"),
        cable_loss_percent=cabloss,
        cable_dielectric_percent=cabdiel,
    )


@attrs.define(kw_only=True, frozen=True)
class ACQPlot7aMoonParams:
    """Parameters for the ACQPlot7aMoon spectrum reading and averaging script.

    Parameters
    ----------
    fstart
        The starting frequency for the spectrum.
    fstop
        The stopping frequency for the spectrum.
    smooth
        The number of frequency bins to smooth and downsamples by.
    tload
        A guess at the internal load temperature, acting as a starting point
        for iterative calibration fitting, but also is applied to "kind of" calibrate
        the data before finding RFI.
    tcal
        Similar to ``tload``, but for the internal load + noise source.
    tstart
        The starting hour (in gha) for reading spectrum integrations.
    tstop
        The ending hour (in gha) for reading spectrum integrations.
    delaystart
        The number of seconds to delay before including spectra from the files.
        Can be useful to ignore beginning of files as the system may still be
        warming up.
    peakpwr
        Spectra are only averaged if the percentage of the total antenna-state power
        that lies above 100 MHz is below this (see
        :func:`edges.filters.power_percent_filter`). Off by default.
    minpwr
        Spectra are only averaged if the percentage of the total antenna-state power
        that lies above 100 MHz is above this (see
        :func:`edges.filters.power_percent_filter`). Off by default.
    pkpwrm
        Spectra are only averaged if the peak power (in dB) in the ORBCOMM band
        (137.5 +/- 1 MHz), relative to the mean above 80 MHz, is below this (see
        :func:`edges.filters.peak_orbcomm_filter`). If negative, spectra are instead
        only averaged if it is above the magnitude of this. Off by default.
    maxrmsf
        Spectra are only averaged if the RMS of the residuals (in K) of a
        :math:`f^{-2.5}` fit to the spectrum between 60 and 80 MHz is below this (see
        :func:`edges.filters.rms_filter`). As in the C-code, the band is defined by
        channel index on a grid of 32768 channels over 0-200 MHz. Off by default.
    maxfm
        Spectra are only averaged if the largest single-channel spike (in K) in the
        FM band (88-120 MHz) is below this (see :func:`edges.filters.maxfm_filter`).
        Off by default.

    Notes
    -----
    The quality cuts (``peakpwr``, ``minpwr``, ``pkpwrm``, ``maxrmsf`` and
    ``maxfm``) are those of the C-code, with the same names as its command-line
    options, and are applied with the corresponding filters of :mod:`edges.filters`.
    They are computed on each (Dicke-switched) spectrum at full resolution over the
    whole band (not just ``fstart``-``fstop``). All but ``peakpwr`` and ``minpwr``
    are computed on the uncalibrated temperature ``Q * tcal + tload``, which (as in
    the C-code) is set to zero where the noise source adds no power.
    """

    fstart: float = 50.0
    fstop: float = 100.0
    smooth: int = 8
    tload: float = 300.0
    tcal: float = 1000.0
    tstart: int = 0
    tstop: int = 23
    delaystart: int = 0
    peakpwr: float = np.inf
    minpwr: float = 0.0
    pkpwrm: float = np.inf
    maxrmsf: float = np.inf
    maxfm: float = np.inf

    @classmethod
    def bowman_2018_defaults(
        cls,
        delaystart=7200,
        smooth=8,
        fstart=40.0,
        fstop=110.0,
        tload=300.0,
        tcal=1000.0,
        tstart=0,
        tstop=23,
    ):
        """Construct a set of parameters with defaults matching Bowman+2018."""
        return cls(
            delaystart=delaystart,
            smooth=smooth,
            fstart=fstart,
            fstop=fstop,
            tload=tload,
            tcal=tcal,
            tstart=tstart,
            tstop=tstop,
        )


def acqplot7amoon(
    acqfile: tp.PathLike | Sequence[tp.PathLike],
    params: ACQPlot7aMoonParams = ACQPlot7aMoonParams(),
    **kwargs,
) -> GSData:
    """A function that does what the acqplot7amoon C-code does.

    Parameters
    ----------
    acqfile
        The path to the ACQ file to process.
    params
        The parameters for the ACQPlot7aMoon function.

    Other Parameters
    ----------------
    kwargs
        Any field of :class:`ACQPlot7aMoonParams`, overriding its value in
        ``params``.
    """
    if kwargs:
        params = attrs.evolve(params, **kwargs)

    return _acqplot7amoon(acqfile, params)[0]


# The band of the -maxrmsf cut of the C-code, which takes channels 9830-13106
# (i.e. int(60 * 32768 / 200) up to, but not including, int(80 * 32768 / 200)) of a
# 32768-channel grid over 0-200 MHz. Padded by half a channel, to select by frequency.
_RMSF_CHANNEL_WIDTH = 200.0 / 32768
_RMSF_FREQ_RANGE = (
    (int(60.0 / _RMSF_CHANNEL_WIDTH) - 0.5) * _RMSF_CHANNEL_WIDTH * un.MHz,
    (int(80.0 / _RMSF_CHANNEL_WIDTH) - 0.5) * _RMSF_CHANNEL_WIDTH * un.MHz,
)


def _passes_quality_cuts(data: GSData, params: ACQPlot7aMoonParams) -> np.ndarray:
    """Return whether each integration passes the quality cuts of the C-code.

    Parameters
    ----------
    data
        The raw (three-position switch) power spectra, at full resolution over the
        whole band.
    params
        The parameters defining the cuts.

    Returns
    -------
    passed
        Boolean array, True for each integration that passes all the cuts.
    """
    data = power_percent_filter(
        data, min_threshold=params.minpwr, max_threshold=params.peakpwr
    )
    temp = approximate_temperature(
        data=dicke_calibration(data), tload=params.tload, tns=params.tcal
    )
    # As in the C-code, the temperature is zero where the noise source adds no power.
    iload = data.loads.index("internal_load")
    ilns = data.loads.index("internal_load_plus_noise_source")
    defined = (data.data[ilns] - data.data[iload]) > 1e-99
    temp = temp.update(data=np.where(defined[None], temp.data, 0.0))

    if np.isfinite(params.pkpwrm):
        temp = peak_orbcomm_filter(
            temp, threshold=params.pkpwrm, mean_freq_range=(80.0, 200.0)
        )
    if np.isfinite(params.maxfm):
        temp = maxfm_filter(temp, threshold=params.maxfm)
    if np.isfinite(params.maxrmsf):
        temp = rms_filter(
            temp,
            threshold=params.maxrmsf,
            freq_range=_RMSF_FREQ_RANGE,
            model=mdl.LinLog(n_terms=1, beta=-2.5),
        )
    return ~np.all(temp.complete_flags, axis=(0, 1, 3))


def _acqplot7amoon(
    acqfile: tp.PathLike | Sequence[tp.PathLike], params: ACQPlot7aMoonParams
) -> tuple[GSData, int]:
    """Do what :func:`acqplot7amoon` does, also returning the number of spectra.

    Returns
    -------
    spectrum
        The averaged spectrum.
    n
        The number of (Dicke-switched) spectra that were averaged (i.e. that pass the
        time selection and the quality cuts), as written in the header of the
        averaged-spectrum files of the C-code.
    """
    data = read_acq_to_gsdata(acqfile, telescope="edges-low")

    if params.tstart > 0 or params.tstop < 23:
        # Note that tstop=23 includes all possible hours since we have <=
        hours = np.array([x.hour for x in data.times[:, 0].datetime])
        data = select_times(
            data, indx=(hours >= params.tstart) & (hours <= params.tstop)
        )

    if params.delaystart > 0:
        secs = (data.times - data.times.min()).sec
        idx = np.all(secs > params.delaystart, axis=1)
        data = select_times(data, indx=idx)

    # The quality cuts are computed on the whole band, so before selecting freqs.
    passed = _passes_quality_cuts(data, params)
    if not np.any(passed):
        raise ValueError(f"No spectra pass the quality cuts in {acqfile}.")
    if not np.all(passed):
        logger.info(f"{np.sum(~passed)} of {passed.size} spectra fail quality cuts.")
        data = select_times(data, indx=passed)

    data = select_freqs(
        data, freq_range=(params.fstart * un.MHz, params.fstop * un.MHz)
    )
    q = dicke_calibration(data)

    if params.smooth > 0:
        q = gauss_smooth(q, size=params.smooth, decimate_at=0)

    n = q.ntimes
    q = average_over_times(q)
    return approximate_temperature(data=q, tload=params.tload, tns=params.tcal), n


@attrs.define(kw_only=True, frozen=True)
class EdgesScriptParams:
    """Parameters for the edges script.

    These are parameters for the script traditionally called either edges2k.c
    or edges3.c.

    Parameters
    ----------
    Lh
        The mode in which to calculate the hot-load loss. -1 uses the model of the
        UT-141C-SP cable (as for EDGES-3). -2 models the loss from the semi-rigid
        cable S-parameters (``s11rig``, ``s12rig`` and ``s22rig``, as for EDGES-2).
        From Python, a path to a file tabulating the loss can also be given. Any
        other value means no hot-load loss.
    wfstart
        The lowest frequency included for fitting the calibration functions, and the
        antenna S11.
    wfstop
        The highest frequency included for fitting the calibration functions, and the
        antenna S11.
    fstart
        The lowest frequency included when reading the spectra and raw calibration
        s11s (and therefore defining the range over which the calibration S11s are fit)
    fstop
        The highest frequency included when reading the spectra and raw calibration
        s11s (and therefore defining the range over which the calibration S11s are fit)
    tcold
        The "true" temperature of the ambient load.
    thot
        The "true" temperature of the hot load.
    tcab
        The "true" temperature of the long-cable calibration loads. By default,
        the same as tcold.
    cfit
        The number of polynomial terms to use for fitting the scale and offset
        calibration temperatures (i.e. Tload and Tlns)
    wfit
        The number of polynomial terms to use for fitting the noise-wave calibration
        temperatures.
    nfit3
        The number of terms used to model the receiver S11.
    nfit2
        The number of terms used to model the load S11s.
    nter
        The number of iterations to perform when fitting the noise-wave
        calibration temperatures.
    lna_poly
        Whether the receiver S11 should be modeled as a polynomial. 0 for False,
        anything else will determine the model based on `nfit3`. If `nfit3` is greater
        than 16, a Fourier model will be used, otherwise a Polynomial model.
    wtmode
        How to set weights for noise-wave fitting. Currently only wtmode=0 is supported,
        which sets the weights based on wfstart/wfstop. We've added the parameter here
        to remind you NOT to set it to anything else when comparing to the legacy
        C-code (i.e. make sure it's set to zero there too!).
    """

    Lh: Annotated[int, Parameter(name=("Lh",))] = -1
    wfstart: float = 50
    wfstop: float = 200
    fstart: float = 50
    fstop: float = 200
    tcold: float = 306.5
    thot: float = 393.22
    tcab: float = attrs.field()
    cfit: int = 7
    wfit: int = 7
    nfit3: int = 10
    nfit2: int = 27
    nter: int = 8
    lna_poly: int = -1
    wtmode: int = attrs.field(default=0, validator=attrs.validators.in_((0,)))

    @tcab.default
    def _tcab_default(self) -> float:
        return self.tcold

    @classmethod
    def bowman_2018_defaults(
        cls,
        cfit=6,
        wfit=5,
        Lh=-2,  # ruff: ignore[invalid-argument-name]
        wfstart=50.0,
        wfstop=100.0,
        tcold=296,
        thot=399,
        nfit2=27,
        nfit3=11,
        lna_poly=0,
        **kwargs,
    ) -> Self:
        """Construct a set of parameters with defaults matching Bowman+2018."""
        return cls(
            cfit=cfit,
            wfit=wfit,
            Lh=Lh,
            wfstart=wfstart,
            wfstop=wfstop,
            tcold=tcold,
            thot=thot,
            nfit2=nfit2,
            nfit3=nfit3,
            lna_poly=lna_poly,
            **kwargs,
        )


def _get_specs(
    spcold: GSData,
    sphot: GSData,
    spopen: GSData,
    spshort: GSData,
    params: EdgesScriptParams,
    tload: tp.TemperatureType,
    tcal: tp.TemperatureType,
) -> dict[str, LoadSpectrum]:
    specs = {}

    for name, spec, temp in (
        ("ambient", spcold, params.tcold),
        ("hot_load", sphot, params.thot),
        ("open", spopen, params.tcab),
        ("short", spshort, params.tcab),
    ):
        specs[name] = LoadSpectrum(
            q=approximate_temperature(
                spec, tload=tload.to_value("K"), tns=tcal.to_value("K"), reverse=True
            ),
            temp_ave=temp * un.K,
        )
        # We use wfstart/wfstop here instead of fstart/fstop, because even though the
        # C code uses fstart/fstop, it only ever uses the spectra when doing the NW
        # fitting, and there it gives zero weight to frequencies outside of
        # wfstart/wfstop. So effectively, the spectra are only used within that range.
        # Note that this doesn't apply to the S11s, which are _modelled_ on the full
        # range fstart/fstop, so need to be read in on that full range.
        specs[name] = specs[name].between_freqs(
            params.wfstart * un.MHz, params.wfstop * un.MHz
        )
    return specs


def _s11_model_params(
    n_terms: int,
    *,
    fourier: bool,
    fourier_transform: mdl.XTransform | None = None,
    polynomial_transform: mdl.XTransform | None = None,
    set_transform_range: bool = True,
    fit_method: str = "alan-qrd",
    find_model_delay: bool = True,
    combine_s12s21: bool = True,
) -> S11ModelParams:
    """Return the parameters of an S11 model as used by the C code.

    The C code models S11s with either a Fourier series (with a period of 1.5) or a
    polynomial, fitted to the real and imaginary parts separately. This is the single
    place where alanmode builds these parameters; the call sites differ only in the
    keyword arguments.

    Parameters
    ----------
    n_terms
        The number of terms of the model.
    fourier
        Whether to use a Fourier series (otherwise a polynomial).
    fourier_transform
        The coordinate transform of the Fourier series. Required if
        ``set_transform_range`` is False. If ``set_transform_range`` is True, its range
        is replaced by the frequency range of the data when fitting, so by default a
        :class:`~edges.modeling.ZerotooneTransform` whose range is a placeholder.
    polynomial_transform
        The coordinate transform of the polynomial. Required if
        ``set_transform_range`` is False. If ``set_transform_range`` is True, its scale
        is replaced by the centre of the frequency range of the data when fitting, so
        by default a :class:`~edges.modeling.Log10Transform` whose scale is a
        placeholder.
    set_transform_range
        Whether to set the range (or scale) of the transform from the data when
        fitting (see :class:`~edges.cal.sparams.S11ModelParams`).
    fit_method
        The method used to fit the model.
    find_model_delay
        Whether to find and remove a delay before fitting the model.
    combine_s12s21
        Whether to model the product S12*S21 (only used for two-port S-parameters).

    Returns
    -------
    S11ModelParams
        The parameters of the S11 model.
    """
    if set_transform_range:
        # Placeholders: the range and the scale are set from the data when fitting.
        if fourier_transform is None:
            fourier_transform = mdl.ZerotooneTransform(range=(1, 2))
        if polynomial_transform is None:
            polynomial_transform = mdl.Log10Transform(scale=1)
    elif fourier_transform is None or polynomial_transform is None:
        raise ValueError(
            "fourier_transform and polynomial_transform are required if "
            "set_transform_range is False"
        )

    if fourier:
        model = mdl.Fourier(n_terms=n_terms, transform=fourier_transform, period=1.5)
    else:
        model = mdl.Polynomial(n_terms=n_terms, transform=polynomial_transform)

    return S11ModelParams(
        model=model,
        complex_model_type=mdl.ComplexRealImagModel,
        set_transform_range=set_transform_range,
        fit_method=fit_method,
        find_model_delay=find_model_delay,
        combine_s12s21=combine_s12s21,
    )


def _get_load_s11s(
    params: EdgesScriptParams,
    s11cold: np.ndarray,
    s11hot: np.ndarray,
    s11open: np.ndarray,
    s11short: np.ndarray,
    s11mask: np.ndarray,
    s11freq: tp.FreqType,
) -> tuple[dict[str, ReflectionCoefficient], S11ModelParams]:
    """Return the raw S11s of the loads (within fstart/fstop) and their model."""
    raw_s11s = {
        name: ReflectionCoefficient(
            reflection_coefficient=s11[s11mask],
            freqs=s11freq,
        )
        for name, s11 in (
            ("ambient", s11cold),
            ("hot_load", s11hot),
            ("open", s11open),
            ("short", s11short),
        )
    }

    s11_modelling_params = _s11_model_params(params.nfit2, fourier=params.nfit2 > 16)
    return raw_s11s, s11_modelling_params


def _get_receiver_s11(
    params: EdgesScriptParams,
    s11lna: np.ndarray,
    s11mask: np.ndarray,
    s11freq: tp.FreqType,
) -> tuple[ReflectionCoefficient, S11ModelParams]:
    """Return the raw S11 of the receiver (within fstart/fstop) and its model."""
    raw_receiver = ReflectionCoefficient(
        reflection_coefficient=s11lna[s11mask],
        freqs=s11freq,
    )
    receiver_model = _s11_model_params(
        params.nfit3, fourier=params.nfit3 > 16 or params.lna_poly == 0
    )
    return raw_receiver, receiver_model


def _get_hotload_loss(
    params: EdgesScriptParams,
    s11rig,
    s12rig,
    s22rig,
    s11mask,
    s11freq,
    spec_fq,
) -> Callable | None:
    if params.Lh == -1:
        hot_loss_model = get_cable_loss_model("UT-141C-SP")
    elif params.Lh == -2:
        if s11rig is None or s12rig is None or s22rig is None:
            raise ValueError("must provide rigid cable s11/s12/s22 if Lh=-2")

        # Unlike for the S11s of the loads and the receiver, the transforms are fixed
        # here (not set from the data), and no delay is removed before fitting.
        hlc_model_params = _s11_model_params(
            params.nfit2,
            fourier=params.nfit2 > 16,
            fourier_transform=mdl.ZerotooneTransform(
                range=(s11freq.min().to_value("MHz"), s11freq.max().to_value("MHz"))
            ),
            polynomial_transform=mdl.Log10Transform(scale=1),
            set_transform_range=False,
            fit_method="lstsq",
            find_model_delay=False,
            combine_s12s21=False,
        )

        hot_load_cable_s11 = SParams(
            s11=s11rig,
            s12=s12rig,
            s21=s12rig,
            s22=s22rig,
            freqs=s11freq,
        ).smoothed(params=hlc_model_params, freqs=spec_fq)
        # Here we do something a little bogus. Alan fits the smoothing model
        # to the product s12*s21, which is what s12rig represents. So that's
        # why we passed s12rig as s12 above (instead of passing sqrt(s12rig)).
        # Now, we need to take the sqrt of s12 so have the right values.
        hot_load_cable_s11 = attrs.evolve(
            hot_load_cable_s11,
            s12=np.sqrt(hot_load_cable_s11.s12),
            s21=np.sqrt(hot_load_cable_s11.s12),
        )
        hot_loss_model = LossFunctionGivenSparams(sparams=hot_load_cable_s11)
    elif isinstance(params.Lh, Path):
        hot_loss_model = get_loss_model_from_file(params.Lh)
    else:
        hot_loss_model = None

    return hot_loss_model


def edges(
    spcold: GSData,
    sphot: GSData,
    spopen: GSData,
    spshort: GSData,
    s11freq: tp.FreqType,
    s11hot: np.ndarray,
    s11cold: np.ndarray,
    s11lna: np.ndarray,
    s11open: np.ndarray,
    s11short: np.ndarray,
    tload: tp.TemperatureType,
    tcal: tp.TemperatureType,
    params: EdgesScriptParams | None = None,
    s11rig: np.ndarray | None = None,
    s12rig: np.ndarray | None = None,
    s22rig: np.ndarray | None = None,
    **kwargs,
) -> tuple[
    CalibrationObservation, Calibrator, S11ModelParams, S11ModelParams, Callable | None
]:
    """A function that does what the edges3.c and edges2k.c C-code do.

    The primary purpose of this function is to model the input S11's, and then
    determine the noise-wave parameters. The :class:`CalibrationObservation` is
    assembled with :meth:`CalibrationObservation.from_components` (as in
    :mod:`edges.cal`), from the given spectra and the raw S11s within
    ``fstart``/``fstop``.

    Parameters
    ----------
    spcold, sphot, spopen, spshort
        The time-averaged spectra for the ambient (cold), hot_load, open, and short
        loads respectively.
    s11freq
        The frequencies at which the S11 data is sampled.
    s11hot, s11cold, s11lna, s11open, s11short
        The S11 measurements for the hot, ambient, LNA, open, and short loads
        respectively.
    tload
        A guess of the internal load temperature (a temperature Quantity), used as the
        initial guess for the optimization. **MUST MATCH** tload used to generate the
        time-averaged spectra.
    tcal
        Like tload, but for the internal load + noise source.
    params
        An object defining the parameters used in determining the calibration. By
        default, ``EdgesScriptParams(**kwargs)``.
    s11rig, s12rig, s22rig
        The S11, S12, and S22 measurements for the semi-rigid cable respectively.
        Optional -- generally required for EDGES-2.
    **kwargs
        Any field of :class:`EdgesScriptParams`, overriding its value in ``params``.

    Returns
    -------
    calobs
        The CalibrationObservation that holds all the data relevant for performing
        the receiver calibration.
    calibrator
        The final calibration solutions.
    s11_model_params
        The parameters used to create models of the load S11s.
    receiver_model_params
        The parameters used to create models of the receiver S11.
    hot_loss_model
        The model used to account for losses in the hot load.
    """
    if params is None:
        params = EdgesScriptParams(**kwargs)
    elif kwargs:
        params = attrs.evolve(params, **kwargs)

    # First set up the spectra, then the raw S11s and their models.
    specs = _get_specs(spcold, sphot, spopen, spshort, params, tload, tcal)
    spec_fq = specs["ambient"].freqs

    s11mask = get_mask(s11freq, low=params.fstart * un.MHz, high=params.fstop * un.MHz)
    s11freq = s11freq[s11mask]

    raw_load_s11s, s11_model_params = _get_load_s11s(
        params, s11cold, s11hot, s11open, s11short, s11mask, s11freq
    )
    raw_receiver, receiver_model_params = _get_receiver_s11(
        params, s11lna, s11mask, s11freq
    )

    hot_loss_model = _get_hotload_loss(
        params, s11rig, s12rig, s22rig, s11mask, s11freq, spec_fq
    )

    # The raw S11s are already restricted to fstart/fstop (and stored that way), so
    # the S11 models are fit to all of them. The hot-load loss model is already
    # defined at the frequencies of the spectra, so it must not be smoothed again
    # (loss_model_params=None). As in the C code, the ambient temperature of every
    # load is tcold.
    calobs = CalibrationObservation.from_components(
        spectra=specs,
        raw_s11s=raw_load_s11s,
        raw_receiver=raw_receiver,
        s11_model_params=s11_model_params,
        receiver_model_params=receiver_model_params,
        s11_freq_range=None,
        loss_models={"hot_load": hot_loss_model},
        loss_model_params=None,
        ambient_temperature=params.tcold * un.K,
    )
    calibrator = get_noise_wave_calibration_iterative(
        calobs,
        cterms=params.cfit,
        wterms=params.wfit,
        apply_loss_to_true_temp=False,
        smooth_scale_offset_within_loop=False,
        ncal_iter=params.nter,
        cable_delay_sweep=np.arange(0, -1e-8, -1e-9),  # hard-coded in the C code.
        fit_method="lstsq",  # "alan-qrd",
        scale_offset_poly_spacing=0.5,
        t_load_guess=tload,
        t_load_ns_guess=tcal,
    )
    return calobs, calibrator, s11_model_params, receiver_model_params, hot_loss_model


def _average_spectra(
    specfiles: dict[str, list[Path]],
    out: Path,
    redo_spectra: bool,
    fstart,
    fstop,
    telescope: str,
    **kwargs,
) -> dict[str, GSData]:
    spectra = {}
    for load, files in specfiles.items():
        outfile = out / f"sp{load}.txt"
        if redo_spectra or not outfile.exists():
            if len(files) == 0:
                raise ValueError(f"{load} has no spectrum files!")

            logger.info(f"Averaging {load} spectra")
            params = ACQPlot7aMoonParams(fstart=fstart, fstop=fstop, **kwargs)
            spectra[load], n = _acqplot7amoon(files, params)

            alanio.write_spec_txt_gsd(spectra[load], outfile, n=n)

        # Always read the spectra back in, because that's what Alan's C-code does.
        # This has the small effect of reducing the precision of the spectra.
        logger.info(f"Reading averaged {load} spectra")

        spectra[load] = alanio.read_spec_txt(
            outfile,
            time=spectra[load].times[0, 0] if load in spectra else Time.now(),
            telescope=telescope,
            name=load,
        )

    return spectra


@attrs.define(kw_only=True, frozen=True)
class Edges3CalobsParams:
    """
    Parameters defining the calibration observation data for EDGES 3.

    Parameters
    ----------
    specyear
        The year of the spectrum data.
    specday
        The day of the spectrum data.
    s11date
        The date of the S11 measurement in the format YYYY_DDD_HH.
    datadir
        The root directory of the observation data. By default, the ``edges3_data``
        option of the global configuration (:data:`edges.config.config`).
    match_resistance
        The measured impedance of the "match" calkit standard. Used to calibrate
        the Receiver s11.
    calkit_delays
        The delays of the three calkit standards. To set each individually, use
        the ``load_delay``, ``open_delay``, and ``short_delay`` parameters.
    load_delay
        The delay of the "load" calkit stsandard. By default the same as
        ``calkit_delays``.
    open_delay
        The delay of the "open" calkit standard. By default the same as
        ``calkit_delays``.
    short_delay
        The delay of the "short" calkit standard. By default the same as
        ``calkit_delays``.
    lna_cable_length
        The length of the cable joining the receiver to the VNA in inches.
    lna_cable_loss
        The loss of the cable joining the receiver to the VNA in percent.
    lna_cable_dielectric
        The dielectric constant of the cable joining the receiver to the VNA
        in percent.
    """

    specyear: int
    specday: int
    s11date: str
    datadir: Path = attrs.field(factory=lambda: config.edges3_data, converter=Path)
    match_resistance: Annotated[float, Parameter(name=("res",))] = 49.8
    calkit_delays: Annotated[float, Parameter(name=("ps",))] = 33
    load_delay: float = attrs.field()
    open_delay: float = attrs.field()
    short_delay: float = attrs.field()
    lna_cable_length: Annotated[float, Parameter(name=("cablen",))] = 4.26
    lna_cable_loss: Annotated[float, Parameter(name=("cabloss",))] = -91.5
    lna_cable_dielectric: Annotated[float, Parameter(name=("cabdiel",))] = -1.24

    @load_delay.default
    def _load_delay_default(self):
        return self.calkit_delays

    @open_delay.default
    def _open_delay_default(self):
        return self.calkit_delays

    @short_delay.default
    def _short_delay_default(self):
        return self.calkit_delays

    def get_caldef(self) -> CalObsDefEDGES3:
        """Get a calibration file definition."""
        return CalObsDefEDGES3.from_standard_layout(
            rootdir=self.datadir,
            year=self.specyear,
            day=self.specday,
            s11_year=int(self.s11date.split("_")[0]) if self.s11date else None,
            s11_day=int(self.s11date.split("_")[1]) if self.s11date else None,
            s11_hour=int(self.s11date.split("_")[2]) if self.s11date else None,
        )

    def get_raw_s11s(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Read all the raw S11 information."""
        caldef = self.get_caldef()

        raws11s = {}
        for load in [*list(caldef.loads.keys()), "receiver_s11"]:
            if load in caldef.loads:
                s11def = caldef.loads[load].s11
            else:
                s11def = caldef.receiver_s11

            raws11s[load] = reads1p1(
                s11_file_open=s11def.calkit.open,
                s11_file_short=s11def.calkit.short,
                s11_file_load=s11def.calkit.match,
                s11_file_ant=s11def.external,
                res=self.match_resistance,
                loadps=self.load_delay,
                openps=self.open_delay,
                shortps=self.short_delay,
            )

            if load == "receiver_s11":
                # Correction for path length
                raws11s[load] = corrcsv(
                    raws11s[load],
                    self.lna_cable_length,
                    self.lna_cable_dielectric,
                    self.lna_cable_loss,
                )

            # Update the precision of the raws11s because we need to match the
            # C-code which writes to file and reads it back in.
            # Don't round in-place: the freqs array belongs to a frozen object.
            s11freq = np.round(raws11s[load].freqs, decimals=16)
            raws11s[load] = (
                np.round(raws11s[load].reflection_coefficient.real, decimals=16)
                + np.round(raws11s[load].reflection_coefficient.imag, decimals=16) * 1j
            )
        return s11freq, raws11s

    def get_spectrum_files(self) -> dict[str, list[Path]]:
        """Return a dictionary of all associated spectrum files."""
        caldef = self.get_caldef()
        return {name: load.spectra for name, load in caldef.loads.items()}


@attrs.define(kw_only=True, frozen=True)
class Edges2CalobsParams:
    """Parameters defining the calibration observation data for EDGES 2.

    Parameters
    ----------
    s11_path
        The path to the S11 measurement file. Currently, this file must be a single,
        precalibrated file in Raul's legacy output format (CSV).
    ambient_acqs
        A list of paths to the ambient acq spectra files.
    hot_load_acqs
        A list of paths to the hot load acq spectra files.
    open_acqs
        A list of paths to the open acq spectra files.
    short_acqs
        A list of paths to the short acq spectra files.
    """

    s11_path: Path
    ambient_acqs: list[Path]
    hot_load_acqs: list[Path]
    open_acqs: list[Path]
    short_acqs: list[Path]

    def get_raw_s11s(self):
        """Read all the raw S11 information."""
        s11s = alanio.read_raul_s11_format(self.s11_path)
        s11freq = s11s.pop("freq") << un.MHz
        s11s = {alanio.LOADMAP.get(load, load): val for load, val in s11s.items()}

        return s11freq, s11s

    def get_spectrum_files(self):
        """Return a dictionary of all associated spectrum files."""
        return {
            "ambient": self.ambient_acqs,
            "hot_load": self.hot_load_acqs,
            "open": self.open_acqs,
            "short": self.short_acqs,
        }


def alancal(
    defparams: Edges3CalobsParams | Edges2CalobsParams,
    out: Path = Path(),
    redo_spectra: bool = False,
    redo_cal: bool = True,
    acqparams: ACQPlot7aMoonParams = ACQPlot7aMoonParams(),
    calparams: EdgesScriptParams = EdgesScriptParams(),
) -> (
    tuple[
        CalibrationObservation,
        Calibrator,
        S11ModelParams,
        S11ModelParams,
        Callable | None,
    ]
    | None
):
    """Run a calibration in as close a manner to Alan's code as possible.

    This exists mostly for being able to compare to Alan's memos etc in an easy way. It
    is much less flexible than using the library directly, and is not recommended for
    general use.

    This is supposed to emulate one of Alan's C-shell scripts, usually called "docal",
    and thus it runs a complete calibration, not just a single part. However, you can
    turn off parts of the calibration by setting the appropriate flags to False.

    Parameters
    ----------
    defparams
        Parameters that define where to find files and how to read/calibrate the
        raw S11s. This is different between EDGES3 and EDGES2.
    out
        A directory where outputs can be written.
    redo_spectra
        Whether to re-average the spectra if they already exist in the output directory.
    redo_cal
        Whether to re-compute the calibration coefficients if they already exist
        (i.e. if ``out`` contains a ``specal.txt`` file). If False and they exist,
        nothing is computed (except for re-averaging the spectra, if
        ``redo_spectra`` is True) and None is returned.
    acqparams
        Parameters governing how to average the spectrum files.
    calparams
        Parameters governing how to model S11s and perform the calibration.

    Returns
    -------
    calobs
        The CalibrationObservation that holds all the data relevant for performing
        the receiver calibration.
    calibrator
        The final calibration solutions.
    s11_model_params
        The parameters used to create models of the load S11s.
    receiver_model_params
        The parameters used to create models of the receiver S11.
    hot_loss_model
        The model used to account for losses in the hot load.

    If the calibration is not re-run (see ``redo_cal``), None is returned instead.
    """
    out = Path(out)

    # If the calibration exists and is not to be redone, the only thing left to do
    # is to re-average the spectra (if requested).
    skip_cal = not redo_cal and (out / "specal.txt").exists()
    if skip_cal and not redo_spectra:
        logger.info(f"Calibration already exists in {out}, not re-running it.")
        return None

    s11freq, raws11s = defparams.get_raw_s11s()
    specfiles = defparams.get_spectrum_files()

    if "receiver_s11" in raws11s:
        lna = raws11s.pop("receiver_s11")
    else:
        lna = raws11s.pop("lna")

    # Now average the spectra
    spectra = _average_spectra(
        specfiles=specfiles,
        out=out,
        redo_spectra=redo_spectra,
        fstart=acqparams.fstart,
        fstop=acqparams.fstop,
        smooth=acqparams.smooth,
        tload=acqparams.tload,
        tcal=acqparams.tcal,
        tstart=acqparams.tstart,
        tstop=acqparams.tstop,
        delaystart=acqparams.delaystart,
        telescope="edges3"
        if isinstance(defparams, Edges3CalobsParams)
        else "edges-low",
    )

    # Now do the calibration
    if skip_cal:
        logger.info(f"Calibration already exists in {out}, not re-running it.")
        return None

    logger.info("Performing calibration")
    return edges(
        spcold=spectra["ambient"],
        sphot=spectra["hot_load"],
        spopen=spectra["open"],
        spshort=spectra["short"],
        s11freq=s11freq,
        s11cold=raws11s["ambient"],
        s11hot=raws11s["hot_load"],
        s11open=raws11s["open"],
        s11short=raws11s["short"],
        s11lna=lna,
        s11rig=raws11s.get("s11rig"),
        s12rig=raws11s.get("s12rig"),
        s22rig=raws11s.get("s22rig"),
        params=calparams,
        tcal=acqparams.tcal * un.K,
        tload=acqparams.tload * un.K,
    )
