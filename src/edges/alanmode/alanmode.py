"""Functions that run the calibration in a style similar to the C-code."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Annotated

import attrs
import numpy as np
from astropy import units as un
from astropy.time import Time
from cyclopts import Parameter
from pygsdata import GSData
from pygsdata.select import select_freqs, select_times
from read_acq.gsdata import read_acq_to_gsdata

from edges.cal.s11.base import CalibratedSParams
from edges.frequencies import get_mask
from edges.io.calobsdef3 import CalObsDefEDGES3

from .. import modelling as mdl
from .. import types as tp
from ..averaging.freqbin import gauss_smooth
from ..averaging.lstbin import average_over_times
from ..cal import reflection_coefficient as rc
from ..cal.apply import approximate_temperature
from ..cal.calobs import CalibrationObservation, Calibrator, Load
from ..cal.dicke import dicke_calibration
from ..cal.loss import (
    LossFunctionGivenSparams,
    get_cable_loss_model,
    get_loss_model_from_file,
)
from ..cal.receiver_cal import get_calcoeffs_iterative
from ..cal.s11 import CalibratedS11, StandardsReadings
from ..cal.s11.receiver import correct_receiver_for_extra_cable
from ..cal.s11.s11model import S11ModelParams
from ..cal.spectra import LoadSpectrum
from ..io.vna import SParams
from . import alanio

logger = logging.getLogger(__name__)

SOURCES = ["ambient", "hot_load", "open", "short"]


def reads1p1(
    res: float,
    s11_file_open: str,
    s11_file_short: str,
    s11_file_load: str,
    s11_file_ant: str,
    loadps: float = 33.0,
    openps: float = 33.0,
    shortps: float = 33.0,
):
    """Reads the s1p1 file and returns the data."""
    standards = StandardsReadings(
        open=SParams.from_s1p_file(s11_file_open),
        short=SParams.from_s1p_file(s11_file_short),
        match=SParams.from_s1p_file(s11_file_load),
    )
    load = SParams.from_s1p_file(s11_file_ant)
    freq = standards.freq

    calkit = rc.get_calkit(rc.AGILENT_ALAN, resistance_of_match=res * un.ohm)

    calkit = calkit.clone(
        short={"offset_delay": shortps * un.ps},
        open={"offset_delay": openps * un.ps},
        match={"offset_delay": loadps * un.ps},
    )
    smatrix = rc.SMatrix.from_calkit_and_vna(calkit, standards)
    calibrated = rc.gamma_de_embed(load.s11, smatrix)
    return freq, calibrated


def corrcsv(
    freq: np.ndarray | tp.FreqType,
    s11: np.ndarray,
    cablen: float,
    cabdiel: float,
    cabloss: float,
):
    """Corrects the S11 data (LNA) for cable effects.

    This function is a direct translation of the C-code function corrcsv.

    Parameters
    ----------
    freq : np.ndarray
        The frequency array.
    s11 : np.ndarray
        The S11 data.
    cablen : float
        The cable length, in inches.
    cabdiel : float
        The cable dielectric constant, as a percent.
    cabloss : float
        The cable loss, as a percent.
    """
    if not hasattr(freq, "unit"):
        freq = freq * un.MHz

    cable_length = (cablen * un.imperial.inch).to("m")
    return correct_receiver_for_extra_cable(
        s11_in=s11,
        freq=freq,
        cable_length=cable_length,
        cable_loss_percent=cabloss,
        cable_dielectric_percent=cabdiel,
    )


@attrs.define(kw_only=True, frozen=True)
class ACQPlot7aMoonParams:
    fstart: float = 50.0
    fstop: float = 100.0
    smooth: int = 8
    tload: float = 300.0
    tcal: float = 1000.0
    pfit: int | None = None
    rfi: float | None = None
    peakpwr: float | None = None
    minpwr: float | None = None
    pkpwrm: float | None = None
    maxrmsf: float | None = None
    maxfm: float | None = None
    nrfi: int = 0
    tstart: int = 0
    tstop: int = 23
    delaystart: int = 0

    def __attrs_post_init__(self):
        """Warn if any non-implemented parameters are given."""
        if any(
            p is not None
            for p in (
                self.pfit,
                self.rfi,
                self.peakpwr,
                self.minpwr,
                self.pkpwrm,
                self.maxrmsf,
                self.maxfm,
            )
        ):
            warnings.warn(
                "pfit, rfi, peakpwr, minpwr, pkpwrm, maxrmsf, and maxfm are not yet "
                "implemented. This is almost certainly OK for calibration purposes, as "
                "no calibration load data is typically filtered out by these "
                "parameters.",
                stacklevel=2,
            )


def acqplot7amoon(
    acqfile: str | Path, params: ACQPlot7aMoonParams = ACQPlot7aMoonParams(), **kwargs
) -> GSData:
    """A function that does what the acqplot7amoon C-code does."""
    if kwargs:
        params = ACQPlot7aMoonParams(**kwargs)

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

    data = select_freqs(
        data, freq_range=(params.fstart * un.MHz, params.fstop * un.MHz)
    )
    q = dicke_calibration(data)

    if params.smooth > 0:
        q = gauss_smooth(q, size=params.smooth, decimate_at=0)

    q = average_over_times(q)
    return approximate_temperature(data=q, tload=params.tload, tns=params.tcal)


@attrs.define(kw_only=True, frozen=True)
class EdgesScriptParams:
    Lh: Annotated[int, Parameter(name=("Lh",))] = -1
    wfstart: float = 50
    wfstop: float = 190
    tcold: float = 306.5
    thot: float = 393.22
    tcab: float = attrs.field()
    cfit: int = 7
    wfit: int = 7
    nfit3: int = 10
    nfit2: int = 27
    tload: float = 300
    tcal: float = 1000.0
    nter: int = 8
    mfit: int | None = None
    lmode: int | None = None
    tant: float | None = None
    ldb: float | None = None
    adb: float | None = None
    delaylna: float | None = None
    nfit4: int | None = None
    lna_poly: int = -1

    @tcab.default
    def _tcab_default(self) -> float:
        return self.tcold

    def __attrs_post_init__(self):
        """Warn if non-implemented parameters are given.

        Some of the parameters are defined, but not yet implemented,
        so we warn/error here. We do this explicitly because it serves as a
        reminder to implement them in the future as necessary
        """
        if self.mfit is not None or self.tant is not None:
            warnings.warn(
                "mfit, smooth and tant are not used in this function, because "
                "they are only used for making output plots in the C-code."
                "They can be used in higher-level scripts instead. Continuing...",
                stacklevel=2,
            )

        if any(
            p is not None
            for p in (self.lmode, self.ldb, self.adb, self.delaylna, self.nfit4)
        ):
            raise NotImplementedError(
                "lmode, ldb, adb, delaylna, and nfit4 are not yet implemented."
            )


def _get_specs(
    spcold: GSData,
    sphot: GSData,
    spopen: GSData,
    spshort: GSData,
    params: EdgesScriptParams,
) -> dict[str, LoadSpectrum]:
    specs = {}

    for name, spec, temp in zip(
        SOURCES,
        [spcold, sphot, spopen, spshort],
        [params.tcold, params.thot, params.tcab, params.tcab],
        strict=False,
    ):
        specs[name] = LoadSpectrum(
            q=approximate_temperature(
                spec, tload=params.tload, tns=params.tcal, reverse=True
            ),
            temp_ave=temp * un.K,
        )
        specs[name] = specs[name].between_freqs(
            params.wfstart * un.MHz, params.wfstop * un.MHz
        )
    return specs


def _get_load_s11s(
    params: EdgesScriptParams,
    s11cold,
    s11hot,
    s11open,
    s11short,
    s11mask: np.ndarray,
    s11freq: tp.FreqType,
    spec_freqs: tp.FreqType,
) -> tuple[dict[str, CalibratedS11], S11ModelParams, dict[str, CalibratedS11]]:
    raw_s11s = {
        name: CalibratedS11(
            s11=s11[s11mask],
            freqs=s11freq,
        )
        for name, s11 in zip(
            SOURCES, [s11cold, s11hot, s11open, s11short], strict=False
        )
    }

    mdltype = mdl.Fourier if params.nfit2 > 16 else mdl.Polynomial
    s11_modelling_params = S11ModelParams(
        model=mdltype(
            n_terms=params.nfit2,
            transform=(
                mdl.ZerotooneTransform(range=(1, 2))
                if params.nfit2 > 16
                else mdl.Log10Transform(scale=1)
            ),
            period=1.5,
        ),
        complex_model_type=mdl.ComplexRealImagModel,
        set_transform_range=True,
        fit_method="alan-qrd",
        find_model_delay=True,
    )

    s11_models = {
        name: s11.smoothed(params=s11_modelling_params, freqs=spec_freqs)
        for name, s11 in raw_s11s.items()
    }

    return raw_s11s, s11_modelling_params, s11_models


def _get_receiver_s11(params: EdgesScriptParams, s11lna, s11mask, s11freq, spec_fq):
    mt = mdl.Fourier if (params.nfit3 > 16 or params.lna_poly == 0) else mdl.Polynomial

    raw_receiver = CalibratedS11(
        s11=s11lna[s11mask],
        freqs=s11freq,
    )
    model_transform = (
        mdl.ZerotooneTransform(range=(1, 2))
        if mt == mdl.Fourier
        else mdl.Log10Transform(scale=120)
    )
    model_kwargs = {"period": 1.5} if mt == mdl.Fourier else {}
    receiver_model = S11ModelParams(
        model=mt(n_terms=params.nfit3, transform=model_transform, **model_kwargs),
        complex_model_type=mdl.ComplexRealImagModel,
        set_transform_range=True,
        fit_method="alan-qrd",
        find_model_delay=True,
    )
    receiver = raw_receiver.smoothed(receiver_model, freqs=spec_fq)
    return raw_receiver, receiver_model, receiver


def _get_hotload_loss(
    params: EdgesScriptParams,
    s11rig,
    s12rig,
    s22rig,
    s11mask,
    s11freq,
    spec_fq,
) -> callable | None:
    if params.Lh == -1:
        hot_loss_model = get_cable_loss_model("UT-141C-SP")
    elif params.Lh == -2:
        if s11rig is None or s12rig is None or s22rig is None:
            raise ValueError("must provide rigid cable s11/s12/s22 if Lh=-2")

        mdltype = mdl.Fourier if params.nfit2 > 16 else mdl.Polynomial

        mdlopts = {
            "transform": (
                mdl.ZerotooneTransform(
                    range=(s11freq.min().to_value("MHz"), s11freq.max().to_value("MHz"))
                )
                if params.nfit2 > 16
                else mdl.Log10Transform(scale=1)
            ),
            "n_terms": params.nfit2,
        }
        if params.nfit2 > 16:
            mdlopts["period"] = 1.5

        hlc_model_params = S11ModelParams(
            model=mdltype(**mdlopts),
            set_transform_range=False,
            complex_model_type=mdl.ComplexRealImagModel,
            fit_method="lstsq",
        )

        hot_load_cable_s11 = CalibratedSParams(
            s11=s11rig,
            s12=s12rig,
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
    s11freq: np.ndarray,
    s11hot: np.ndarray,
    s11cold: np.ndarray,
    s11lna: np.ndarray,
    s11open: np.ndarray,
    s11short: np.ndarray,
    tload: float,
    tcal: float,
    params: EdgesScriptParams = EdgesScriptParams(),
    s11rig: np.ndarray | None = None,
    s12rig: np.ndarray | None = None,
    s22rig: np.ndarray | None = None,
    **kwargs,
) -> tuple[CalibrationObservation, Calibrator]:
    """A function that does what the edges3.c and edges2k.c C-code do.

    The primary purpose of this function is to model the input S11's, and then
    determine the noise-wave parameters.
    """
    if kwargs:
        params = EdgesScriptParams(**kwargs)

    # First set up the S11 models
    specs = _get_specs(spcold, sphot, spopen, spshort, params)
    spec_fq = specs["ambient"].freqs

    s11mask = get_mask(
        s11freq, low=params.wfstart * un.MHz, high=params.wfstop * un.MHz
    )
    s11freq = s11freq[s11mask]

    raw_load_s11s, s11_model_params, load_s11s = _get_load_s11s(
        params, s11cold, s11hot, s11open, s11short, s11mask, s11freq, spec_fq
    )
    raw_receiver, receiver_model_params, receiver = _get_receiver_s11(
        params, s11lna, s11mask, s11freq, spec_fq=spec_fq
    )

    hot_loss_model = _get_hotload_loss(
        params, s11rig, s12rig, s22rig, s11mask, s11freq, spec_fq
    )

    loads = {
        name: Load(
            spectrum=specs[name],
            s11=load_s11s[name],
            raw_s11=raw_load_s11s[name],
            loss=hot_loss_model(spec_fq, load_s11s[name].s11)
            if name == "hot_load" and hot_loss_model
            else np.ones(spec_fq.size),
            ambient_temperature=params.tcold * un.K,
        )
        for name in specs
    }

    calobs = CalibrationObservation(
        loads=loads, receiver=receiver, raw_receiver=raw_receiver
    )
    calibrator = get_calcoeffs_iterative(
        calobs,
        cterms=params.cfit,
        wterms=params.wfit,
        apply_loss_to_true_temp=False,
        smooth_scale_offset_within_loop=False,
        ncal_iter=params.nter,
        cable_delay_sweep=np.arange(0, -1e-8, -1e-9),  # hard-coded in the C code.
        fit_method="alan-qrd",
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
) -> GSData:
    spectra = {}
    for load, files in specfiles.items():
        outfile = out / f"sp{load}.txt"
        if redo_spectra or not outfile.exists():
            if len(files) == 0:
                raise ValueError(f"{load} has no spectrum files!")

            logger.info(f"Averaging {load} spectra")
            spectra[load] = acqplot7amoon(
                acqfile=files, fstart=fstart, fstop=fstop, **kwargs
            )

            alanio.write_spec_txt_gsd(spectra[load], outfile)

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
    specyear: int
    specday: int
    datadir: Path = Path("/data5/edges/data/EDGES3_data/MRO/")
    match_resistance: Annotated[float, Parameter(name=("res",))] = 49.8
    calkit_delays: Annotated[float, Parameter(name=("ps",))] = 33
    load_delay: float = attrs.field()
    open_delay: float = attrs.field()
    short_delay: float = attrs.field()
    lna_cable_length: Annotated[float, Parameter(name=("cablen",))] = (4.26,)
    lna_cable_loss: Annotated[float, Parameter(name=("cabloss",))] = (-91.5,)
    lna_cable_dielectric: Annotated[float, Parameter(name=("cabdiel",))] = -1.24
    s11date: str

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

    def get_raw_s11s(self):
        """Read all the raw S11 information."""
        caldef = self.get_caldef()

        raws11s = {}
        for load in [*list(caldef.loads.keys()), "receiver_s11"]:
            if load in caldef.loads:
                s11def = caldef.loads[load].s11
            else:
                s11def = caldef.receiver_s11

            s11freq, raws11s[load] = reads1p1(
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
                    s11freq,
                    raws11s[load],
                    self.lna_cable_length,
                    self.lna_cable_dielectric,
                    self.lna_cable_loss,
                )

            # Update the precision of the raws11s because we need to match the
            # C-code which writes to file and reads it back in.
            np.round(s11freq, decimals=16, out=s11freq)
            raws11s[load] = (
                np.round(raws11s[load].real, decimals=16)
                + np.round(raws11s[load].imag, decimals=16) * 1j
            )
        return s11freq, raws11s

    def get_spectrum_files(self) -> dict[str, list[Path]]:
        """Return a dictionary of all associated spectrum files."""
        caldef = self.get_caldef()
        return {name: load.spectra for name, load in caldef.loads.items()}


@attrs.define(kw_only=True, frozen=True)
class Edges2CalobsParams:
    s11_path: Path
    ambient_acqs: list[Path]
    hotload_acqs: list[Path]
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
            "hot_load": self.hotload_acqs,
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
        Whether to re-compute the calibration coefficients if they already exist.
    acqparams
        Parameters governing how to average the spectrum files.
    calparams
        Parameters governing how to model S11s and perform the calibration.
    """
    out = Path(out)

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
    outfile = out / "specal.txt"
    if not redo_cal and outfile.exists():
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
