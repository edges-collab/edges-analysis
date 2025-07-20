"""Functions that run the calibration in a style similar to the C-code."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from pygsdata import GSData

import numpy as np
from astropy import units as un
from astropy.constants import c as speed_of_light
from pygsdata.select import select_freqs, select_times
from read_acq.gsdata import read_acq_to_gsdata

from edges.io.calobsdef3 import get_s1p_files

from .. import modelling as mdl
from ..averaging.freqbin import gauss_smooth
from ..averaging.lstbin import average_over_times
from ..io.vna import SParams
from ..cal import reflection_coefficient as rc
from ..cal.calobs import CalibrationObservation, Calibrator, Load
from ..cal.dicke import dicke_calibration
from ..cal.loss import HotLoadCorrection, get_cable_loss_model, get_loss_model_from_file
from ..cal.s11 import S11Model, StandardsReadings
from ..cal.spectra import LoadSpectrum
from ..cal.s11.receiver import correct_receiver_for_extra_cable
from ..cal.apply import approximate_temperature
from edges.frequencies import get_mask
from ..cal.receiver_cal import get_calcoeffs_iterative

from . import alanio

logger = logging.getLogger(__name__)


def reads1p1(
    res: float,
    Tfopen: str,
    Tfshort: str,
    Tfload: str,
    Tfant: str,
    loadps: float = 33.0,
    openps: float = 33.0,
    shortps: float = 33.0,
):
    """Reads the s1p1 file and returns the data."""
    standards = StandardsReadings(
        open=SParams.from_s1p_file(Tfopen),
        short=SParams.from_s1p_file(Tfshort),
        match=SParams.from_s1p_file(Tfload),
    )
    load = SParams.from_s1p_file(Tfant)
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
    freq: np.ndarray, s11: np.ndarray, cablen: float, cabdiel: float, cabloss: float
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
    cable_length = (cablen * un.imperial.inch).to("m")
    return correct_receiver_for_extra_cable(
        s11_in=s11, freq=freq*un.MHz, cable_length=cable_length,
        cable_loss_percent=cabloss, cable_dielectric_percent=cabdiel
    )


def acqplot7amoon(
    acqfile: str | Path,
    fstart: float,
    fstop: float,
    smooth: int = 8,
    tload: float = 300.0,
    tcal: float = 1000.0,
    pfit: int | None = None,
    rfi: float | None = None,
    peakpwr: float | None = None,
    minpwr: float | None = None,
    pkpwrm: float | None = None,
    maxrmsf: float | None = None,
    maxfm: float | None = None,
    nrfi: int = 0,
    tstart: int = 0,
    tstop: int = 23,
    delaystart: int = 0,
) -> GSData:
    """A function that does what the acqplot7amoon C-code does."""
    # We raise/warn when non-implemented parameters are passed. Serves as a reminder
    # to implement them in the future as necessary
    if any(p is not None for p in (pfit, rfi, peakpwr, minpwr, pkpwrm, maxrmsf, maxfm)):
        warnings.warn(
            "pfit, rfi, peakpwr, minpwr, pkpwrm, maxrmsf, and maxfm are not yet "
            "implemented. This is almost certainly OK for calibration purposes, as no "
            "calibration load data is typically filtered out by these parameters.",
            stacklevel=2,
        )

    data = read_acq_to_gsdata(acqfile, telescope="edges-low")

    if tstart > 0 or tstop < 23:
        # Note that tstop=23 includes all possible hours since we have <=
        hours = np.array([x.hour for x in data.times[:, 0].datetime])
        data = select_times(data, indx=(hours >= tstart) & (hours <= tstop))

    if delaystart > 0:
        secs = (data.times - data.times.min()).sec
        idx = np.all(secs > delaystart, axis=1)
        data = select_times(data, indx=idx)

    data = select_freqs(data, freq_range=(fstart * un.MHz, fstop * un.MHz))
    q = dicke_calibration(data)

    if smooth > 0:
        q = gauss_smooth(q, size=smooth, decimate_at=0)

    q = average_over_times(q)
    return approximate_temperature(data=q, t_load=tload, tns=tcal)


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
    Lh: int = -1,
    wfstart: float = 50,
    wfstop: float = 190,
    tcold: float = 306.5,
    thot: float = 393.22,
    tcab: float | None = None,
    cfit: int = 7,
    wfit: int = 7,
    nfit3: int = 10,
    nfit2: int = 27,
    tload: float = 300,
    tcal: float = 1000.0,
    nter: int = 8,
    mfit: int | None = None,
    smooth: int | None = None,
    lmode: int | None = None,
    tant: float | None = None,
    ldb: float | None = None,
    adb: float | None = None,
    delaylna: float | None = None,
    nfit4: int | None = None,
    s11rig: np.ndarray | None = None,
    s12rig: np.ndarray | None = None,
    s22rig: np.ndarray | None = None,
    lna_poly: int = -1,
) -> tuple[CalibrationObservation, Calibrator]:
    """A function that does what the edges3.c and edges2k.c C-code do.

    The primary purpose of this function is to model the input S11's, and then
    determine the noise-wave parameters.
    """
    # Some of the parameters are defined, but not yet implemented,
    # so we warn/error here. We do this explicitly because it serves as a
    # reminder to implement them in the future as necessary
    if tcab is None:
        tcab = tcold

    if mfit is not None or smooth is not None or tant is not None:
        warnings.warn(
            "mfit, smooth and tant are not used in this function, because "
            "they are only used for making output plots in the C-code."
            "They can be used in higher-level scripts instead. Continuing...",
            stacklevel=2,
        )

    if any(p is not None for p in (lmode, ldb, adb, delaylna, nfit4)):
        raise NotImplementedError(
            "lmode, ldb, adb, delaylna, and nfit4 are not yet implemented."
        )

    # First set up the S11 models
    sources = ["ambient", "hot_load", "open", "short"]
    s11_models = {}
    s11mask = get_mask(s11freq, low=wfstart * un.MHz, high=wfstop * un.MHz)
    s11freq = s11freq[s11mask]

    for name, s11 in zip(sources, [s11cold, s11hot, s11open, s11short], strict=False):
        s11_models[name] = S11Model(
            raw_s11=s11[s11mask],
            freq=s11freq,
            n_terms=nfit2,
            model_type=mdl.Fourier if nfit2 > 16 else mdl.Polynomial,
            complex_model_type=mdl.ComplexRealImagModel,
            model_transform=mdl.ZerotooneTransform(range=(1, 2))
            if nfit2 > 16
            else mdl.Log10Transform(scale=1),
            set_transform_range=True,
            fit_kwargs={"method": "alan-qrd"},
            model_kwargs={"period": 1.5},
        ).with_model_delay()

    mt = mdl.Fourier if (nfit3 > 16 or lna_poly == 0) else mdl.Polynomial

    receiver = S11Model(
        raw_s11=s11lna[s11mask],
        freq=s11freq,
        n_terms=nfit3,
        model_type=mt,
        complex_model_type=mdl.ComplexRealImagModel,
        model_transform=mdl.ZerotooneTransform(range=(1, 2))
        if mt == mdl.Fourier
        else mdl.Log10Transform(scale=120),
        set_transform_range=True,
        fit_kwargs={"method": "alan-qrd"},
        model_kwargs={"period": 1.5} if mt == mdl.Fourier else {},
    ).with_model_delay()

    specs = {}

    for name, spec, temp in zip(
        sources,
        [spcold, sphot, spopen, spshort],
        [tcold, thot, tcab, tcab],
        strict=False,
    ):
        specs[name] = LoadSpectrum(
            q=approximate_temperature(spec, tload=tload, tns=tcal, reverse=True),
            temp_ave=temp,
        )
        specs[name] = specs[name].between_freqs(wfstart * un.MHz, wfstop * un.MHz)

    if Lh == -1:
        hot_loss_model = get_cable_loss_model("UT-141C-SP")
    elif Lh == -2:
        if s11rig is None or s12rig is None or s22rig is None:
            raise ValueError("must provide rigid cable s11/s12/s22 if Lh=-2")
        mdlopts = {
            "transform": (
                mdl.ZerotooneTransform(
                    range=(s11freq.min().to_value("MHz"), s11freq.max().to_value("MHz"))
                )
                if nfit2 > 16
                else mdl.Log10Transform(scale=1)
            ),
            "n_terms": nfit2,
        }
        if nfit2 > 16:
            mdlopts["period"] = 1.5

        hot_loss_model = HotLoadCorrection(
            freq=s11freq,
            raw_s11=s11rig,
            raw_s12s21=s12rig,
            raw_s22=s22rig,
            model=mdl.Fourier(**mdlopts) if nfit2 > 16 else mdl.Polynomial(**mdlopts),
            complex_model=mdl.ComplexRealImagModel,
        )
    elif isinstance(Lh, Path):
        hot_loss_model = get_loss_model_from_file(Lh)
    else:
        hot_loss_model = None

    loads = {
        name: Load(
            spectrum=specs[name],
            reflections=s11_models[name],
            loss_model=hot_loss_model,
            ambient_temperature=tcold,
        )
        for name in specs
    }

    calobs = CalibrationObservation(
        loads=loads, receiver=receiver
    )
    calibrator = get_calcoeffs_iterative(
        calobs, 
        cterms=cfit,
        wterms=wfit,
        apply_loss_to_true_temp=False,
        smooth_scale_offset_within_loop=False,
        ncal_iter=nter,
        cable_delay_sweep=np.arange(0, -1e-8, -1e-9),  # hard-coded in the C code.
        fit_method="alan-qrd",
        scale_offset_poly_spacing=0.5,
    )
    return calobs, calibrator

def _average_spectra(
    specfiles: dict[str, list[Path]],
    out: Path,
    redo_spectra: bool,
    avg_spectra_path,
    fstart,
    fstop,
    telescope: str,
    **kwargs,
) -> GSData:
    spectra = {}
    for load, files in specfiles.items():
        outfile = out / f"sp{load}.txt"
        if (redo_spectra or not outfile.exists()) and not avg_spectra_path:
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
            outfile if outfile.exists() else avg_spectra_path,
            time = spectra[load].times[0,0] if load in spectra else Time.now(),
            telescope = telescope,
            name=load,
        )

    return spectra


def get_s11date(datadir, year, day):
    """Return the name of the nearest s11 date.

    Eg. 2022_318_14 for an input (2022,316).
    Default finds a file within 5 days of the input.
    This can be changed with allow_closes_within argument.
    """
    file_name = get_s1p_files(root_dir=datadir, year=year, day=day, load="open")[
        "input"
    ].name

    return file_name.rsplit("_", 1)[0]


def alancal(
    specyear,
    specday,
    datadir="/data5/edges/data/EDGES3_data/MRO/",
    out=".",
    redo_s11=True,
    redo_spectra=False,
    redo_cal=True,
    match_resistance=49.8,
    calkit_delays=33,
    load_delay=None,
    open_delay=None,
    short_delay=None,
    lna_cable_length=4.26,
    lna_cable_loss=-91.5,
    lna_cable_dielectric=-1.24,
    fstart=50.0,
    fstop=120.0,
    smooth=8,
    tload=300,
    tcal=1000,
    Lh=-1,
    wfstart=52.0,
    wfstop=118.0,
    tcold=306.5,
    thot=393.22,
    tcab=306.5,
    cfit=5,
    wfit=6,
    nfit3=10,
    nfit2=23,
    plot=False,
    avg_spectra_path=None,
    tstart=0,
    tstop=24,
    delaystart=0,
    s11date=None,
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
    s11date
        A date-string of the form 2022_319_04 (if doing EDGES-3 cal) or a full path
        to a file containing all calibrated S11s (if doing EDGES-2 cal).
        If no input is provided, a file within 5 days of the
        (specyear, specday) will be taken.
    specyear
        The year the spectra were taken in, if doing EDGES-3 cal. Otherwise, zero.
    specday
        The day the spectra were taken on, if doing EDGES-3 cal. Otherwise, zero.
    """
    loads = ("amb", "hot", "open", "short")
    datadir = Path(datadir)
    out = Path(out)

    if s11date is None:
        s11date = get_s11date(Path(datadir), year=specyear, day=specday)

    if load_delay is None:
        load_delay = calkit_delays
    if open_delay is None:
        open_delay = calkit_delays
    if short_delay is None:
        short_delay = calkit_delays

    raws11s = {}
    for load in (*loads, "lna"):
        outfile = out / f"s11{load}.csv"
        if redo_s11 or not outfile.exists():
            logger.info(f"Calibrating {load} S11")

            fstem = f"{s11date}_lna" if load == "lna" else s11date
            s11freq, raws11s[load] = reads1p1(
                Tfopen=Path(datadir) / f"{fstem}_O.s1p",
                Tfshort=Path(datadir) / f"{fstem}_S.s1p",
                Tfload=Path(datadir) / f"{fstem}_L.s1p",
                Tfant=Path(datadir) / f"{s11date}_{load}.s1p",
                res=match_resistance,
                loadps=load_delay,
                openps=open_delay,
                shortps=short_delay,
            )

            if load == "lna":
                # Correction for path length
                raws11s[load] = corrcsv(
                    s11freq,
                    raws11s[load],
                    lna_cable_length,
                    lna_cable_dielectric,
                    lna_cable_loss,
                )

            # write out the CSV file
            with open(out / f"s11{load}.csv", "w") as fl:
                fl.write("BEGIN\n")
                for freq, s11 in zip(s11freq, raws11s[load], strict=False):
                    fl.write(
                        f"{freq.to_value('MHz'):1.16e},{s11.real:1.16e},{s11.imag:1.16e}\n"
                    )
                fl.write("END")

        # Always re-read the S11's to match the precision of the C-code.
        logger.info(f"Reading calibrated {load} S11")
        s11freq, raws11s[load] = read_s11_csv(outfile)
        s11freq <<= un.MHz

    lna = raws11s.pop("lna")

    # Now average the spectra
    spectra = {}
    specdate = f"{specyear:04}_{specday:03}"
    specfiles = {
        load: sorted(
            Path(f"{datadir}/mro/{load}/{specyear:04}").glob(f"{specdate}*{load}.acq")
        )
        for load in loads
    }
    spectra = _average_spectra(
        specfiles,
        out,
        redo_spectra,
        avg_spectra_path,
        fstart=fstart,
        fstop=fstop,
        smooth=smooth,
        tload=tload,
        tcal=tcal,
        tstart=tstart,
        tstop=tstop,
        delaystart=delaystart,
        telescope='edges3',

    )

    # Now do the calibration
    outfile = out / "specal.txt"
    if not redo_cal and outfile.exists():
        return None

    logger.info("Performing calibration")
    return edges(
        spfreq=spectra['amb'].freqs,
        spcold=spectra["amb"].data.squeeze(),
        sphot=spectra["hot"].data.squeeze(),
        spopen=spectra["open"].data.squeeze(),
        spshort=spectra["short"].data.squeeze(),
        s11freq=s11freq,
        s11cold=raws11s["amb"],
        s11hot=raws11s["hot"],
        s11open=raws11s["open"],
        s11short=raws11s["short"],
        s11lna=lna,
        Lh=Lh,
        wfstart=wfstart,
        wfstop=wfstop,
        tcold=tcold,
        thot=thot,
        tcab=tcab,
        cfit=cfit,
        wfit=wfit,
        nfit3=nfit3,
        nfit2=nfit2,
        tload=tload,
        tcal=tcal,
    )


def alancal2(
    s11_path: Path,
    ambient_acqs: list[Path],
    hotload_acqs: list[Path],
    open_acqs: list[Path],
    short_acqs: list[Path],
    out: Path,
    redo_spectra: bool = True,
    redo_cal: bool = True,
    fstart: float = 40.0,
    fstop: float = 110.0,
    smooth: int = 8,
    tload: float = 300.0,
    tcal: float = 1000.0,
    Lh: int = -2,
    wfstart: float = 50.0,
    wfstop: float = 100.0,
    tcold: float = 296.0,
    thot: float = 399.0,
    tcab: float | None = None,
    cfit: int = 6,
    wfit: int = 5,
    nfit3: int = 11,
    nfit2: int = 27,
    avg_spectra_path: Path | None = None,
    s11s_in_raul_format: bool = True,
    lna_poly: int = 0,
    tstart: float = 0,
    tstop: float = 23,
    delaystart: float = 7200,
) -> CalibrationObservation:
    """Run a calibration in as close a manner to Alan's code as possible.

    This exists mostly for being able to compare to Alan's memos etc in an easy way. It
    is much less flexible than using the library directly, and is not recommended for
    general use.

    This is supposed to emulate one of Alan's C-shell scripts, usually called "docal",
    and thus it runs a complete calibration, not just a single part. However, you can
    turn off parts of the calibration by setting the appropriate flags to False.

    Parameters
    ----------
    s11date
        A date-string of the form 2022_319_04 (if doing EDGES-3 cal) or a full path
        to a file containing all calibrated S11s (if doing EDGES-2 cal).
    specyear
        The year the spectra were taken in, if doing EDGES-3 cal. Otherwise, zero.
    specday
        The day the spectra were taken on, if doing EDGES-3 cal. Otherwise, zero.
    """
    if s11_path is None or not Path(s11_path).exists():
        raise ValueError("s11_path does not exist")

    out = Path(out)

    if s11s_in_raul_format:
        s11s = read_raul_s11_format(s11_path)
        s11freq = s11s.pop("freq") << un.MHz
        raws11s = s11s
    else:
        raise NotImplementedError(
            "We have not yet implemented S11 calibration in alanmode."
        )

    lna = raws11s.pop("lna")

    # Now average the spectra
    specfiles = {
        "amb": [Path(fl) for fl in ambient_acqs],
        "hot": [Path(fl) for fl in hotload_acqs],
        "short": [Path(fl) for fl in short_acqs],
        "open": [Path(fl) for fl in open_acqs],
    }
    spectra = _average_spectra(
        specfiles,
        out,
        redo_spectra,
        avg_spectra_path,
        fstart=fstart,
        fstop=fstop,
        smooth=smooth,
        tload=tload,
        tcal=tcal,
        tstart=tstart,
        tstop=tstop,
        delaystart=delaystart,
        telescope='edges-low'
    )

    # Now do the calibration
    outfile = out / "specal.txt"
    if not redo_cal and outfile.exists():
        return None

    logger.info("Performing calibration")
    return edges(
        spfreq=spectra['amb'].freqs,
        spcold=spectra["amb"].data.squeeze(),
        sphot=spectra["hot"].data.squeeze(),
        spopen=spectra["open"].data.squeeze(),
        spshort=spectra["short"].data.squeeze(),
        s11freq=s11freq,
        s11cold=raws11s["amb"],
        s11hot=raws11s["hot"],
        s11open=raws11s["open"],
        s11short=raws11s["short"],
        s11lna=lna,
        Lh=Lh,
        wfstart=wfstart,
        wfstop=wfstop,
        tcold=tcold,
        thot=thot,
        tcab=tcab,
        cfit=cfit,
        wfit=wfit,
        nfit3=nfit3,
        nfit2=nfit2,
        tload=tload,
        tcal=tcal,
        lna_poly=lna_poly,
        s11rig=raws11s["s11rig"],
        s12rig=raws11s["s12rig"],
        s22rig=raws11s["s22rig"],
    )

