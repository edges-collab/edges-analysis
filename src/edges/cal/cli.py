"""CLI functions for edges-cal."""

import json
import os
from datetime import UTC, datetime
from importlib.util import find_spec
from pathlib import Path

import click
import numpy as np
import papermill as pm
import yaml
from astropy import units as un
from nbconvert import PDFExporter
from rich.console import Console
from traitlets.config import Config

from ..cal import calobs as cc
from ..cal.alanmode import (
    acqplot7amoon,
    read_spec_txt,
    write_modelled_s11s,
    write_spec_txt,
    write_specal,
)
from ..cal.alanmode import (
    alancal as acal,
)
from ..cal.alanmode import (
    alancal2 as acal2,
)
from ..cal.calobs import CalibrationObservation
from ..config import config

console = Console()

main = click.Group()


@main.command()
@click.argument(
    "settings", type=click.Path(dir_okay=False, file_okay=True, exists=True)
)
@click.argument("path", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.option(
    "-o",
    "--out",
    type=click.Path(dir_okay=True, file_okay=False, exists=True),
    default=".",
    help="output directory",
)
@click.option(
    "-g",
    "--global-config",
    type=str,
    default=None,
    help="json string representing global configuration options",
)
@click.option(
    "-p/-P",
    "--plot/--no-plot",
    default=True,
    help="whether to make diagnostic plots of calibration solutions.",
)
@click.option(
    "-s",
    "--simulators",
    multiple=True,
    default=[],
    help="antenna simulators to create diagnostic plots for.",
)
def run(settings, path, out, global_config, plot, simulators):
    """Calibrate using lab measurements in PATH, and make all relevant plots."""
    out = Path(out)

    if global_config:
        config.update(json.loads(global_config))

    obs = cc.CalibrationObservation.from_yaml(settings, obs_path=path)
    io_obs = obs.metadata["io"]
    if plot:
        # Plot Calibrator properties
        fig = obs.plot_raw_spectra()
        fig.savefig(out / "raw_spectra.png")

        ax = obs.plot_s11_models()
        fig = ax.flatten()[0].get_figure()
        fig.savefig(out / "s11_models.png")

        fig = obs.plot_calibrated_temps(bins=256)
        fig.savefig(out / "calibrated_temps.png")

        fig = obs.plot_coefficients()
        fig.savefig(out / "calibration_coefficients.png")

        # Calibrate and plot antsim
        for name in simulators:
            antsim = obs.new_load(load_name=name, io_obj=obs.metadata["io"])
            fig = obs.plot_calibrated_temp(antsim, bins=256)
            fig.savefig(out / f"{name}_calibrated_temp.png")

    # Write out data
    obs.write(out / io_obs.path.parent.name)


@main.command()
@click.argument("config", type=click.Path(dir_okay=False, file_okay=True, exists=True))
@click.argument("path", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.option(
    "-w", "--max-wterms", type=int, default=20, help="maximum number of wterms"
)
@click.option(
    "-r/-R",
    "--repeats/--no-repeats",
    default=False,
    help="explore repeats of switch and receiver s11",
)
@click.option(
    "-n/-N", "--runs/--no-runs", default=False, help="explore runs of s11 measurements"
)
@click.option(
    "-c", "--max-cterms", type=int, default=20, help="maximum number of cterms"
)
@click.option(
    "-w", "--max-wterms", type=int, default=20, help="maximum number of wterms"
)
@click.option(
    "-r/-R",
    "--repeats/--no-repeats",
    default=False,
    help="explore repeats of switch and receiver s11",
)
@click.option(
    "-n/-N", "--runs/--no-runs", default=False, help="explore runs of s11 measurements"
)
@click.option(
    "-t",
    "--delta-rms-thresh",
    type=float,
    default=0,
    help="threshold marking rms convergence",
)
@click.option(
    "-o",
    "--out",
    type=click.Path(dir_okay=True, file_okay=False, exists=True),
    default=".",
    help="output directory",
)
@click.option(
    "-c",
    "--cache-dir",
    type=click.Path(dir_okay=True, file_okay=False),
    default=".",
    help="directory in which to keep/search for the cache",
)
def sweep(
    config,
    path,
    max_cterms,
    max_wterms,
    repeats,
    runs,
    delta_rms_thresh,
    out,
    cache_dir,
):
    """Perform a sweep of number of terms to obtain the best parameter set."""
    with open(config) as fl:
        settings = yaml.load(fl, Loader=yaml.FullLoader)

    if cache_dir != ".":
        settings.update(cache_dir=cache_dir)

    obs = cc.CalibrationObservation(path=path, **settings)

    cc.perform_term_sweep(
        obs,
        direc=out,
        verbose=True,
        max_cterms=max_cterms,
        max_wterms=max_wterms,
        explore_repeat_nums=repeats,
        explore_run_nums=runs,
        delta_rms_thresh=delta_rms_thresh,
    )


@main.command()
@click.argument(
    "cal-settings",
    type=click.Path(dir_okay=False, file_okay=True, exists=True),
)
@click.argument("path", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.option(
    "-o",
    "--out",
    type=click.Path(dir_okay=True, file_okay=False, exists=True),
    default=None,
    help="output directory",
)
@click.option(
    "-g",
    "--global-config",
    type=str,
    default=None,
    help="json string representing global configuration options",
)
@click.option("-r/-R", "--report/--no-report", default=True)
@click.option("-u/-U", "--upload/--no-upload", default=False, help="auto-upload file")
@click.option("-t", "--title", type=str, help="title of the memo", default=None)
@click.option(
    "-a",
    "--author",
    type=str,
    help="adds an author to the author list",
    default=None,
    multiple=True,
)
@click.option("-n", "--memo", type=int, help="which memo number to use", default=None)
@click.option("-q/-Q", "--quiet/--loud", default=False)
@click.option("-p/-P", "--pdf/--no-pdf", default=True)
def report(
    cal_settings,
    path,
    out,
    global_config,
    report,
    upload,
    title,
    author,
    memo,
    quiet,
    pdf,
):
    """Make a full notebook report on a given calibration."""
    single_notebook = Path(__file__).parent / "notebooks/calibrate-observation.ipynb"

    console.print(f"Creating report for '{path}'...")

    path = Path(path)

    out = path / "outputs" if out is None else Path(out)

    if not out.exists():
        out.mkdir()

    # Describe the filename...
    fname = Path(
        f"calibration_{datetime.now(tz=UTC).strftime('%Y-%m-%d-%H.%M.%S')}.ipynb"
    )

    global_config = json.loads(global_config) if global_config else {}

    settings = {
        "observation": str(path),
        "settings": cal_settings,
        "global_config": global_config,
    }

    console.print("Settings:")
    with open(cal_settings) as fl:
        console.print(fl.read())

    # This actually runs the notebook itself.
    pm.execute_notebook(
        str(single_notebook),
        out / fname,
        parameters=settings,
        kernel_name="edges",
        log_output=True,
    )

    console.print(f"Saved interactive notebook to '{out / fname}'")

    if pdf:  # pragma: no cover
        make_pdf(out / fname)
        if upload:
            upload_memo(out / fname.with_suffix(".pdf"), title, memo, quiet)


@main.command()
@click.argument(
    "cal-settings",
    type=click.Path(dir_okay=False, file_okay=True, exists=True),
)
@click.argument("path", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.argument(
    "cmp-settings",
    type=click.Path(dir_okay=False, file_okay=True, exists=True),
)
@click.argument("cmppath", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.option(
    "-o",
    "--out",
    type=click.Path(dir_okay=True, file_okay=False, exists=True),
    default=None,
    help="output directory",
)
@click.option(
    "-g",
    "--global-config",
    type=str,
    default=".",
    help="global configuration options as json",
)
@click.option("-r/-R", "--report/--no-report", default=True)
@click.option("-u/-U", "--upload/--no-upload", default=False, help="auto-upload file")
@click.option("-t", "--title", type=str, help="title of the memo", default=None)
@click.option(
    "-a",
    "--author",
    type=str,
    help="adds an author to the author list",
    default=None,
    multiple=True,
)
@click.option("-n", "--memo", type=int, help="which memo number to use", default=None)
@click.option("-q/-Q", "--quiet/--loud", default=False)
@click.option("-p/-P", "--pdf/--no-pdf", default=True)
def compare(
    cal_settings,
    path,
    cmp_settings,
    cmppath,
    out,
    global_config,
    report,
    upload,
    title,
    author,
    memo,
    quiet,
    pdf,
):
    """Make a full notebook comparison report between two observations."""
    single_notebook = Path(__file__).parent / "notebooks/compare-observation.ipynb"

    console.print(f"Creating comparison report for '{path}' compared to '{cmppath}'")

    path = Path(path)
    cmppath = Path(cmppath)

    out = path / "outputs" if out is None else Path(out)

    if not out.exists():
        out.mkdir()

    # Describe the filename...
    fname = Path(
        f"calibration-compare-{cmppath.name}_"
        f"{datetime.now(tz=UTC).strftime('%Y-%m-%d-%H.%M.%S')}.ipynb"
    )

    global_config = json.loads(global_config) if global_config else {}

    console.print("Settings for Primary:")
    with open(cal_settings) as fl:
        console.print(fl.read())

    console.print("Settings for Comparison:")
    with open(cmp_settings) as fl:
        console.print(fl.read())

    # This actually runs the notebook itself.
    pm.execute_notebook(
        str(single_notebook),
        out / fname,
        parameters={
            "observation": str(path),
            "cmp_observation": str(cmppath),
            "settings": cal_settings,
            "cmp_settings": cmp_settings,
            "global_config": global_config,
        },
        kernel_name="edges",
    )
    console.print(f"Saved interactive notebook to '{out / fname}'")

    # Now output the notebook to pdf
    if pdf:  # pragma: no cover
        pdf = make_pdf(out / fname)
        if upload:
            upload_memo(pdf, title, memo, quiet)


def make_pdf(ipy_fname) -> Path:
    """Make a PDF out of an ipynb."""
    # Now output the notebook to pdf
    c = Config()
    c.TemplateExporter.exclude_input_prompt = True
    c.TemplateExporter.exclude_output_prompt = True
    c.TemplateExporter.exclude_input = True

    exporter = PDFExporter(config=c)
    body, _resources = exporter.from_filename(ipy_fname)
    with open(ipy_fname.with_suffix(".pdf"), "wb") as fl:
        fl.write(body)

    out = ipy_fname.with_suffix(".pdf")
    console.print(f"Saved PDF to '{out}'")
    return out


def upload_memo(fname, title, memo, quiet):  # pragma: no cover
    """Upload as memo to loco.lab.asu.edu."""
    try:
        find_spec("upload_memo")
    except ImportError as e:
        raise ImportError(
            "You need to manually install upload-memo to use this option."
        ) from e

    opts = ["memo", "upload", "-f", str(fname)]
    if title:
        opts.extend(["-t", title])

    if memo:
        opts.extend(["-n", memo])
    if quiet:
        opts.append("-q")

    run(opts)


class AlanModeOpts:
    out = click.option(
        "-o",
        "--out",
        type=click.Path(dir_okay=True, file_okay=False, exists=True),
        default=".",
        help="output directory",
    )
    redo_spectra = click.option("--redo-spectra/--no-spectra", default=None)
    redo_cal = click.option("--redo-cal/--no-cal", default=True)
    fstart = click.option(
        "-fstart",
        type=float,
        default=48.0,
        help="in mhz",
    )
    fstop = click.option(
        "-fstop",
        type=float,
        default=198.0,
        help="in mhz",
    )
    smooth = click.option(
        "-smooth",
        type=int,
        default=8,
    )
    tload = click.option(
        "-tload",
        type=float,
        default=300.0,
        help="guess at the load temp",
    )
    tcal = click.option(
        "-tcal",
        type=float,
        default=1000.0,
        help="guess at the load+noise source temp",
    )
    Lh = click.option("-Lh", "Lh", type=int, default=-1)
    wfstart = click.option("-wfstart", type=float, default=50.0)
    wfstop = click.option("-wfstop", type=float, default=190.0)
    tcold = click.option("-tcold", type=float, default=306.5)
    thot = click.option("-thot", type=float, default=393.22)
    tcab = click.option("-tcab", type=float, default=None)
    cfit = click.option("-cfit", type=int, default=7)
    tstart = click.option("-tstart", type=int, default=0)
    tstop = click.option("-tstop", type=int, default=24)
    delaystart = click.option("-delaystart", type=int, default=0)
    wfit = click.option(
        "-wfit",
        type=int,
        default=7,
    )
    nfit3 = click.option(
        "-nfit3",
        type=int,
        default=10,
    )
    nfit2 = click.option(
        "-nfit2",
        type=int,
        default=27,
    )
    plot = click.option(
        "--plot/--no-plot",
        default=True,
    )
    avg_spectra_path = click.option(
        "--avg-spectra-path",
        type=click.Path(dir_okay=True, file_okay=False, exists=True),
        help=(
            "Path to a file containing averaged spectra in the format output by this "
            "script (or the C code)"
        ),
    )
    modelled_s11_path = click.option(
        "--modelled-s11-path",
        type=click.Path(dir_okay=False, file_okay=True, exists=True),
        help=(
            "path to a file containing modelled S11s in the format output by this "
            "script (or the C code)"
        ),
    )
    inject_lna_s11 = click.option(
        "--inject-lna-s11/--no-inject-lna-s11",
        default=True,
        help="inject LNA s11 form modelled_s11_path (if given)",
    )
    inject_source_s11s = click.option(
        "--inject-source-s11s/--no-inject-source-s11s",
        default=True,
        help="inject source s11s from modelled_s11_path (if given)",
    )
    write_h5 = click.option(
        "--write-h5/--no-h5",
        default=False,
        help="write the final calibrator object to h5 file as well as the specal.txt",
    )

    @classmethod
    def add_opts(cls, fnc, ignore=None):
        """Add all opts to a given function."""
        ignore = ignore or []
        for name, opt in reversed(cls.__dict__.items()):
            if name not in ignore and callable(opt):
                fnc = opt(fnc)
        return fnc


def _average_spectra(
    specfiles: dict[str, list[Path]],
    out: Path,
    redo_spectra: bool,
    avg_spectra_path,
    **kwargs,
):
    spectra = {}
    for load, files in specfiles.items():
        outfile = out / f"sp{load}.txt"
        if (redo_spectra or not outfile.exists()) and not avg_spectra_path:
            if len(files) == 0:
                raise ValueError(f"{load} has no spectrum files!")

            console.print(f"Averaging {load} spectra")
            flstr = " ".join([str(fl.absolute()) for fl in files])
            os.system(f"cat {flstr} > {out}/temp.acq")

            spfreq, n, spectra[load] = acqplot7amoon(acqfile=out / "temp.acq", **kwargs)

            write_spec_txt(spfreq, n, spectra[load], outfile)

        # Always read the spectra back in, because that's what Alan's C-code does.
        # This has the small effect of reducing the precision of the spectra.
        console.print(f"Reading averaged {load} spectra")

        if outfile.exists():
            spec, _ = read_spec_txt(outfile)
        elif avg_spectra_path:
            spec, _ = read_spec_txt(avg_spectra_path)

        spfreq = spec["freq"] * un.MHz
        spectra[load] = spec["spectra"]
    return spfreq, spectra


def _make_plots(out: Path, calobs: CalibrationObservation, plot):
    write_specal(calobs, out / "specal.txt")
    console.print("BEST DELAY: ", calobs.cal_coefficient_models["NW"].delay)

    # Also save the modelled S11s
    console.print("Saving modelled S11s")
    write_modelled_s11s(calobs, out / "s11_modelled.txt")

    console.print("Saving hot-load loss model")
    np.savetxt(
        out / "hot_load_loss.txt",
        np.array([
            calobs.freq.to_value("MHz"),
            calobs.hot_load.loss_model(
                calobs.freq,
                calobs.hot_load.reflections.s11_model(calobs.freq),
            ),
        ]).T,
        header="# freq, hot_load_loss",
    )

    console.print("Saving calibrated temperatures")
    np.savetxt(
        out / "calibrated_temps.txt",
        np.array(
            [
                calobs.freq.to_value("MHz"),
            ]
            + [calobs.calibrate(load) for load in calobs.loads.values()]
        ).T,
        header="# freq, " + ", ".join(calobs.loads),
    )

    console.print("Saving known load temperatures")
    np.savetxt(
        out / "known_load_temps.txt",
        np.array(
            [
                calobs.freq.to_value("MHz"),
            ]
            + [
                calobs.source_thermistor_temps.get(load.load_name, load.temp_ave)
                * np.ones(calobs.freq.size)
                for load in calobs.loads.values()
            ]
        ).T,
        header="# freq, " + ", ".join(calobs.loads),
    )

    if plot:
        # Make plots...
        console.print("Plotting S11 models...")
        ax = calobs.plot_s11_models()
        ax.flatten()[0].figure.savefig(out / "s11_models.png")

        console.print("Plotting raw spectra...")
        fig = calobs.plot_raw_spectra()
        fig.savefig(out / "raw_spectra.png")

        console.print("Plotting calibration coefficients...")
        fig = calobs.plot_coefficients()
        fig.savefig(out / "calibration_coefficients.png")

        console.print("Plotting calibrated temperatures...")
        fig = calobs.plot_calibrated_temps(bins=1)
        fig.savefig(out / "calibrated_temps_rawres.png")

        fig = calobs.plot_calibrated_temps(bins=64)
        fig.savefig(out / "calibrated_temps_smoothed.png")


@main.command()
@click.argument("s11date", type=str)
@click.argument("specyear", type=int)
@click.argument("specday", type=int)
@click.option("--redo-s11/--no-s11", default=None)
@click.option("-res", "--match-resistance", type=float, default=50.0)
@click.option("-ps", "--calkit-delays", type=float, default=33.0, help="in nanoseconds")
@click.option(
    "-loadps",
    "--load-delay",
    type=float,
    default=None,
    help="in nanoseconds. Overrides -ps.",
)
@click.option(
    "-openps",
    "--open-delay",
    type=float,
    default=None,
    help="in nanoseconds. Overrides -ps.",
)
@click.option(
    "-shortps",
    "--short-delay",
    type=float,
    default=None,
    help="in nanoseconds. Overrides -ps.",
)
@click.option(
    "-cablen",
    "--lna-cable-length",
    type=float,
    default=4.26,
    help="in inches",
)
@click.option(
    "-cabloss",
    "--lna-cable-loss",
    type=float,
    default=-1.24,
    help="as percent",
)
@click.option(
    "-cabdiel",
    "--lna-cable-dielectric",
    type=float,
    default=-91.5,
    help="as percent",
)
@click.option(
    "-d",
    "--datadir",
    type=click.Path(dir_okay=True, file_okay=False, exists=True),
    default="/data5/edges/data/EDGES3_data/MRO/",
)
@AlanModeOpts.add_opts
def alancal(
    s11date,
    specyear,
    specday,
    datadir,
    out,
    redo_s11,
    redo_spectra,
    redo_cal,
    match_resistance,
    calkit_delays,
    load_delay,
    open_delay,
    short_delay,
    lna_cable_length,
    lna_cable_loss,
    lna_cable_dielectric,
    fstart,
    fstop,
    smooth,
    tload,
    tcal,
    Lh,  # noqa: N803
    wfstart,
    wfstop,
    tcold,
    thot,
    tcab,
    cfit,
    wfit,
    nfit3,
    nfit2,
    plot,
    avg_spectra_path,
    modelled_s11_path,
    inject_lna_s11,
    inject_source_s11s,
    tstart,
    tstop,
    delaystart,
    write_h5,
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
    specyear
        The year the spectra were taken in, if doing EDGES-3 cal. Otherwise, zero.
    specday
        The day the spectra were taken on, if doing EDGES-3 cal. Otherwise, zero.
    """
    calobs = acal(
        s11date=s11date,
        specyear=specyear,
        specday=specday,
        datadir=datadir,
        out=out,
        redo_s11=redo_s11,
        redo_spectra=redo_spectra,
        redo_cal=redo_cal,
        match_resistance=match_resistance,
        calkit_delays=calkit_delays,
        load_delay=load_delay,
        open_delay=open_delay,
        short_delay=short_delay,
        lna_cable_length=lna_cable_length,
        lna_cable_loss=lna_cable_loss,
        lna_cable_dielectric=lna_cable_dielectric,
        fstart=fstart,
        fstop=fstop,
        smooth=smooth,
        tload=tload,
        tcal=tcal,
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
        plot=plot,
        avg_spectra_path=avg_spectra_path,
        tstart=tstart,
        tstop=tstop,
        delaystart=delaystart,
    )

    loads = ("amb", "hot", "open", "short")
    if modelled_s11_path:
        calobs = _inject_s11s(
            calobs, modelled_s11_path, loads, inject_lna_s11, inject_source_s11s
        )
    else:
        for name, load in calobs.loads.items():
            console.print(f"Using delay={load.reflections.model_delay} for load {name}")

    out = Path(out)
    _make_plots(out, calobs, plot)

    if write_h5:
        h5file = out / "specal.h5"
        console.print(f"Writing calibration results to {h5file}")
        calobs.write(h5file)


@main.command()
@click.option("--s11-path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--ambient-acqs", type=click.Path(exists=True, dir_okay=True), multiple=True
)
@click.option(
    "--hotload-acqs", type=click.Path(exists=True, dir_okay=True), multiple=True
)
@click.option("--open-acqs", type=click.Path(exists=True, dir_okay=True), multiple=True)
@click.option(
    "--short-acqs", type=click.Path(exists=True, dir_okay=True), multiple=True
)
@click.option(
    "--s11s-in-raul-format/--s11s-in-s1p",
    default=True,
    help="set to true if the S11's have been pre-calibrated and are Rauls format.",
)
@click.option(
    "--lna-poly",
    default=-1,
    help=(
        "Set to zero to force the LNA to be smoothed by a polynomial, not Fourier "
        "series, even if it has <16 terms"
    ),
)
@AlanModeOpts.add_opts
def alancal2(
    s11_path,
    ambient_acqs,
    hotload_acqs,
    open_acqs,
    short_acqs,
    out,
    redo_spectra,
    redo_cal,
    fstart,
    fstop,
    smooth,
    tload,
    tcal,
    Lh,  # noqa: N803
    wfstart,
    wfstop,
    tcold,
    thot,
    tcab,
    cfit,
    wfit,
    nfit3,
    nfit2,
    plot,
    avg_spectra_path,
    modelled_s11_path,
    inject_lna_s11,
    inject_source_s11s,
    s11s_in_raul_format,
    lna_poly,
    tstart,
    tstop,
    delaystart,
    write_h5,
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
    specyear
        The year the spectra were taken in, if doing EDGES-3 cal. Otherwise, zero.
    specday
        The day the spectra were taken on, if doing EDGES-3 cal. Otherwise, zero.
    """
    out = Path(out)

    if s11_path is None or not Path(s11_path).exists():
        raise ValueError("s11_path does not exist")
    loads = ("amb", "hot", "open", "short")

    calobs = acal2(
        s11_path=s11_path,
        ambient_acqs=ambient_acqs,
        hotload_acqs=hotload_acqs,
        open_acqs=open_acqs,
        short_acqs=short_acqs,
        out=out,
        redo_spectra=redo_spectra,
        redo_cal=redo_cal,
        fstart=fstart,
        fstop=fstop,
        smooth=smooth,
        tload=tload,
        tcal=tcal,
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
        avg_spectra_path=avg_spectra_path,
        s11s_in_raul_format=s11s_in_raul_format,
        lna_poly=lna_poly,
        tstart=tstart,
        tstop=tstop,
        delaystart=delaystart,
    )

    if modelled_s11_path:
        calobs = _inject_s11s(
            calobs, modelled_s11_path, loads, inject_lna_s11, inject_source_s11s
        )
    else:
        for name, load in calobs.loads.items():
            console.print(f"Using delay={load.reflections.model_delay} for load {name}")
        console.print(f"Using delay={calobs.receiver.model_delay} for Receiver")

    _make_plots(out, calobs, plot)

    if write_h5:
        h5file = out / "specal.h5"
        console.print(f"Writing calibration results to {h5file}")
        calobs.write(h5file)


def _inject_s11s(
    calobs, modelled_s11_path, loads, inject_lna_s11: bool, inject_source_s11s: bool
):
    _alans11m = np.genfromtxt(
        modelled_s11_path,
        comments="#",
        names=True,
    )

    alans11m = {}
    for load in [*loads, "lna"]:
        alans11m[load] = _alans11m[f"{load}_real"] + 1j * _alans11m[f"{load}_imag"]

    return calobs.inject(
        lna_s11=alans11m["lna"] if inject_lna_s11 else None,
        source_s11s={
            "ambient": alans11m["amb"],
            "hot_load": alans11m["hot"],
            "short": alans11m["short"],
            "open": alans11m["open"],
        }
        if inject_source_s11s
        else None,
    )
