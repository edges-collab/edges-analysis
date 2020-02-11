import logging
from os.path import join

import click
from edges_cal import cal_coefficients as cc

from . import io
from .logging import logger

main = click.Group()


@main.command()
@click.argument("path", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.option(
    "-f", "--f-low", type=float, default=None, help="minimum frequency to calibrate"
)
@click.option(
    "-F", "--f-high", type=float, default=None, help="maximum frequency to calibrate"
)
@click.option("-n", "--run-num", type=int, default=None, help="run number to read")
@click.option(
    "-p",
    "--ignore_times_percent",
    type=float,
    default=5,
    help="percentage of data at start of files to ignore",
)
@click.option(
    "-r",
    "--resistance-f",
    type=float,
    default=50.0002,
    help="female resistance standard",
)
@click.option(
    "-R", "--resistance-m", type=float, default=50.166, help="male resistance standard"
)
@click.option(
    "-C", "--c-terms", type=int, default=11, help="number of terms to fit for C1 and C2"
)
@click.option(
    "-W",
    "--w-terms",
    type=int,
    default=12,
    help="number of terms to fit for TC, TS and TU",
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
def run(
    path,
    f_low,
    f_high,
    run_num,
    ignore_times_percent,
    resistance_f,
    resistance_m,
    c_terms,
    w_terms,
    out,
    cache_dir,
):
    """
    Calibrate using lab measurements in PATH, and make all relevant plots.
    """
    obs = cc.CalibrationObservation(
        path=path,
        f_low=f_low,
        f_high=f_high,
        run_num=run_num,
        ignore_times_percent=ignore_times_percent,
        resistance_f=resistance_f,
        resistance_m=resistance_m,
        cterms=c_terms,
        wterms=w_terms,
        cache_dir=cache_dir,
    )

    # Plot Calibrator properties
    fig = obs.plot_raw_spectra()
    fig.savefig(join(out, "raw_spectra.png"))

    figs = obs.plot_s11_models()
    for kind, fig in figs.items():
        fig.savefig(join(out, f"{kind}_s11_model.png"))

    fig = obs.plot_calibrated_temps(bins=256)
    fig.savefig(join(out, "calibrated_temps.png"))

    fig = obs.plot_coefficients()
    fig.savefig(join(out, "calibration_coefficients.png"))

    # Calibrate and plot antsim
    antsim = obs.new_load(load_name="AntSim3")
    fig = obs.plot_calibrated_temp(antsim, bins=256)
    fig.savefig(join(out, "antsim_calibrated_temp.png"))

    # Write out data
    obs.write_coefficients()
