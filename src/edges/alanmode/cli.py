"""CLI functions for edges-cal."""

from pathlib import Path
from typing import Annotated

import numpy as np
from attr import dataclass
from cyclopts import App, Parameter
from rich.console import Console

from edges.cal.calibrator import Calibrator

from ..cal import plots
from ..cal.calobs import CalibrationObservation
from ..cli import app
from . import (
    alancal as acal,
)
from . import (
    write_modelled_s11s,
    write_s11_csv,
    write_specal,
)
from .alanmode import (
    ACQPlot7aMoonParams,
    Edges2CalobsParams,
    Edges3CalobsParams,
    EdgesScriptParams,
)

console = Console()

app.command(amode := App(name="alanmode", help="Run scripts similar to legacy C code"))


def _make_plots(
    out: Path,
    calobs: CalibrationObservation,
    calibrator: Calibrator,
    load_s11_mdl,
    rcv_s11_mdl,
    hot_loss_model,
    plot: bool,
    t_load: float = 300.0,
    t_load_ns: float = 1000.0,
):
    write_specal(calibrator, out / "specal.txt", t_load=t_load, t_load_ns=t_load_ns)

    # Also save the modelled S11s
    console.print("Saving modelled S11s")
    write_modelled_s11s(calobs, out / "s11_modelled.txt", hot_loss_model)

    console.print("Saving hot-load loss model")
    np.savetxt(
        out / "hot_load_loss.txt",
        np.array([
            calobs.freqs.to_value("MHz"),
            calobs.hot_load.loss,
        ]).T,
        header="# freq, hot_load_loss",
    )

    console.print("Saving calibrated temperatures")
    np.savetxt(
        out / "calibrated_temps.txt",
        np.array(
            [
                calobs.freqs.to_value("MHz"),
            ]
            + [calibrator.calibrate_load(load) for load in calobs.loads.values()]
        ).T,
        header="# freq, " + ", ".join(calobs.loads),
    )

    console.print("Saving known load temperatures")
    np.savetxt(
        out / "known_load_temps.txt",
        np.array(
            [
                calobs.freqs.to_value("MHz"),
            ]
            + [
                calobs.source_thermistor_temps.get(load.load_name, load.temp_ave)
                * np.ones(calobs.freqs.size)
                for load in calobs.loads.values()
            ]
        ).T,
        header="# freq, " + ", ".join(calobs.loads),
    )

    if plot:
        # Make plots...
        console.print("Plotting S11 models...")
        ax = plots.plot_s11_models(calobs)
        ax.flatten()[0].figure.savefig(out / "s11_models.png")

        console.print("Plotting raw spectra...")
        fig = plots.plot_raw_spectra(calobs)
        fig.savefig(out / "raw_spectra.png")

        console.print("Plotting calibration coefficients...")
        fig = plots.plot_cal_coefficients(calibrator)
        fig.savefig(out / "calibration_coefficients.png")

        console.print("Plotting calibrated temperatures...")
        fig = plots.plot_calibrated_temps(calobs, calibrator)
        fig.savefig(out / "calibrated_temps_rawres.png")

        fig = plots.plot_calibrated_temps(calobs, calibrator, bins=64)
        fig.savefig(out / "calibrated_temps_smoothed.png")


@dataclass
class AlanCalOpts:
    """CLI options for the Alan calibration mode."""

    avg: ACQPlot7aMoonParams = ACQPlot7aMoonParams()
    cal: EdgesScriptParams = EdgesScriptParams()

    plot: bool = True
    "Whether to make plots of the various calibration quantities."

    out: Path = Path()
    "The directory into which to write the results."

    redo_spectra: bool = False
    """Whether to re-average the spectra, even if averaged spectrum files are found.
Reading and averaging the sepctra is the most intensive part of the calculation,
so using cached results is often preferable."""

    redo_cal: bool = True
    """Whether to rerun the calibration at all, if calibration solutions are found
in the given --out directory."""


# @amode.command(name='run')
def _alancal(
    data: Edges3CalobsParams | Edges2CalobsParams,
    opts: AlanCalOpts,
):
    """Run a calibration in as close a manner to Alan's code as possible.

    This exists mostly for being able to compare to Alan's memos etc in an easy way. It
    is much less flexible than using the library directly, and is not recommended for
    general use.

    This is supposed to emulate one of Alan's C-shell scripts, usually called "docal",
    and thus it runs a complete calibration, not just a single part. However, you can
    turn off parts of the calibration by setting the appropriate flags to False.
    """
    calobs, calibrator, load_s11_mdl, rcv_s11_mdl, hot_loss_model = acal(
        defparams=data,
        out=opts.out,
        redo_spectra=opts.redo_spectra,
        redo_cal=opts.redo_cal,
        acqparams=opts.avg,
        calparams=opts.cal,
    )

    out = Path(opts.out)

    for name, load in calobs.loads.items():
        # Output Raw S11s
        write_s11_csv(load._raw_s11.freqs, load._raw_s11.s11, out / f"s11{name}.csv")

    # Also write the LNA S11
    write_s11_csv(
        calobs._raw_receiver.freqs, calobs._raw_receiver.s11, out / "s11lna.csv"
    )

    _make_plots(
        out,
        calobs,
        calibrator,
        load_s11_mdl,
        rcv_s11_mdl,
        hot_loss_model,
        opts.plot,
        t_load=opts.avg.tload,
        t_load_ns=opts.avg.tcal,
    )


@amode.command(name="cal-edges2")
def alancal_edges2(
    data: Edges2CalobsParams,
    opts: Annotated[AlanCalOpts, Parameter(name="*")] = AlanCalOpts(),
):
    """Run a calibration for EDGES2 in as close a manner to Alan's code as possible.

    This exists mostly for being able to compare to Alan's memos etc in an easy way. It
    is much less flexible than using the library directly, and is not recommended for
    general use.

    This is supposed to emulate one of Alan's C-shell scripts, usually called "docal",
    and thus it runs a complete calibration, not just a single part. However, you can
    turn off parts of the calibration by setting the appropriate flags to False.
    """
    _alancal(data=data, opts=opts)


@amode.command(name="cal-edges3")
def alancal_edges3(
    data: Edges3CalobsParams,
    opts: Annotated[AlanCalOpts, Parameter(name="*")] = AlanCalOpts(),
):
    """Run a calibration for EDGES3 in as close a manner to Alan's code as possible.

    This exists mostly for being able to compare to Alan's memos etc in an easy way. It
    is much less flexible than using the library directly, and is not recommended for
    general use.

    This is supposed to emulate one of Alan's C-shell scripts, usually called "docal",
    and thus it runs a complete calibration, not just a single part. However, you can
    turn off parts of the calibration by setting the appropriate flags to False.
    """
    _alancal(data=data, opts=opts)
