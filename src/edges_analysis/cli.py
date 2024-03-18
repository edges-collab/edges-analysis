"""CLI routines for edges-analysis."""

from __future__ import annotations

import functools
import glob
import inspect
import logging
import operator
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

import click
import h5py
import psutil
import tqdm
import yaml
from edges_io import io
from pathos.multiprocessing import ProcessPool as Pool
from pygsdata import GSDATA_PROCESSORS, GSData
from read_acq.read_acq import ACQError
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from . import _workflow as wf
from .aux_data import WeatherError

console = Console()

main = click.Group()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            console=console,
            show_time=False,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            tracebacks_width=console.width,
        )
    ],
)

logger = logging.getLogger(__name__)


def _get_files(pth: Path, filt=h5py.is_hdf5) -> list[Path]:
    if pth.is_dir():
        return sorted({fl for fl in pth.glob("*") if filt(fl)})
    else:
        return sorted({Path(fl) for fl in glob.glob(str(pth)) if filt(Path(fl))})


def _file_filter(pth: Path):
    return pth.suffix[1:] in [*io.Spectrum.supported_formats, "gsh5"]


@main.command()
@click.argument("workflow", type=click.Path(dir_okay=False, exists=True))
@click.option(
    "-i",
    "--path",
    type=click.Path(dir_okay=True),
    multiple=True,
    help="""The path(s) to input files. Multiple specifications of ``-i`` can be
    included. Each input path may have glob-style wildcards, eg. ``/path/to/file.*``.
    If the path is a directory, all HDF5/ACQ files in the directory will be used. You
    may prefix the path with a colon to indicate the "standard" location (given by
    ``config['paths']``), e.g. ``-i :big-calibration/``.
    """,
)
@click.option(
    "-o",
    "--outdir",
    default=".",
    type=click.Path(dir_okay=True, file_okay=False),
    help="""The directory into which to save the outputs. Relative paths in the workflow
    are deemed relative to this directory.
    """,
)
@click.option(
    "-v", "--verbosity", default="info", help="level of verbosity of the logging"
)
@click.option("-j", "--nthreads", default=1, help="How many threads to use.")
@click.option(
    "--mem-check/--no-mem-check", default=True, help="Whether to perform a memory check"
)
@click.option(
    "-e/-E",
    "--exit-on-inconsistent/--ignore-inconsistent",
    default=False,
    help=(
        "Whether to immediately exit if any *complete* step is inconsistent with the "
        "progressfile."
    ),
)
@click.option(
    "-r/-a",
    "--restart/--append",
    default=False,
    help=(
        "whether any new input paths should be appended, or if everything should be "
        "restarted with just those files."
    ),
)
@click.option(
    "--stop",
    default=None,
    type=str,
    help="The name of the step at which to stop the workflow.",
)
@click.option(
    "--start",
    default=None,
    type=str,
    help="The name of the step at which to start the workflow.",
)
def process(
    workflow,
    path,
    outdir,
    verbosity,
    nthreads,
    mem_check,
    exit_on_inconsistent,
    restart,
    stop,
    start,
):
    """Process a dataset to the STEP level of averaging/filtering using SETTINGS.

    WORKFLOW
        is a YAML file. Containing a "steps" parameter which should be a list of
        steps to execute.
    """
    logging.getLogger("edges_analysis").setLevel(verbosity.upper())
    logging.getLogger("edges_io").setLevel(verbosity.upper())
    logging.getLogger("edges_cal").setLevel(verbosity.upper())

    console.print(
        Panel("edges-analysis [blue]processing[/]", box=box.DOUBLE_EDGE),
        style="bold",
        justify="center",
    )

    console.print(Rule("Setting Up"))

    steps = wf.Workflow.read(workflow)
    if start and start not in steps:
        raise ValueError(f"The --start option needs to exist! Got {start}")

    outdir = Path(outdir)
    if not outdir.exists():
        outdir.mkdir(parents=True)

    progressfile = outdir / "progressfile.yaml"

    # Get input files (if any)
    path = [Path(p) for p in path]
    input_files = sorted(
        set(
            functools.reduce(
                operator.iadd, (_get_files(p, filt=_file_filter) for p in path), []
            )
        )
    )

    # First, write out a progress file if it doesn't exist.
    if not progressfile.exists() or restart:
        if not input_files:
            raise ValueError(
                "The first time a workflow is run, you need to provide "
                "input files via -i or --path"
            )
        progress = wf.ProgressFile.create(progressfile, steps, input_files)
    else:
        progress = wf.ProgressFile.read(progressfile)
        progress.harmonize_with_workflow(steps, exit_on_inconsistent, start)

        if input_files:
            progress.add_inputs(input_files)

    if stop and stop not in steps:
        raise ValueError(
            f"Your stopping step '{stop}' does not exist! "
            f"Available: {list(steps.keys())}"
        )

    console.print("[bold]Input Files:")
    input_files = progress["convert"].get_all_inputs()
    for fl in input_files:
        console.print(f"   {fl}")
    console.print()

    data: list[GSData] = []
    for istep, step in enumerate(steps):
        stepname = step.name
        fncname = step.function
        params = step.params
        console.print(
            f"[bold underline] {istep:>02}. {fncname.upper()} ({stepname})[/]"
        )
        # This finds only the files that need to be processed at this step.
        # Files that have already been processed through this step won't be returned.
        files = progress.get_files_to_read_for_step(stepname)

        # We need to remove any files that we have already read in.
        current_files = [d.filename.absolute() for d in data if d.filename]
        files = [f for f in files if f not in current_files]

        if len(data + files) > 0 and step.params:
            # Print this out first before adding data, because on the convert step, the
            # from_file() method IS the step, and we want to print the settings before
            # we perform the step itself.
            console.print()
            tab = Table(title="Settings", show_header=False)
            tab.add_column()
            tab.add_column()
            for k, v in params.items():
                tab.add_row(k, str(v))
            console.print(tab)
            console.print()

        # Add any files that were output by previous runs that need to be loaded now.
        if stepname != "convert":
            data += [GSData.from_file(f) for f in files]
        else:
            # For the convert step, the from_file method IS the step, so we need to
            # pass the parameters, if any.
            data = []
            for fl in files:
                try:
                    data.append(GSData.from_file(fl, **step.params))
                except ACQError as e:  # noqa: PERF203
                    logger.warning(f"Could not read {fl}: {e}")
                    progress.remove_inputs([fl])

        if not data:
            console.print(f"{stepname} does not need to run on any files.")

        else:
            if step.name == "convert" and step.write:
                # The input files that we just loaded
                inp = [str(d.filename.absolute()) for d in data]

                # Now the data objects will have the new filename.
                data = [write_data(d, step, outdir=outdir) for d in data]

                # Update the progress file. Each input file gets one output file.
                progress.update_step(
                    "convert",
                    [([x], [d.filename.absolute()]) for x, d in zip(inp, data)],
                )
            elif step.name != "convert":
                data = perform_step_on_object(
                    data,
                    progress,
                    step,
                    nthreads,
                    mem_check,
                )

        console.print()

        if stepname == stop:
            break


@main.command()
@click.argument("workflow", type=click.Path(dir_okay=False, exists=True))
@click.argument("forked", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.option(
    "-o",
    "--outdir",
    default=".",
    type=click.Path(dir_okay=True, file_okay=False),
    help="""The directory into which to save the outputs. Relative paths in the workflow
    are deemed relative to this directory.
    """,
)
def fork(workflow, forked, outdir):
    """Fork the workflow."""
    outdir = Path(outdir).absolute()
    forked = Path(forked).absolute() / "progressfile.yaml"

    # We copy the whole file structure. It's tempting to use symlinks to save space
    # and time, BUT it's possible the new workflow will do filters different to the
    # previous pipeline, which would update files in-place, and therefore follow the
    # symlinks, changing the original files in the other working directory. This would
    # be a mess.
    # TODO: it may be useful in the future to catch this case in the add_filter function
    # i.e. just check if the file is a symlink, and make a new file if so.
    shutil.copytree(forked.parent, outdir)
    newprogress = wf.ProgressFile.read(outdir / "progressfile.yaml")
    workflow = Path(workflow).absolute()
    steps = wf.Workflow.read(workflow)
    newprogress.harmonize_with_workflow(steps, error=False)

    console.print("Done. Please run the rest of the processing as normal.")


def write_data(data: GSData, step: dict, outdir=None):
    """Write data to disk at a particular step."""
    fname = step.get_output_path(outdir, data.filename)

    if fname is None:
        return data

    if not fname.parent.exists():
        fname.parent.mkdir(parents=True)

    if fname.exists():
        fname.unlink()

    logger.info(f"Writing {fname}")
    return data.write_gsh5(fname)


def interpolate_step_params(params: dict, data: GSData) -> dict:
    """String-interpolation for step parameters, from attributes of the data."""
    interpolators = {
        "prev_stem": data.filename.stem,
        "prev_dir": data.filename.parent,
        "fname": data.filename,
        "name": data.name,
    }
    try:
        yearday = data.get_initial_yearday()
        year = int(yearday.split(":")[0])
        day = int(yearday.split(":")[1])
        interpolators["year"] = year
        interpolators["day"] = day

    except ValueError:
        pass

    out = {}
    for k, v in params.items():
        if isinstance(v, str):
            try:
                out[k] = yaml.safe_load(v.format(**interpolators))
            except yaml.parser.ParserError:
                # Sometimes the string is a string and needed to be quoted.
                out[k] = yaml.safe_load('"' + v.format(**interpolators) + '"')

        else:
            out[k] = v

    return out


def perform_step_on_object(
    data: GSData,
    progress: wf.ProgressFile,
    step: wf.WorkflowStep,
    nthreads: int,
    mem_check: bool,
) -> GSData:
    """Perform a workflow step on a GSData object.

    Parameters
    ----------
    data : GSData
        The data to process.
    step : dict
        The step to perform.
    """
    oldfiles = [d.filename.absolute() for d in data]

    if step.kind == "gather":
        data = step(*data)
        if isinstance(data, GSData):
            data = [data]

        out = [write_data(d, step, outdir=progress.path.parent) for d in data]

        if step.write:
            progress.update_step(
                step.name, filemap=[(oldfiles, [d.absolute() for d in out])]
            )
        return out

    if step.kind == "filter":
        params = {
            **step.params,
            "flag_id": step.name,
            "write": False if step.write is not None else None,
        }
    else:
        params = step.params

    def run_process_with_memory_checks(data: GSData):
        if data.complete_flags.all():
            return

        if step.kind == "filter" and step.name in data.flags:
            logger.warning(f"Overwriting existing flags for filter '{step.name}'")
            data = data.remove_flags(step.name)

        pr = psutil.Process()

        if mem_check:
            paused = False
            if psutil.virtual_memory().available < 4 * 1024**3:
                logger.warning(
                    "Available Memory < 4GB, waiting for resources on "
                    f"pid={os.getpid()}. Cancel and restart with fewer threads if this"
                    "thread appears to be frozen"
                )
                paused = True

            while psutil.virtual_memory().available < 4 * 1024**3:
                time.sleep(2)

            if paused:
                logger.warning(f"Resuming processing on pid={os.getpid()}")

        logger.debug(f"Initial memory: {pr.memory_info().rss / 1024**2} MB")

        interp_params = interpolate_step_params(params, data)
        try:
            data = step._function(data, **interp_params)
        except WeatherError as e:
            logger.warning(str(e))
            return

        if data.complete_flags.all():
            return

        return write_data(data, step, outdir=progress.path.parent)

    mp = Pool(nthreads).map if nthreads > 1 else map

    newdata = list(
        tqdm.tqdm(
            mp(run_process_with_memory_checks, data), total=len(data), unit="files"
        )
    )

    if step.write or step.kind == "filter":
        # Update the progress file. Each input file gets one output file.
        progress.update_step(
            step.name,
            [
                ([x], [d.filename.absolute()] if d else [])
                for x, d in zip(oldfiles, newdata)
            ],
        )

    return [d for d in newdata if d is not None]


@main.command()
@click.option(
    "-k", "--kinds", default=None, help="Kinds of data to process", multiple=True
)
def avail(kinds):
    """List all available GSData processing commands."""
    bykind = defaultdict(dict)

    for command, fnc in GSDATA_PROCESSORS.items():
        bykind[fnc.kind][command] = fnc

    for kind, commands in bykind.items():
        if kinds and kind not in kinds:
            continue

        console.print(f"[bold underline]{kind.upper()}[/]")
        for command, fnc in commands.items():
            console.print(f"[bold]{command}[/]")

            args = inspect.signature(fnc)

            for pname, p in args.parameters.items():
                if p.annotation == GSData or pname == "data" or pname == "self":
                    continue

                if p.annotation and p.annotation is not inspect._empty:
                    console.print(f"    {pname}: [dim]{p.annotation}[/]")
                else:
                    console.print(f"    {pname}")

        console.print()
