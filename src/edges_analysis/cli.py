"""CLI routines for edges-analysis."""
from __future__ import annotations

import click
import glob
import h5py
import inspect
import logging
import os
import psutil
import shutil
import time
import tqdm
from collections import defaultdict
from edges_io import io
from functools import partial
from pathlib import Path
from pathos.multiprocessing import ProcessPool as Pool
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from typing import Callable

from . import _workflow as wf
from . import const
from .aux_data import WeatherError
from .gsdata import GSDATA_PROCESSORS, GSData

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
    return pth.suffix[1:] in io.Spectrum.supported_formats


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
    input_files = sorted(set(sum((_get_files(p, filt=_file_filter) for p in path), [])))

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

        if input_files:
            # TODO: I think this is wrong!
            # Here we're appending the input files.
            progress.update_step("convert", [([], [x]) for x in input_files])

    progress.harmonize_with_workflow(steps, exit_on_inconsistent, start)

    if stop and stop not in steps:
        raise ValueError(
            f"Your stopping step '{stop}' does not exist! "
            f"Available: {list(steps.keys())}"
        )

    console.print("[bold]Input Files:")
    input_files = progress["convert"].get_all_outputs()  # WHY OUTPUTS???
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

        files = progress.get_files_to_read_for_step(stepname)

        if data:
            if params:
                console.print()
                tab = Table(title="Settings", show_header=False)
                tab.add_column()
                tab.add_column()
                for k, v in params.items():
                    tab.add_row(k, str(v))
                console.print(tab)
                console.print()

            data = perform_step_on_object(
                data,
                step,
                params,
                step,
                nthreads,
                outdir,
                name=stepname,
                mem_check=mem_check,
            )
        else:
            if stepname == "convert":
                telescope_loc = params.pop(
                    "telescope_location", "alan_edges"
                )  # "edges")
                telescope_loc = const.KNOWN_LOCATIONS[telescope_loc]

                data = [
                    GSData.from_file(f, telescope_location=telescope_loc, **params)
                    for f in files
                ]
                if not data:
                    console.print(f"{stepname} does not need to run on any files.")

                if "write" in step:
                    data = [
                        write_data(d, step, GSData.from_file, outdir, **params)
                        for d in data
                    ]

                oldfiles = [str(d.filename.absolute()) for d in data]

                # Update the progress file. Each input file gets one output file.
                progress.update_step(
                    stepname,
                    [
                        ([x], [str(d.filename.absolute())] if d else [])
                        for x, d in zip(oldfiles, data)
                    ],
                )

                continue

            console.print(f"{stepname} does not need to run on any files.")

        # Add any files that were output by previous runs that need to be loaded now.
        data += [GSData.from_file(f) for f in files]

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


def write_data(data: GSData, step: dict, fnc: callable, outdir=None, **params):
    """Write data to disk at a particular step."""
    if "write" not in step:
        return data

    fname = step["write"]

    # Now, use templating to create the actual filename
    fname = fname.format(
        prev_stem=data.filename.stem,
        prev_dir=data.filename.parent,
        fncname=fnc.__name__,
        **params,
    )

    fname = Path(fname)

    if not fname.is_absolute():
        fname = Path(outdir) / fname

    if not fname.parent.exists():
        fname.parent.mkdir(parents=True)

    if fname.exists():
        fname.unlink()

    logger.info(f"Writing {fname}")
    return data.write_gsh5(fname)


def perform_step_on_object(
    data: GSData,
    fnc: Callable,
    params: dict,
    step: dict,
    nthreads: int,
    outdir: str,
    name: str,
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
    oldfiles = [str(d.filename.absolute()) for d in data]
    progressfile = outdir / "progressfile.yaml"

    if fnc.kind == "gather":
        data = fnc(*data, **params)
        data = write_data(data, step, fnc, outdir, **params)
        if "write" in step:
            update_progressfile(
                progressfile, name, [(oldfiles, [str(data.filename.absolute())])]
            )
        return [data]

    if fnc.kind == "filter":
        params = {**params, "flag_id": name, "write": True}

    this_fnc = partial(fnc, **params)

    def run_process_with_memory_checks(data: GSData):
        if data.complete_flags.all():
            return

        if fnc.kind == "filter" and name in data.flags:
            logger.warning(f"Overwriting existing flags for filter '{name}'")
            data = data.remove_flags(name)

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
        try:
            data = this_fnc(data)
        except WeatherError as e:
            logger.warning(str(e))
            return

        if data.complete_flags.all():
            return

        return write_data(data, step, fnc, outdir, **params)

    mp = Pool(nthreads).map if nthreads > 1 else map

    newdata = list(
        tqdm.tqdm(
            mp(run_process_with_memory_checks, data), total=len(data), unit="files"
        )
    )

    if "write" in step or fnc.kind == "filter":
        # Update the progress file. Each input file gets one output file.
        update_progressfile(
            progressfile,
            name,
            [
                ([x], [str(d.filename.absolute())] if d else [])
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
