"""CLI routines for edges-analysis."""
from __future__ import annotations

import glob
import inspect
import logging
import os
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Callable

import click
import h5py
import psutil
import tqdm
import yaml
from edges_io import io
from jinja2 import Template
from pathos.multiprocessing import ProcessPool as Pool
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

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
        return sorted(fl for fl in pth.glob("*") if filt(fl))
    else:
        return sorted(Path(fl) for fl in glob.glob(str(pth)) if filt(Path(fl)))


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
    "-c/-C",
    "--clobber/--no-clobber",
    help="Whether to overwrite any existing data at the output location",
)
@click.option(
    "-v", "--verbosity", default="info", help="level of verbosity of the logging"
)
@click.option("-s", "--start", default=None, help="Starting step of the workflow")
@click.option("-j", "--nthreads", default=1, help="How many threads to use.")
@click.option("--mem-check/--no-mem-check", default=True, help="Whether to perform a memory check")
def process(workflow, path, outdir, clobber, verbosity, start, nthreads, mem_check):
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

    with open(workflow) as fl:
        workflowd = yaml.load(fl, Loader=yaml.FullLoader)

    global_params = workflowd.pop("globals", {})

    with open(workflow) as fl:
        txt = Template(fl.read())
        txt = txt.render(globals=global_params)
        workflow = yaml.load(txt, Loader=yaml.FullLoader)

    steps = workflow.pop("steps")

    all_names = [step.get("name", step["function"]).lower() for step in steps]
    for name in all_names:
        if all_names.count(name) > 1:
            raise ValueError(
                f"Duplicate step name {name}. "
                "Please give one of the steps an explicit 'name'."
            )

    if start:
        if start.lower() not in all_names:
            logger.error(f"Step {start} not found in workflow")
            return
        start_step = all_names.index(start.lower())
        steps = steps[start_step:]
        all_names = all_names[start_step:]

    if steps[0]["function"] == "convert":

        def file_filter(pth: Path):
            return pth.suffix[1:] in io.Spectrum.supported_formats

    else:
        file_filter = h5py.is_hdf5

    # Get input files
    path = [Path(p) for p in path]
    input_files = sum((_get_files(p, filt=file_filter) for p in path), [])

    if not input_files:
        logger.error("No input files found!")
        return

    console.print("[bold]Input Files:")
    for fl in input_files:
        console.print(f"   {fl}")
    console.print()

    # First run either convert or read function
    # TODO: this is bad memory-wise (reads everything in up-front, should be chunked)
    if steps[0]["function"] == "convert":
        step0 = steps.pop(0)
        params = step0.get("params", {})
        params.update({"telescope_location": const.edges_location})
        data = [GSData.from_file(f, **params) for f in input_files]
    else:
        data = [GSData.from_file(f) for f in input_files]

    console.print(Rule("Running Workflow"))

    for istep, (step, stepname) in enumerate(zip(steps, all_names)):
        fncname = step["function"]
        params = step.get("params", {})

        fnc = GSDATA_PROCESSORS.get(fncname.lower(), None)

        if fnc is None:
            raise ValueError(
                f"Unknown function: {fncname}. Available: {GSDATA_PROCESSORS.keys()}"
            )

        console.print(f"[bold underline] {istep:>02}. {fncname.upper()} [/]")
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
            data, fnc, params, clobber, step, nthreads, outdir, name=stepname, mem_check=mem_check
        )
        console.print()


def perform_step_on_object(
    data: GSData,
    fnc: Callable,
    params: dict,
    clobber: bool,
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

    def write_data(data: GSData, step: dict):
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
            if clobber:
                fname.unlink()
            else:
                raise FileExistsError(
                    f"File {fname} exists. Use --clobber to overwrite!"
                )

        logger.info(f"Writing {fname}")
        return data.write_gsh5(fname)

    if fnc.kind == "gather":
        data = fnc(*data, **params)
        write_data(data, step)
        return [data]

    if fnc.kind == "filter":
        params = {**params, "flag_id": name}

    this_fnc = partial(fnc, **params)

    def run_process_with_memory_checks(data: GSData):
        if data.complete_flags.all():
            return

        if fnc.kind == "filter" and name in data.flags:
            if clobber:
                logger.warning(f"Overwriting existing flags for filter '{name}'")
                data = data.remove_flags(name)
            else:
                raise ValueError(
                    f"Flags for filter '{name}' already exist. "
                    "Use --clobber to overwrite!"
                )

        pr = psutil.Process()

        print("MEM_CHECK: ", mem_check)
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
        except (WeatherError) as e:
            logger.warning(str(e))
            return

        if data.complete_flags.all():
            return

        return write_data(data, step)

    mp = Pool(nthreads).map if nthreads > 1 else map
    newdata = list(
        tqdm.tqdm(
            mp(run_process_with_memory_checks, data), total=len(data), unit="files"
        )
    )

    # Some of the data is now potentially None, because it was all flagged or something
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
