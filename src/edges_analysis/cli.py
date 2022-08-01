"""CLI routines for edges-analysis."""
from __future__ import annotations
import glob
import logging
from pathlib import Path
import time
import os

import click
import h5py
import yaml
from edges_io import io
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

import psutil
import astropy
from pathos.multiprocessing import ProcessPool as Pool
import tqdm
from .gsdata import GSData, GSDATA_PROCESSORS
from functools import partial
from .aux_data import WeatherError
import inspect

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
        return sorted([fl for fl in pth.glob("*") if filt(fl)])
    else:
        return sorted([Path(fl) for fl in glob.glob(str(pth)) if filt(Path(fl))])

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
@click.option("-j", "--nthreads", default=1, help="How many threads to use.")
def process(
    workflow, path, outdir, clobber, verbosity
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
        Panel(f"edges-analysis [blue]processing[/]", box=box.DOUBLE_EDGE),
        style="bold",
        justify="center",
    )

    console.print(Rule("Setting Up"))

    with open(workflow) as fl:
        workflow = yaml.load(fl, Loader=astropy.io.misc.yaml.AstropyLoader)

    steps = workflow.pop("steps")

    console.print(Rule("Running Workflow"))

    if steps[0]['function'] == "convert":
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
    if steps[0]['function'] == "convert":
        step0 = steps.pop(0)
        data = [GSData.from_file(f, **step0.get('params', {})) for f in input_files]
    else:
        data = [GSData.from_file(f) for f in input_files]

    for istep, step in enumerate(steps):
        fncname = step['function']
        params = step.get('params', {})

        fnc = GSDATA_PROCESSORS.get(fncname.lower(), None)

        if fnc is None:
            raise ValueError(f"Unknown function: {fncname}. Available: {GSDATA_PROCESSORS.keys()}")

        console.print(f"[bold] {istep:>02}. {fncname} [/]")
        if params:
            console.print()
            tab = Table(title="Settings", show_header=False)
            tab.add_column()
            tab.add_column()
            for k, v in params.items():
                tab.add_row(k, str(v))
            console.print(tab)
            console.print()

        data = perform_step_on_object(data, fnc, params, clobber, step, outdir)

def perform_step_on_object(data: GSData, fnc: Callable, params: dict, clobber: bool, step: dict, nthreads: int, outdir: str) -> GSData:
    """Perform a workflow step on a GSData object.

    Parameters
    ----------
    data : GSData
        The data to process.
    step : dict
        The step to perform.
    """
    def write_data(data: GSData, step: dict):
        if 'write' not in step:
            return

        fname = step['write']

        # Now, use templating to create the actual filename
        fname = fname.format(
            yearday = data.get_initial_yearday(),
            prev_stem = data.filename.stem,
            prev_dir = data.filename.parent,
            fncname = fnc.__name__, 
            **params,
        )

        if not fname.is_absolute():
            fname = Path(outdir) / fname

        if fname.exists():
            if clobber:
                fname.unlink()
            else:
                raise FileExistsError(
                    f"File {fname} exists. Use --clobber to overwrite!"
                )

        data.write(fname)

    if fnc.gatherer:
        data = fnc(data, **params)
    else:
        this_fnc  = partial(fnc, **params)
        
        
    def run_process_with_memory_checks(data):
        pr = psutil.Process()

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
            logger.warning(f"All data flagged for {data.filename}")
            return

        return data

    mp = Pool(nthreads).map if nthreads > 1 else map
    data = list(
        tqdm.tqdm(
            mp(run_process_with_memory_checks, data), total=len(data), unit='files'
        )
    )

    # Some of the data is now potentially None, because it was all flagged or something
    return [d for d in data if d is not None]  



@main.command()
def avail():
    """List all available GSData processing commands."""
    for command, fnc in GSDATA_PROCESSORS.items():
        console.print(f"[bold]{command}[/]")
        
        args = inspect.signature(fnc)
        for pname, p in args.parameters.items():
            if p.annotation == GSData or pname=='data':
                continue
            if p.annotation:
                console.print(f'    {pname}: [dim]{p.annotation}[/]')
            else:
                console.print(f'    {pname}')
