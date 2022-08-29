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
import yaml
from collections import defaultdict
from edges_io import io
from functools import partial
from jinja2 import Template
from pathlib import Path
from pathos.multiprocessing import ProcessPool as Pool
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from typing import Callable

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


def read_progress(fl: Path) -> list[dict]:
    """Read the .progress file."""
    with open(fl, "r") as openfile:
        progress = yaml.load(openfile, Loader=yaml.FullLoader)

    return progress


def write_progressfile(outdir: Path, steps: tuple[dict], complete: dict | None = None):
    """Write out the progress file."""
    complete = complete or {}
    progress = []

    for step in steps:
        if step["name"] in complete:
            progress.append({**step, **{"files": complete[step["name"]]}})
        elif "files" in step:
            progress.append(step)
        else:
            progress.append({**step, **{"files": []}})

    with open(outdir / "progressfile.yaml", "w") as fl:
        yaml.dump(progress, fl)


def update_progressfile(outdir: Path, name: str, files: list[Path]):
    """Update the progress file."""
    prg = read_progress(outdir / "progressfile.yaml")

    names = [p["name"] for p in prg]

    if name not in names:
        raise ValueError(f"Progress file has no step called '{name}'")

    write_progressfile(outdir, prg, {name: files})


class WorkflowProgressError(RuntimeError):
    """Exception raised when the workflow and progress files are discrepant."""

    pass


def check_workflow_compatibility(
    steps: tuple[dict], progressfile: Path, error: bool = True
) -> str:
    """Check the compatibility of the current steps with the progressfile."""
    # progress should be a list very similar to "steps" except with a few extra
    # keys (like "files")
    progress = read_progress(progressfile)
    # where are we up to?
    first_incomplete = None
    look_for_it = True
    for p in progress:
        if not p["files"] and look_for_it:
            first_incomplete = p
            look_for_it = False
        elif p["files"]:
            all_exist = all(Path(fl).exists() for fl in p["files"])
            if look_for_it and not all_exist:
                first_incomplete = p
                look_for_it = False
            elif all_exist:
                look_for_it = True
                first_incomplete = None

    # We need to ensure that all steps before the first incomplete step are the same.
    # Otherwise, we need to backtrack to that step.
    for (s, p) in zip(steps, progress):
        name = s.get("name", s.get("function"))

        # If everything is consistent up to the first incomplete step, we're good to
        # go from there.
        if p == first_incomplete:
            return name

        ps = {k: v for k, v in p.items() if k not in ["files"]}

        # Otherwise, if there's an inconsistency, we have to go back to the step that
        # is inconsistent -- if it has changed, then we have to re-run it.
        if ps != s:
            msg = (
                f"Found inconsistency at step {name} betweeen workflow and progress "
                f"file.\nStep Definition:\n{yaml.dump(s)}\n"
                f"Progress Definition:\n{yaml.dump(ps)}"
            )
            if error:
                raise WorkflowProgressError(msg)
            else:
                logger.warning(msg)

            return name

        fls = [Path(fl) for fl in p["files"]]
        non_existent = [
            str(fl.relative_to(progressfile.absolute().parent))
            for fl in fls
            if not fl.exists()
        ]
        if non_existent:
            fls = "\n\t".join(non_existent)
            msg = f"Non-existent output files for step '{name}':\n\t{fls}"
            if error:
                raise WorkflowProgressError(msg)
            else:
                logger.warning(msg)

            return name
    else:
        return None


def get_steps_from_workflow(workflow: Path) -> tuple[dict]:
    """Read the steps from a workflow."""
    with open(workflow) as fl:
        workflowd = yaml.load(fl, Loader=yaml.FullLoader)

    global_params = workflowd.pop("globals", {})

    with open(workflow) as fl:
        txt = Template(fl.read())
        txt = txt.render(globals=global_params)
        workflow = yaml.load(txt, Loader=yaml.FullLoader)

    steps = workflow.pop("steps")

    for step in steps:
        if "name" not in step:
            step["name"] = step["function"]

    return steps


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
def process(
    workflow,
    path,
    outdir,
    clobber,
    verbosity,
    start,
    nthreads,
    mem_check,
    exit_on_inconsistent,
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

    steps = get_steps_from_workflow(workflow)
    all_names = [step["name"].lower() for step in steps]
    for name in all_names:
        if all_names.count(name) > 1:
            raise ValueError(
                f"Duplicate step name {name}. "
                "Please give one of the steps an explicit 'name'."
            )

    # First, write out a progress file if it doesn't exist.
    outdir = Path(outdir)

    if not (outdir / "progressfile.yaml").exists():
        write_progressfile(outdir, steps)

    # Check for inconsistencies between progressfile and workflow. Also get the first
    # step that is incomplete/inconsistent.
    start = check_workflow_compatibility(
        steps, outdir / "progressfile.yaml", error=exit_on_inconsistent
    )

    if start is None:
        console.print("[green bold] Your workflow is already finished![/]")
        return
    else:
        console.print(f"Starting at '{start}'.")

    # Now, we need to update the existing progressfile
    progress = read_progress(outdir / "progressfile.yaml")
    completed = {}
    for s in progress:
        if s["name"] == start:
            break
        completed[s["name"]] = s["files"]

    write_progressfile(outdir, steps, complete=completed)
    progress = read_progress(outdir / "progressfile.yaml")

    # Get the starting step (not just name)
    for start_step in steps:
        if start_step["name"] == start:
            break

    if start == "convert":
        # Using convert means we're getting the files raw from somewhere else.
        # Otherwise, we're referencing files inside the working directory.
        path = [Path(p) for p in path]

        def file_filter(pth: Path):
            return pth.suffix[1:] in io.Spectrum.supported_formats

    else:
        for p in progress:
            if p["name"] == start:
                break

            path = [Path(pp) for pp in p["files"]]

        file_filter = h5py.is_hdf5

    # Get input files
    input_files = sum((_get_files(p, filt=file_filter) for p in path), [])

    if not input_files:
        logger.error("No input files found!")
        return

    if start == "convert":
        console.print("[bold]Input Files:")
        for fl in input_files:
            console.print(f"   {fl}")
        console.print()

    # First run either convert or read function
    # TODO: this is bad memory-wise (reads everything in up-front, should be chunked)
    if start == "convert":
        step0 = steps.pop(0)
        params = step0.get("params", {})
        params.update({"telescope_location": const.edges_location})
        data = [GSData.from_file(f, **params) for f in input_files]
    else:
        data = [GSData.from_file(f) for f in input_files]

    console.print(Rule("Running Workflow"))

    # Remove all steps before the one we care about.
    for i, step in enumerate(steps):
        if step["name"] == start:
            break

    for istep, (step, stepname) in enumerate(zip(steps[i:], all_names[i:]), start=i):
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
            data,
            fnc,
            params,
            clobber,
            step,
            nthreads,
            outdir,
            name=stepname,
            mem_check=mem_check,
        )
        console.print()


@main.command()
@click.argument("workflow", type=click.Path(dir_okay=False, exists=True))
@click.argument("forked", type=click.Path(dir_okay=False, exists=True))
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
    "-s/-c",
    "--symlink/--copy",
    help="Whether to symlink the already-computed files.",
    default=True,
)
def fork(workflow, forked, outdir, symlink):
    """Fork the workflow."""
    outdir = Path(outdir).absolute()
    forked = Path(forked).absolute()
    workflow = Path(workflow).absolute()
    steps = get_steps_from_workflow(workflow)
    first = check_workflow_compatibility(steps, forked, error=False)

    console.print(
        f"Forking workflow at {forked.parent} to new workflow {workflow.name}."
    )
    console.print(f"First deviating or unperformed step: {first}")

    # Now, for all the files in steps up to the first, copy them over:
    progress = read_progress(forked)
    for prg in progress:
        if prg["name"] == first:
            break

        fls = [Path(fl) for fl in prg["files"]]

        for fl in fls:
            outfile = outdir / fl.relative_to(forked.parent)

            if not outfile.parent.exists():
                outfile.parent.mkdir(parents=True)

            if symlink:
                outfile.symlink_to(fl)
            else:
                shutil.copy(fl, outfile)

    # Now copy over the progressfile, but update the locations.
    shutil.copy(forked, outdir / "progressfile.yaml")
    with open(outdir / "progressfile.yaml", "r") as fl:
        txt = fl.read()

    txt = txt.replace(str(forked.parent), str(outdir))

    with open(outdir / "progressfile.yaml", "w") as fl:
        fl.write(txt)

    console.print("Done. Please run the rest of the processing as normal.")


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
        data = write_data(data, step)
        return [data]

    if fnc.kind == "filter":
        params = {**params, "flag_id": name, "write": True}

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

    if "write" in step:
        # Update the progress file
        update_progressfile(
            outdir, step["name"], [str(d.filename.absolute()) for d in newdata]
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
