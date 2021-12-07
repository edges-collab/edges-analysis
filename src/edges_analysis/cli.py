"""CLI routines for edges-analysis."""
from __future__ import annotations
import glob
import logging
import sys
from pathlib import Path
import shutil
import time
import os

import click
import h5py
import p_tqdm
import questionary as qs
import yaml
from edges_io import io
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from .analysis import levels, filters
from .config import config
import psutil

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


def _get_settings(settings, **cli_settings):
    with open(settings) as fl:
        settings = yaml.load(fl, Loader=yaml.FullLoader)

    settings.update(cli_settings)

    console.print()
    tab = Table(title="Settings", show_header=False)
    tab.add_column()
    tab.add_column()
    for k, v in settings.items():
        tab.add_row(k, str(v))
    console.print(tab)
    console.print()

    return settings


def _ctx_to_dct(args):
    dct = {}
    j = 0
    while j < len(args):
        arg = args[j]
        if "=" in arg:
            a = arg.split("=")
            dct[a[0].replace("--", "")] = a[-1]
            j += 1
        else:
            dct[arg.replace("--", "")] = args[j + 1]
            j += 2

    for k in dct:
        for tp in (int, float):
            try:
                dct[k] = tp(dct[k])
                break
            except TypeError:
                pass

    return dct


def _get_files(pth: Path, filt=h5py.is_hdf5) -> list[Path]:
    if pth.is_dir():
        return [fl for fl in pth.glob("*") if filt(fl)]
    else:
        return [Path(fl) for fl in glob.glob(str(pth)) if filt(Path(fl))]


def get_output_dir(prefix, label, settings):
    """Get an output directory from given settings."""
    out = Path(prefix) / label

    if out.exists():
        try:
            with open(out / "settings.yaml") as fl:
                existing_settings = yaml.load(fl, Loader=yaml.FullLoader)
        except FileNotFoundError:
            # Empty directory most likely due to an error in a previous run.
            return out

        if existing_settings != settings:
            tab = Table("existing", "proposed", width=console.width)
            tab.add_row(yaml.dump(existing_settings), yaml.dump(settings))
            console.print(tab)

            if qs.confirm(
                f"{out} has existing files with different settings. Remove existing and"
                f"continue?"
            ).ask():
                for fl in out.glob("*"):
                    if fl.is_file():
                        fl.unlink()
            else:
                console.print("Fine. Be that way.")
                sys.exit()

        else:
            console.print("Using existing label with identical settings.")

    return out


def expand_colon(pth: str, band: str = "", raw=True) -> Path:
    """Expand the meaning of : in front of a path pointing to raw field data."""
    if pth[0] != ":":
        return Path(pth)
    elif raw:
        if not band:
            raise ValueError("must provide 'band' in settings to find raw files!")

        return Path(config["paths"]["raw_field_data"]) / "mro" / band / pth[1:]
    else:
        return Path(config["paths"]["field_products"]) / pth[1:]


@main.command(
    context_settings={  # Doing this allows arbitrary options to override config
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
)
@click.argument(
    "step",
    type=click.Choice(
        ["raw", "calibrate", "model", "combine", "day", "bin"], case_sensitive=False
    ),
)
@click.argument("settings", type=click.Path(dir_okay=False, exists=True))
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
    "-l",
    "--label",
    default="",
    help="""A label for the output. This label should be unique to the input settings
    (but may be applied to different input files). If the same label is used for
    different settings, the existing processed data will be removed (after prompting).
    """,
)
@click.option(
    "-m",
    "--message",
    default="",
    help="""A message to save with the data. The message will be saved in a README.txt
    file alongside the output data file(s). It is intended to provide a
    human-understandable "reason" for running the particular analysis with the
    particular settings.
    """,
)
@click.option(
    "-c/-C",
    "--clobber/--no-clobber",
    help="Whether to overwrite any existing data at the output location",
)
@click.option(
    "-o",
    "--output",
    default="",
    help="Name of an output file. Only required for the 'combine' step.",
)
@click.option(
    "-v", "--verbosity", default="info", help="level of verbosity of the logging"
)
@click.option("-j", "--nthreads", default=1, help="How many threads to use.")
@click.pass_context
def process(
    ctx, step, settings, path, label, message, clobber, output, nthreads, verbosity
):
    """Process a dataset to the STEP level of averaging/filtering using SETTINGS.

    STEP
        defines the analysis step as a string. Each of the steps should be applied
        in turn.
    SETTINGS
        is a YAML settings file. The available settings for each step can be seen
        in the respective documentation for the classes "promote" method.

    Each STEP should take one or more ``--input`` files that are the output of a
    previous step. The first step (``raw``) should take raw ``.acq`` or ``.h5``
    spectrum files.

    The output files are placed in a directory inside the input file directory, with a
    name determined by the ``--label``.
    """
    logging.getLogger("edges_analysis").setLevel(verbosity.upper())
    logging.getLogger("edges_io").setLevel(verbosity.upper())
    logging.getLogger("edges_cal").setLevel(verbosity.upper())

    console.print(
        Panel(f"edges-analysis [blue]{step}[/]", box=box.DOUBLE_EDGE),
        style="bold",
        justify="center",
    )

    console.print(Rule("Setting Up"))

    step_cls = {
        "raw": levels.RawData,
        "calibrate": levels.CalibratedData,
        "model": levels.ModelData,
        "combine": levels.CombinedData,
        "day": levels.DayAveragedData,
        "bin": levels.BinnedData,
    }[step]

    cli_settings = _ctx_to_dct(ctx.args)
    settings = _get_settings(settings, **cli_settings)
    label = settings.pop("label", "") or label

    if not label:
        label = qs.text("Provide a short label to identify this run:").ask()

    if step == "raw":

        def file_filter(pth: Path):
            return pth.suffix[1:] in io.Spectrum.supported_formats

    else:
        file_filter = h5py.is_hdf5

    # Get input file(s). If doing initial calibration, get them from raw_field_data
    # otherwise they should be in field_products.
    path = [
        expand_colon(p, band=settings.get("band"), raw=step == "raw").expanduser()
        for p in path
    ]
    input_files = sum((_get_files(p, filt=file_filter) for p in path), [])

    if not input_files:
        logger.error(f"No input files were found! Paths: {path}")
        return
    else:
        console.print("[bold]Input Files:")
        for fl in input_files:
            console.print(f"   {fl}")
        console.print()
    # Check that input files are all homogeneously processed
    if step != "raw" and len({p.parent for p in input_files}) != 1:
        raise ValueError("Your input files do not come from a single processing.")

    input_file_type = levels.get_step_type(input_files[0])
    # Get unique output directory
    if step == "raw":
        output_dir = Path(config["paths"]["field_products"]) / label
    else:
        output_dir = get_output_dir(input_files[0].parent, label, settings)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        f"[bold]Output Directory: [dim]{output_dir}",
    )

    # In most cases, the output filename will be the same as the (sole) input.
    # However, when combining files, we need some extra label.
    if step == "combine":
        if not output:
            output = [
                Path(
                    qs.text("Provide a filename for the output combined file:").ask()
                ).with_suffix(".h5")
            ]
        else:
            output = [Path(output).with_suffix(".h5")]
    elif step != "raw":
        output = [Path(p.name) for p in input_files]
    else:
        output = None

    if step == "bin" and input_file_type in (
        levels.CombinedData,
        levels.CombinedBinnedData,
    ):
        step_cls = levels.CombinedBinnedData

    if output:
        for pth in output:
            if (output_dir / pth).exists():
                if clobber:
                    (output_dir / pth).unlink()
                else:
                    raise FileExistsError(
                        f"File {output_dir/pth} exists. Use --clobber to overwrite!"
                    )

    if message:
        with open(output_dir / "README.txt", "w") as fl:
            fl.write(message)

    with open(output_dir / "settings.yaml", "w") as fl:
        yaml.dump(settings, fl)

    # Actually call the relevant function
    console.print()
    console.print(Rule("Beginning Processing"))
    out_paths = promote(
        input_files, nthreads, output_dir, output, step_cls, settings, clobber
    )
    console.print(Rule("Done Processing"))

    for pth in out_paths:
        with h5py.File(pth, "a") as fl:
            fl.attrs["message"] = message

    console.print()
    if step == "combine":
        console.print(f"[bold]Output File: [blue]{out_paths[0]}")
    elif step in ("raw", "calibrate", "model"):
        console.print(
            f"[bold]All files written to: [dim]{output_dir}",
        )
    else:
        console.print("[bold]Output Files:")
        for fname in output:
            console.print(f"\t[bold]{output_dir}/{fname}")


def promote(
    input_files: list[Path],
    nthreads: int,
    output_dir: Path,
    output_fname: list[Path | None] | None,
    step_cls: type[levels._ReductionStep],
    settings: dict,
    clobber: bool,
) -> list[Path]:
    """Calibrate field data to produce CalibratedData files."""
    if not input_files:
        raise ValueError("No input files!")

    if step_cls._multi_input:
        data = step_cls.promote(prev_step=input_files, **settings)
        data.write(output_dir / output_fname[0])
        return [output_dir / output_fname[0]]
    else:
        output_fname = output_fname or [None] * len(input_files)

        def _pro(fl, fname):
            pr = psutil.Process()

            paused = False
            if psutil.virtual_memory().available < 4 * 1024 ** 3:
                logger.warning(
                    "Available Memory < 4GB, waiting for resources on "
                    f"pid={os.getpid()}. Cancel and restart with fewer threads if this"
                    "thread appears to be frozen"
                )
                paused = True

            while psutil.virtual_memory().available < 4 * 1024 ** 3:
                time.sleep(2)

            if paused:
                logger.warning(f"Resuming processing on pid={os.getpid()}")

            logger.debug(f"Initial memory: {pr.memory_info().rss / 1024**2} MB")
            try:
                data = step_cls.promote(prev_step=fl, **settings)
                data._parent.clear()
            except (levels.FullyFlaggedError, levels.WeatherError) as e:
                logger.warning(str(e))
                return

            fname = fname or f"{data.datestring}.h5"
            fname = output_dir / fname
            data.write(fname, clobber=clobber)

            logger.debug(f"Memory After Writing: {pr.memory_info().rss / 1024**2} MB")

            data.clear()

            logger.debug(f"Memory After Clearing: {pr.memory_info().rss / 1024**2} MB")

            return fname

        if len(input_files) == 1:
            out = [
                _pro(infile, outfile)
                for infile, outfile in zip(input_files, output_fname)
            ]
        else:
            if nthreads > 1:

                def prg(fnc, x, y, **args):
                    return p_tqdm.p_map(fnc, x, y, num_cpus=nthreads, **args)

            else:
                prg = p_tqdm.t_map

            out = list(prg(_pro, input_files, output_fname, unit="files"))
        return [o for o in out if o is not None]


@main.command()  # noqa: A001
@click.argument("settings", type=click.Path(dir_okay=False, exists=True))
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
@click.option("-j", "--nthreads", default=1, help="How many threads to use.")
@click.option(
    "--flag-idx",
    default=-1,
    type=int,
    help="""
    Set this to a non-negative integer to copy the input files to a new location
    and clear all flags up to the given index, performing the filter based on those
    flags.
    """,
)
@click.option(
    "-l",
    "--label",
    default="",
    help="""A label for the output. This label should be unique to the input settings
    (but may be applied to different input files).
    """,
)
@click.option(
    "-c/-C",
    "--clobber/--no-clobber",
    default=False,
    help="""Whether to clobber files -- only applies if flag-idx is applied and a label
    is given, and the label already exists. If False, the program will interactively
    ask.
    """,
)
def filter(settings, path, nthreads, flag_idx, label, clobber):  # noqa: A001
    """Filter a dataset using SETTINGS.

    SETTINGS
        is a YAML settings file. The available settings for each step can be seen
        in the respective documentation for the classes "promote" method.

    Takes one or more ``--input`` files that are the output of a process.

    The output is written within the given input files, inside a special "flags"
    hDF5 group.
    """
    console.print(
        Panel("edges-analysis [blue]filter[/]", box=box.DOUBLE_EDGE),
        style="bold",
        justify="center",
    )

    console.print(Rule("Setting Up"))

    with open(settings) as fl:
        settings = yaml.load(fl, Loader=yaml.FullLoader)

    if isinstance(settings, dict):
        raise OSError("The settings file for filters should be a list")

    console.print()
    tab = Table(title="Settings", show_header=False)
    tab.add_column()
    tab.add_column()
    tab.add_column()
    for item in settings:
        k = list(item.keys())[0]
        v = item[k]
        if v:
            for i, (param, val) in enumerate(v.items()):
                tab.add_row(k if not i else "", param, str(val))
    console.print(tab)
    console.print()

    file_filter = h5py.is_hdf5

    # Get input file(s). If doing initial calibration, get them from raw_field_data
    # otherwise they should be in field_products.
    path = [expand_colon(p, raw=False).expanduser() for p in path]
    input_files = sum((_get_files(p, filt=file_filter) for p in path), [])

    if not input_files:
        logger.error(f"No input files were found! Paths: {path}")
        return
    else:
        console.print("[bold]Input Files:")
        for fl in input_files:
            console.print(f"   {fl}")
        console.print()

    # Check that input files are all homogeneously processed
    if len({p.parent for p in input_files}) != 1:
        raise ValueError("Your input files do not come from a single processing.")

    input_data = [levels.read_step(fl, validate=False) for fl in input_files]

    # Save the settings file
    output_dir = input_files[0].parent

    if flag_idx >= 0:
        if label:
            output_dir /= label

            if output_dir.exists():
                if (
                    clobber
                    or qs.confirm(f"The label '{label}' already exists. Remove?").ask()
                ):
                    shutil.rmtree(output_dir)
                else:
                    logger.info("OK. Exiting")
                    sys.exit()

            output_dir.mkdir()

            for fl in input_files:
                shutil.copy(fl, output_dir / fl.name)

            input_files = [output_dir / fl.name for fl in input_files]
            input_data = [levels.read_step(fl, validate=False) for fl in input_files]
        elif not (
            clobber
            or qs.confirm(
                "Using flag_idx without a label removes flagging steps in place. "
                "Is this really what you want?"
            ).ask()
        ):
            logger.info("OK. Exiting.")

        for d in input_data:
            with d.open("r+") as flobj:
                if "flags" not in flobj:
                    if flag_idx > 0:
                        raise ValueError(
                            f"{d.filename} has no flag array, but you want to keep "
                            f"{flag_idx} filters."
                        )
                    else:
                        continue

                dset = flobj["flags"]["flags"]
                dset.resize(flag_idx, axis=0)

                for name, indx in dict(flobj["flags"].attrs).items():
                    if indx >= flag_idx:
                        del flobj["flags"].attrs[name]
                        del flobj["flags"][name]

    n_filters = len(list(output_dir.glob("filter_*.yaml")))
    with open(output_dir / f"filter_settings_{n_filters}.yaml", "w") as fl:
        yaml.dump(settings, fl)

    # Actually call the relevant function
    console.print()
    console.print(Rule("Beginning Filtering"))

    for item in settings:
        filt = list(item.keys())[0]
        cfg = item[filt] or {}
        fnc = filters.get_step_filter(filt)
        fnc(data=input_data, in_place=True, n_threads=nthreads, **cfg)

    console.print(Rule("Done Filtering"))

    console.print()
    console.print("[bold]All flags written inside input files.")
