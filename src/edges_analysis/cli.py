import glob
import logging
import os
import sys
from pathlib import Path
from typing import List, Type, Optional, Union

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
from .analysis import filters
from .analysis import levels
from .config import config

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


def _get_settings(settings, xrfi, **cli_settings):
    with open(settings, "r") as fl:
        settings = yaml.load(fl, Loader=yaml.FullLoader)

    settings.update(cli_settings)
    if not xrfi and settings["xrfi_pipe"]:
        settings["xrfi_pipe"] = {}

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


def _get_files(pth: Path, filter=h5py.is_hdf5) -> List[Path]:
    if pth.is_dir():
        return [fl for fl in pth.glob("*") if filter(fl)]
    else:
        return [Path(fl) for fl in glob.glob(str(pth)) if filter(Path(fl))]


def get_output_dir(prefix, label, settings):
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
                f"{out} has existing files with different settings. Remove existing and "
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
        ["calibrate", "filter", "model", "combine", "day", "bin"], case_sensitive=False
    ),
)
@click.argument("settings", type=click.Path(dir_okay=False, exists=True))
@click.option(
    "-i",
    "--path",
    type=click.Path(dir_okay=True),
    multiple=True,
    help="""The path(s) to input files. Multiple specifications of ``-i`` can be included.
    Each input path may have glob-style wildcards, eg. ``/path/to/file.*``. If the path
    is a directory, all HDF5/ACQ files in the directory will be used. You may prefix the
    path with a colon to indicate the "standard" location (given by ``config['paths']``),
    e.g. ``-i :big-calibration/``.
    """,
)
@click.option(
    "-l",
    "--label",
    default="",
    help="""A label for the output. This label should be unique to the input settings
    (but may be applied to different input files). If the same label is used for different
    settings, the existing processed data will be removed (after prompting).
    """,
)
@click.option(
    "-m",
    "--message",
    default="",
    help="""A message to save with the data. The message will be saved in a README.txt
    file alongside the output data file(s). It is intended to provide a human-understandable
    "reason" for running the particular analysis with the particular settings.
    """,
)
@click.option(
    "-x/-X",
    "--xrfi/--no-xrfi",
    default=True,
    help="Manually turn off xRFI. Useful to quickly shut off xRFI without changing settings.",
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
@click.option("-j", "--nthreads", default=1, help="How many threads to use.")
@click.pass_context
def process(ctx, step, settings, path, label, message, xrfi, clobber, output, nthreads):
    """Process a dataset to the STEP level of averaging/filtering using SETTINGS.

    STEP
        defines the analysis step as a string. Each of the steps should be applied
        in turn.
    SETTINGS
        is a YAML settings file. The available settings for each step can be seen
        in the respective documentation for the classes "promote" method.

    Each STEP should take one or more ``--input`` files that are the output of a previous
    step. The first step (``calibrate``) should take raw ``.acq`` or ``.h5`` spectrum
    files.

    The output files are placed in a directory inside the input file directory, with a
    name determined by the ``--label``.
    """
    console.print(
        Panel(f"edges-analysis [blue]{step}[/]", box=box.DOUBLE_EDGE),
        style="bold",
        justify="center",
    )

    console.print(Rule("Setting Up"))

    step_cls = {
        "calibrate": levels.CalibratedData,
        "filter": levels.FilteredData,
        "model": levels.ModelData,
        "combine": levels.CombinedData,
        "day": levels.DayAveragedData,
        "bin": levels.BinnedData,
    }[step]

    cli_settings = _ctx_to_dct(ctx.args)
    settings = _get_settings(settings, xrfi, **cli_settings)
    label = settings.pop("label", "") or label

    if not label:
        label = qs.text("Provide a short label to identify this run:").ask()

    if step == "calibrate":

        def file_filter(pth: Path):
            return pth.suffix[1:] in io.Spectrum.supported_formats

    else:
        file_filter = h5py.is_hdf5

    # Get input file(s). If doing initial calibration, get them from raw_field_data
    # otherwise they should be in field_products.
    path = [
        expand_colon(p, band=settings.get("band"), raw=step == "calibrate").expanduser()
        for p in path
    ]
    input_files = sum((_get_files(p, filter=file_filter) for p in path), [])

    if not input_files:
        logger.error(f"No input files were found! Paths: {path}")
        return
    else:
        console.print("[bold]Input Files:")
        for fl in input_files:
            console.print(f"   {fl}")
        console.print()
    # Check that input files are all homogeneously processed
    if step != "calibrate" and len({p.parent for p in input_files}) != 1:
        raise ValueError("Your input files do not come from a single processing.")

    # Get unique output directory
    if step == "calibrate":
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
                Path(qs.text("Provide a filename for the output combined file:").ask()).with_suffix(
                    ".h5"
                )
            ]
        else:
            output = [Path(output).with_suffix(".h5")]
    elif step != "calibrate":
        output = [p.name for p in input_files]
    else:
        output = None

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
    out_paths = promote(input_files, nthreads, output_dir, output, step_cls, settings)
    console.print(Rule("Done Processing"))

    for pth in out_paths:
        with h5py.File(pth, "a") as fl:
            fl.attrs["message"] = message

    console.print()
    if step == "combine":
        console.print(f"[bold]Output File: [blue]{out_paths[0]}")
    elif step == "calibrate":
        console.print(
            f"[bold]All files written to: [dim]{output_dir}",
        )
    else:
        console.print("[bold]Output Files:")
        for fname in output:
            console.print(f"\t[bold]{output_dir}/{fname}")


def promote(
    input_files: List[Path],
    nthreads: int,
    output_dir: Path,
    output_fname: Optional[List[Path]],
    step_cls: Type[levels._ReductionStep],
    settings: dict,
) -> List[Path]:
    """Calibrate field data to produce CalibratedData files."""

    if step_cls._multi_input:
        data = step_cls.promote(prev_step=input_files, **settings)
        data.write(output_dir / output_fname[0])
        return [output_dir / output_fname[0]]
    else:
        output_fname = output_fname or [None] * len(input_files)

        def _pro(fl, fname):
            try:
                data = step_cls.promote(prev_step=fl, **settings)
            except (levels.FullyFlaggedError, levels.WeatherError) as e:
                logger.warning(str(e))
                return
            fname = fname or f"{data.datestring}.h5"
            fname = output_dir / fname
            data.write(fname)
            return fname

        if len(input_files) == 1:
            out = [_pro(input_files[0], output_fname[0])]
        else:
            out = list(
                p_tqdm.p_umap(_pro, input_files, output_fname, unit="files", num_cpus=nthreads)
            )
        return [o for o in out if o is not None]


@main.command()
@click.argument("path", nargs=-1)
@click.argument("settings", type=click.Path(exists=True, dir_okay=False))
@click.argument("outfile", type=click.Path(exists=False, dir_okay=False))
def rms_info(path, settings, outfile):
    console.print(
        Panel("edges-analysis [blue]RMSInfo[/]", box=box.DOUBLE_EDGE),
        style="bold",
        justify="center",
    )

    # Get input file(s).
    path = [expand_colon(p, raw=False).expanduser() for p in path]
    input_files = sorted(sum((_get_files(p) for p in path), []))
    objects = [levels.read_step(p) for p in input_files]

    n_files = settings.pop("n_files", len(input_files))

    rms_info = filters.get_rms_info(level1=objects[:n_files], **settings)

    rms_info.write(outfile)

    console.print(f"Wrote RMSInfo to {outfile}.")


# def check_existing_file_settings(output_dir, settings, clobber):
#     # If the directory is not empty, we need to check whether the files that are
#     # already there are consistent with these files.
#     if not output_dir.glob("*.h5"):
#         return
#
#     current_files = output_dir.glob("*")
#
#     if clobber:
#         for fl in current_files:
#             os.remove(str(fl))
#         return
#
#     for fl in [fl for fl in current_files if h5py.is_hdf5(fl)]:
#         with h5py.File(fl, "r") as ff:
#             for k, v in settings.items():
#                 if k not in ["calfile", "s11_path"] and (
#                     k not in ff.attrs or ff.attrs[k] != v
#                 ):
#                     if k in ff.attrs:
#                         try:
#                             v = Path(v).expanduser().absolute()
#                         except Exception:
#                             pass
#
#                         if ff.attrs[k] == str(v):
#                             continue
#
#                     meta = "\n\t".join(f"{kk}: {vv}" for kk, vv in ff.attrs.items())
#                     raise ValueError(
#                         f"""
# The directory you want to write to has a non-consistent file for key '{k}' [required {v}].
# Filename: {fl.name}
# Metadata in file:
#     {meta}
# """
#                     )
#
#
# @process.command()
# @click.pass_context
# def level(ctx, level, settings, path, label, prev_label, prefix, message, xrfi, clobber):
#     """Bump from a level to the next level."""
#     assert level > 1
#
#     console.print(
#         Panel(f"edges-analysis [blue]Level {level}[/]", box=box.DOUBLE_EDGE),
#         style="bold",
#         justify="center",
#     )
#
#     in_files = _get_input_files(level - 1, path, prev_label, level in [2])
#
#     if isinstance(in_files, list):
#         console.print(f"[bold]Combining {len(in_files)} Level{level - 1} files:")
#         for fl in in_files:
#             console.print(f"[blue]\t{fl.absolute()}")
#     else:
#         console.print(f"[bold]Processing[/] '{in_files}'")
#
#     # Get the output structure ready
#     output_file = get_output_path(level, settings, in_files, label, prefix)
#
#     if output_file.exists() and not clobber:
#         logger.error(
#             f"[bold red]The output file [blue]'{output_file}'[/] already exists -- use clobber!"
#         )
#         return
