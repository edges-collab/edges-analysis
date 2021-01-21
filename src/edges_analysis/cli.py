import hashlib
import glob
import os
import click
from pathlib import Path
import yaml
import h5py
import tqdm
from rich.console import Console
from rich.logging import RichHandler
from .analysis.levels import Level1
from .analysis import levels
from .config import config
import logging
import numpy as np

from rich.panel import Panel
from rich import box
from rich.rule import Rule
from rich.table import Table

console = Console()

main = click.Group()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            # console=console,
            show_time=False,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
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

    console.print("[bold]Settings:")
    for k, v in settings.items():
        console.print(f"    {k}: [dim]{v}[/]")

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


def _get_input_files(level, path, label, allow_many=False):
    """Get the input file(s) to convert to the next level."""
    path = Path(path)
    root = Path(config["paths"]["field_products"])
    lvl = root / f"level{level}"

    if label and Path(label).is_file():
        with open(label, "r") as fl:
            settings = yaml.load(fl, Loader=yaml.FullLoader)
            label = settings.pop("label", hashlib.md5(repr(settings).encode()).hexdigest())
        console.print(f"[bold]Using auto-generated label: [dim]{label}[/]")

    fnc = _get_files if allow_many else _get_unique_file

    try_paths = np.unique(
        [
            path,
            lvl / label / path,
            lvl / path,
            root / path,
            path / label,
            path / f"level{level}" / label,
        ]
    )
    for pth in try_paths:
        try:
            return fnc(pth)
        except ValueError:
            pass

    console.print("[bold]Tried Following Paths: ")

    for pth in try_paths:
        console.print(f"[dim]\t{pth}")

    return []


def _get_all_files(pth: Path, filter=h5py.is_hdf5):
    if pth.is_file():
        return [pth]
    elif pth.is_dir():
        return [fl for fl in pth.glob("*.h5") if filter(fl)]
    else:
        return [Path(fl) for fl in glob.glob(str(pth)) if filter(fl)]


def _get_unique_file(pth):
    files = _get_all_files(pth)
    if not files:
        raise ValueError
    elif len(files) > 1:
        raise IOError(f"More than one file found in {pth}")

    return files[0]


def _get_files(pth: Path, filter=h5py.is_hdf5):
    files = _get_all_files(pth, filter)
    if not files:
        raise ValueError

    return files


def get_output_dir(level, settings, label, prefix):
    hsh = hashlib.md5(repr(settings).encode()).hexdigest()
    label = label or hsh
    prefix = prefix or Path(config["paths"]["field_products"]) / f"level{level}"

    prefix = Path(prefix) / label
    prefix.mkdir(parents=True, exist_ok=True)
    return prefix


def get_output_path(level, settings, in_file, label, prefix):
    prefix = get_output_dir(level, settings, label, prefix)

    if isinstance(in_file, list):
        return prefix / (hashlib.md5(repr(in_file).encode()).hexdigest() + ".h5")
    else:
        return prefix / in_file.name


@main.command(
    context_settings={  # Doing this allows arbitrary options to override config
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
)
@click.argument("settings", type=click.Path(dir_okay=False, exists=True))
@click.argument("path", type=click.Path(dir_okay=True), nargs=-1)
@click.option("-l", "--label", default="")
@click.option("-p", "--prefix", default="")
@click.option("-m", "--message", default="")
@click.option("-x/-X", "--xrfi/--no-xrfi", default=True, help="manually turn off xRFI")
@click.option(
    "-c/-C",
    "--clobber/--no-clobber",
    help="whether to overwrite any existing data at the output location",
)
@click.pass_context
def calibrate(ctx, settings, path, label, prefix, message, xrfi, clobber):
    """Calibrate field data to produce Level1 files."""
    console.print(
        Panel("edges-analysis [blue]calibrate[/]", box=box.DOUBLE_EDGE),
        style="bold",
        justify="center",
    )

    cli_settings = _ctx_to_dct(ctx.args)
    settings = _get_settings(settings, xrfi, **cli_settings)

    label = settings.pop("label", "") or label

    path = [Path(p) for p in path]

    root = Path(config["paths"]["raw_field_data"])
    root_data = root / "mro" / settings["band"]

    files = []
    for p in path:
        for pth in [p, root / p, root_data / p]:
            try:
                files += _get_files(pth, filter=lambda x: True)
                break
            except ValueError:
                pass

    if not files:
        logger.warning("No input files were found!")
        return

    output_dir = get_output_dir(1, settings, label, prefix)

    console.print(
        f"[bold]Output Directory: [dim]{output_dir}",
    )

    # If the directory is not empty, we need to check whether the files that are
    # already there are consistent with these files.
    if output_dir.glob("*.h5"):
        current_files = output_dir.glob("*")
        if clobber:
            for fl in current_files:
                os.remove(str(fl))
        else:
            for fl in [fl for fl in current_files if h5py.is_hdf5(fl)]:
                with h5py.File(fl, "r") as ff:
                    for k, v in settings.items():
                        if k not in ["calfile", "s11_path"] and (
                            k not in ff.attrs or ff.attrs[k] != v
                        ):
                            if k in ff.attrs:
                                try:
                                    v = Path(v).expanduser().absolute()
                                except Exception:
                                    pass

                                if ff.attrs[k] == str(v):
                                    continue

                            meta = "\n\t".join(f"{kk}: {vv}" for kk, vv in ff.attrs.items())
                            raise ValueError(
                                f"""
The directory you want to write to has a non-consistent file for key '{k}' [required {v}].
Filename: {fl.name}
Metadata in file:
        {meta}
"""
                            )

    if message:
        with open(output_dir / "README.txt", "w") as fl:
            fl.write(message)

    with open(output_dir / "settings.yaml", "w") as fl:
        yaml.dump(settings, fl)

    pbar = tqdm.tqdm(files, unit="files")
    for fl in pbar:
        pbar.set_description(f"{fl.name}")
        l1 = Level1.from_acq(filename=fl, leave_progress=False, **settings)

        t = l1.datetimes[0]
        fname = f"{t.year}_{l1.meta['day']:>03}_{t.hour:>02}_{t.minute:>02}_{t.second:>02}.h5"
        fname = output_dir / fname

        l1.write(fname)

    console.print(
        f"[bold]All files written to: [dim]{output_dir}",
    )


@main.command(
    context_settings={  # Doing this allows arbitrary options to override config
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
)
@click.argument("level", type=int)
@click.argument("settings", type=click.Path(exists=True, dir_okay=False))
@click.option("-i", "--path", default="", help="path to input files")
@click.option("-l", "--label", default="", help="optional short label describing current settings")
@click.option(
    "-L",
    "--prev-label",
    default="",
    help="optional short label describing settings of input data",
)
@click.option(
    "-p",
    "--prefix",
    default="",
    help="optional non-standard location to write the data",
)
@click.option(
    "-m",
    "--message",
    default="",
    help="optional message to insert into the HDF5 file describing the reason for this analysis",
)
@click.option("-x/-X", "--xrfi/--no-xrfi", default=True, help="manually turn off xRFI")
@click.option(
    "-c/-C",
    "--clobber/--no-clobber",
    help="whether to overwrite any existing data at the output location",
)
@click.pass_context
def level(ctx, level, settings, path, label, prev_label, prefix, message, xrfi, clobber):
    """Bump from a level to the next level."""
    assert level > 1

    console.print(
        Panel(f"edges-analysis [blue]Level {level}[/]", box=box.DOUBLE_EDGE),
        style="bold",
        justify="center",
    )

    cli_settings = _ctx_to_dct(ctx.args)
    settings = _get_settings(settings, xrfi, **cli_settings)
    label = settings.pop("label", "") or label

    in_files = _get_input_files(level - 1, path, prev_label, level in [2])
    if not in_files:
        console.print("[bold red]Found no input files!")
        return

    if isinstance(in_files, list):
        console.print(f"[bold]Combining {len(in_files)} Level{level-1} files:")
        for fl in in_files:
            console.print(f"[blue]\t{fl.absolute()}")
    else:
        console.print(f"[bold]Processing[/] '{in_files.name}'")

    # Get the output structure ready
    output_file = get_output_path(level, settings, in_files, label, prefix)

    if output_file.exists() and not clobber:
        logger.error(
            f"[bold red]The output file [blue]'{output_file}'[/] already exists -- use clobber!"
        )
        return

    console.print()
    console.print(Rule("Beginning Level Upgrade"))
    getattr(levels, f"Level{level}").from_previous_level(in_files, output_file, clobber, **settings)

    with h5py.File(output_file, "a") as fl:
        fl.attrs["message"] = message

    console.print(f"[bold]Output File: [blue]{output_file}")
