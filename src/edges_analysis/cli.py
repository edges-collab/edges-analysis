import hashlib
import glob
import os
import click
from pathlib import Path
import yaml
import h5py
import tqdm
from colored import stylize, fg, bg, attr
from .analysis.levels import Level1
from .analysis import levels
from .config import config
from edges_io.logging import logger
import logging

main = click.Group()

logger.setLevel(logging.INFO)


def _get_settings(settings):
    with open(settings, "r") as fl:
        settings = yaml.load(fl, Loader=yaml.FullLoader)

    print(stylize(f"Settings:", attr("bold")))
    for k, v in settings.items():
        print(f"    {k}:", stylize(v, attr("dim")))

    return settings


def _get_input_files(level, path, label, allow_many=False):
    """Get the input file to convert to the next level. Only applicable for levels
    that require a single file."""
    path = Path(path)
    root = Path(config["paths"]["field_products"])
    lvl = root / f"level{level}"

    if label and Path(label).is_file():
        with open(label, "r") as fl:
            settings = yaml.load(fl, Loader=yaml.FullLoader)
            label = settings.pop(
                "label", hashlib.md5(repr(settings).encode()).hexdigest()
            )
        print(
            stylize("Using auto-generated label: ", attr("bold")),
            stylize(label, attr("dim")),
        )

    fnc = _get_files if allow_many else _get_unique_file

    for pth in [path, lvl / label / path, lvl / path, root / path]:
        try:
            return fnc(pth)
        except ValueError:
            pass

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
    if len(files) != 1:
        raise ValueError
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


@main.command()
@click.argument("settings", type=click.Path(dir_okay=False, exists=True))
@click.argument("path", type=click.Path(dir_okay=True), nargs=-1)
@click.option("-l", "--label", default="")
@click.option("-p", "--prefix", default="")
@click.option("-m", "--message", default="")
@click.option(
    "-c/-C",
    "--clobber/--no-clobber",
    help="whether to overwrite any existing data at the output location",
)
def calibrate(settings, path, label, prefix, message, clobber):
    """Calibrate field data to produce Level1 files."""
    settings = _get_settings(settings)
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

    print(
        stylize("Output Directory:", attr("bold")),
        stylize(str(output_dir), attr("dim")),
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

                            meta = "\n\t".join(
                                f"{kk}: {vv}" for kk, vv in ff.attrs.items()
                            )
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


@main.command()
@click.argument("level", type=int)
@click.argument("settings", type=click.Path(exists=True, dir_okay=False))
@click.option("-i", "--path", default="", help="path to input files")
@click.option(
    "-l", "--label", default="", help="optional short label describing current settings"
)
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
@click.option(
    "-c/-C",
    "--clobber/--no-clobber",
    help="whether to overwrite any existing data at the output location",
)
def level(level, settings, path, label, prev_label, prefix, message, clobber):
    """Bump from a level to the next level."""
    assert level > 1

    settings = _get_settings(settings)
    label = settings.pop("label", "") or label

    in_files = _get_input_files(level - 1, path, prev_label, level in [2])
    if not in_files:
        print(stylize("Found no input files!", fg("red") + attr("bold")))
        return

    if isinstance(in_files, list):
        print(stylize(f"Combining {len(in_files)} Level{level-1} files:", attr("bold")))
        for fl in in_files:
            print(stylize(f"\t{fl.absolute()}", attr("dim")))
    else:
        print(stylize(f"Processing {in_files.name}", attr("bold")))

    # Get the output structure ready
    output_file = get_output_path(level, settings, in_files, label, prefix)

    print(stylize("Output File:", attr("bold")), stylize(str(output_file), attr("dim")))

    getattr(levels, f"Level{level}").from_previous_level(
        in_files, output_file, clobber, **settings
    )

    with h5py.File(output_file, "a") as fl:
        fl.attrs["message"] = message
