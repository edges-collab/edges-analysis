from os.path import join, isdir
import hashlib
import glob
import click
from pathlib import Path
import yaml
import h5py
import tqdm
from colored import stylize, fg, bg, attr
from .analysis.levels import Level1, Level2, Level3, Level4
from .config import config

main = click.Group()


@main.command()
@click.argument("path", type=click.Path(dir_okay=True))
@click.option(
    "-b",
    "--band",
    default=None,
    type=click.Choice(["mid", "low1", "low2", "low3", "high"]),
)
@click.option(
    "-c", "--calfile", default=None, type=click.Path(exists=True, dir_okay=False)
)
@click.option("-S", "--s11-path", default=None, type=click.Path(dir_okay=True))
@click.option("-l", "--label", default="")
@click.option("-p", "--prefix", default="")
@click.option("-m", "--message", default="")
@click.option(
    "--weather-file", default=None, type=click.Path(exists=True, dir_okay=False)
)
@click.option(
    "--thermlog-file", default=None, type=click.Path(exists=True, dir_okay=False)
)
@click.option(
    "-s", "--settings", default=None, type=click.Path(exists=True, dir_okay=False)
)
def calibrate(
    path,
    band,
    calfile,
    s11_path,
    label,
    prefix,
    message,
    weather_file,
    thermlog_file,
    settings,
):
    """
    Calibrate field data to produce Level1 files.
    """
    if settings:
        with open(settings, "r") as fl:
            settings = yaml.load(fl, Loader=yaml.FullLoader)
    else:
        settings = {}

    # Overwrite settings for cmd-line provided parameters.
    for k in ["band", "calfile", "s11_path", "weather_file", "thermlog_file"]:
        v = locals()[k]
        if v:
            settings[k] = v

    print(stylize(f"Settings:", attr("bold")))
    for k, v in settings.items():
        print(f"    {k}:", stylize(v, attr("dim")))

    path = Path(path)

    if not path.is_absolute():
        if (Path(config["paths"]["raw_field_data"]) / path).exists():
            path = Path(config["paths"]["raw_field_data"]) / path

    if path.is_dir():
        print(f"Attempting to calibrate all files in {path}")
        path = path / "*"

    files = [Path(fl) for fl in glob.glob(str(path))]

    # Get the output structure ready
    hsh = hashlib.md5(repr(settings).encode()).hexdigest()

    if not label:
        label = settings["band"] + "_" + hsh

    if prefix:
        output_dir = Path(prefix) / label
    else:
        output_dir = Path(config["paths"]["field_products"]) / "level1" / label

    print(
        stylize("Output Directory:", attr("bold")),
        stylize(str(output_dir), attr("dim")),
    )

    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    else:
        # If the directory is not empty, we need to check whether the files that are
        # already there are consistent with these files.
        for fl in [fl for fl in glob.glob(str(output_dir / "*")) if h5py.is_hdf5(fl)]:
            with h5py.File(fl, "r") as ff:
                for k, v in settings.items():
                    if ff.attrs[k] != v:
                        raise ValueError(
                            f"The directory you want to write to has "
                            f"non-consistent files for key: {k} [{v} vs. {ff.attrs[k]}]."
                        )

    if message:
        with open(output_dir / "README.txt", "w") as fl:
            fl.write(message)

    with open(output_dir / "settings.yaml", "w") as fl:
        yaml.dump(settings, fl)

    pbar = tqdm.tqdm(files, unit="files")
    for fl in pbar:
        pbar.set_description(f"{fl.name}")
        l1 = Level1.from_acq(filename=fl, **settings)

        fname = f"level1_{l1.meta['year']}_{l1.meta['day']}_{l1.meta['hour']}_{hsh}.h5"
        fname = output_dir / fname

        l1.write(fname)


@main.command()
@click.argument("path", type=click.Path(dir_okay=True))
@click.option("-l", "--label", default="")
@click.option("-p", "--prefix", default="")
@click.option("-m", "--message", default="")
@click.option("-a/-A", "--append-label/--no-append-label")
@click.option(
    "-s", "--settings", default=None, type=click.Path(exists=True, dir_okay=False)
)
@click.option("-c/-C", "--clobber/--no-clobber")
def combine(path, label, prefix, message, settings, append_label, clobber):
    """Combine several Level1 files and filter."""

    if settings:
        with open(settings, "r") as fl:
            settings = yaml.load(fl, Loader=yaml.FullLoader)
    else:
        settings = {}

    path = Path(path)

    # Get the output structure ready
    hsh = hashlib.md5(repr(settings).encode()).hexdigest()

    if not label:
        label = hsh

    if append_label:
        label += "-" + path.parent.name

    if path.exists():
        if path.is_dir():
            files = [Path(fl) for fl in glob.glob(str(path / "*.h5"))]
        else:
            files = [path]
    else:
        if (Path(config["paths"]["field_products"]) / "level1" / path).exists():
            path = Path(config["paths"]["field_products"]) / "level1" / path
            if path.is_dir():
                files = [Path(fl) for fl in glob.glob(str(path / "*.h5"))]
            else:
                files = [Path(fl) for fl in glob.glob(str(path))]
        else:
            files = [Path(fl) for fl in glob.glob(str(path))]

    print(stylize(f"Combining {len(files)} Level1 files", attr("bold")))

    if prefix:
        output_file = Path(prefix) / (label + ".h5")
    else:
        output_file = (
            Path(config["paths"]["field_products"]) / "level2" / (label + ".h5")
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(stylize("Output File:", attr("bold")), stylize(str(output_file), attr("dim")))

    Level2.from_previous_level(files, output_file, clobber, **settings)

    with h5py.File(output_file, "a") as fl:
        fl.attrs["message"] = message


@main.command()
@click.argument("path", type=click.Path(dir_okay=True))
@click.option("-l", "--label", default="")
@click.option("-p", "--prefix", default="")
@click.option("-m", "--message", default="")
@click.option(
    "-s", "--settings", default=None, type=click.Path(exists=True, dir_okay=False)
)
@click.option("-c/-C", "--clobber/--no-clobber")
def level3(path, label, prefix, message, settings, clobber):

    if settings:
        with open(settings, "r") as fl:
            settings = yaml.load(fl, Loader=yaml.FullLoader)
    else:
        settings = {}

    path = Path(path)

    # Get the output structure ready
    hsh = hashlib.md5(repr(settings).encode()).hexdigest()

    if not label:
        label = hsh

    if path.exists():
        if h5py.is_hdf5(path):
            l2 = path
        elif path.suffix in [".yaml", ".yml"]:
            with open(path, "r") as fl:
                l2_settings = yaml.load(fl, Loader=yaml.Loader)
                l2_hsh = hashlib.md5(repr(l2_settings).encode()).hexdigest()
                l2 = (
                    Path(config["paths"]["field_products"])
                    / "level2"
                    / (l2_hsh + ".h5")
                )
    else:
        l2 = Path(config["paths"]["field_products"]) / "level2" / path

    print(stylize(f"Averaging over nights for Level2 file: ", attr("bold")), l2.name)

    if prefix:
        output_file = Path(prefix) / (label + ".h5")
    else:
        output_file = (
            Path(config["paths"]["field_products"]) / "level3" / (label + ".h5")
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(stylize("Output File:", attr("bold")), stylize(str(output_file), attr("dim")))

    Level3.from_previous_level(l2, output_file, clobber, **settings)

    with h5py.File(output_file, "a") as fl:
        fl.attrs["message"] = message


@main.command()
@click.argument("path", type=click.Path(dir_okay=True))
@click.option("-l", "--label", default="")
@click.option("-p", "--prefix", default="")
@click.option("-m", "--message", default="")
@click.option(
    "-s", "--settings", default=None, type=click.Path(exists=True, dir_okay=False)
)
@click.option("-c/-C", "--clobber/--no-clobber")
def level4(path, label, prefix, message, settings, clobber):

    if settings:
        with open(settings, "r") as fl:
            settings = yaml.load(fl, Loader=yaml.FullLoader)
    else:
        settings = {}

    path = Path(path)

    # Get the output structure ready
    hsh = hashlib.md5(repr(settings).encode()).hexdigest()

    if not label:
        label = hsh

    if path.exists():
        if h5py.is_hdf5(path):
            l2 = path
        elif path.suffix in [".yaml", ".yml"]:
            with open(path, "r") as fl:
                l2_settings = yaml.load(fl, Loader=yaml.Loader)
                l2_hsh = hashlib.md5(repr(l2_settings).encode()).hexdigest()
                l2 = (
                    Path(config["paths"]["field_products"])
                    / "level3"
                    / (l2_hsh + ".h5")
                )
    else:
        l2 = Path(config["paths"]["field_products"]) / "level3" / path

    print(
        stylize(f"Averaging over GHA for Level3 file: ", attr("bold")),
        stylize(l2.name, attr("dim")),
    )

    if prefix:
        output_file = Path(prefix) / (label + ".h5")
    else:
        output_file = (
            Path(config["paths"]["field_products"]) / "level4" / (label + ".h5")
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(stylize("Output File:", attr("bold")), stylize(str(output_file), attr("dim")))

    Level4.from_previous_level(l2, output_file, clobber, **settings)

    with h5py.File(output_file, "a") as fl:
        fl.attrs["message"] = message
