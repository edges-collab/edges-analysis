from os.path import join, isdir
from os import mkdir
import glob
import click
from pathlib import Path
import yaml
import h5py
import tqdm

from .analysis.levels import Level1
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

    settings = {
        **settings,
        **{
            "band": band,
            "calfile": calfile,
            "s11_path": s11_path,
            "label": label,
            "message": message,
            "weather_file": weather_file,
            "thermlog_file": thermlog_file,
        },
    }

    print("Settings: ", settings)

    if isdir(path):
        print(f"Attempting to calibrate all files in {path}")
        path = join(path, "*")

    files = glob.glob(path)

    # Get the output structure ready
    hsh = hash(repr(settings))

    if not label:
        label = band + "_" + hsh

    if prefix:
        output_dir = Path(prefix) / label
    else:
        output_dir = Path(config["paths"]["field_products"]) / "level1" / label

    if not output_dir.exists():
        mkdir(str(output_dir))
    else:
        # If the directory is not empty, we need to check whether the files that are
        # already there are consistent with these files.
        for fl in glob.glob(str(output_dir / "*.h5")):
            with h5py.File(fl, "r") as ff:
                for k, v in settings.items():
                    if ff.attrs[k] != v:
                        raise ValueError(
                            f"The directory you want to write to has "
                            f"non-consistent files for key: {k} [{v} vs. {ff.attrs[k]}]."
                        )

    print(f"Output Directory: {output_dir}")

    pbar = tqdm.tqdm(files, unit="file")
    for fl in pbar:
        pbar.set_description(f"{fl}")
        l1 = Level1(filename=fl, **settings)

        fname = f"level1_{l1.meta['year']}_{l1.meta['day']}_{l1.meta['hour']}_{hsh}.h5"
        fname = output_dir / fname

        l1.write(fname)
