# edges-analysis

**Code for analysing EDGES field data.**

This is the code that originated with @gitul's private repo, and has been
extensively modified and formatted to fit the new code structure in the EDGES
collaboration.

The original code before modification is tagged as `original_version`.

## Installation

This package can be installed with `pip` either by cloning the repo first, or directly
from github:

```bash
git clone https://github.com/edges-collab/edges-analysis
cd edges-analysis
pip install [-e] .
```

or

```bash
pip install git+git://github.com/edges-collab/edges-analysis.git
```

## Usage

`edges-analysis` contains a bunch of routines for calibrating, filtering and averaging
field data, as well as routines for simulating uncertainties in the analysis, and
performing parameter estimation on the data. It is designed to be used in two ways:
firstly as a library of functions to use in scripts or interactively, and secondly as a
pipeline via command-line.

### As a library
The library is split into three main packages: `analysis`, `simulation` and `estimation`.
Most of the functionality is contained in `analysis`, which has routines for correction
of reflection measurements, loss corrections, filtering and averaging.

While these routines may be used ad hoc, they are joined into a pipeline in the `levels`
module, which takes data through a process from raw data to calibrated, filtered and
averaged data in a series of four discrete steps. Each of these steps is represented
as an HDF5 file, with an associated Python class which makes it easier to access the
data of each step (and make plots etc).

### From the command-line
We provide a command-line interface for running the full pipeline, with a command for
each Level. Each sub-command can take a YAML file for settings (some settings are
passed directly to the command line). Example YAML files (very basic ones) are provided
in the `devel/` directory.

Here is a basic workflow:

First get some raw data and calibrate it:

```bash
edges-analysis calibrate [raw_acq_file.acq] -s devel/example_level1_settings.yaml -m "A simple calibration example"
```

The acq file we provide here can also be a glob pattern, in which case each file matching
the pattern will be calibrated. By default, the output Level1 files produced will be put
in a directory in a centralized location, whose name is unique to the input settings.
This can be over-ridden by providing a `--label`.

Note the "message" we append  -- it gets put into a `README.txt` file in the result
directory. This is a good place to put a human-readable reason we are running this
particular calibration (or any other notes).

The stdout from this function will give the unique label/directory into which the output
is written.

Onto Level2, in which we combine a set of consistently calibrated files:

```bash
edges-analysis combine [level1_label] -s example_level2_settings.yaml -m "A simple calibration example"
```

Here, we pass the directory in which the consistently-calibrated files lie. This was
told us by the previous step. This can be an absolute directory, or if the previous files
were saved in the default location, it just needs to be the final label.

The message gets written into the HDF5 file itself in this case.

Now Level3:

```bash
edges-analysis level3 example_level2_settings.yaml -s example_level3_settings.yaml -m "A simple calibration example"
```

Note that here instead of a label/directory pointing to the level2 file, we instead
passed the Level2 settings file. From this file, a unique label is determined and it
can find the Level2 data. You could instead pass the directory like in the previous
step.

Finally, Level4:

```bash
edges-analysis level4 example_level3_settings.yaml -s example_level4_settings.yaml -m "A simple calibration example"
```
