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
edges-analysis calibrate devel/example_level1_settings.yaml [raw_acq_file.acq] -m "A simple calibration example"
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

The rest of the levels share a unified interface:
```bash
edges-analysis level LEVEL_NUMBER SETTINGS_FILE [OPTIONS]
```

A particular useful feature is that by setting the option `-L` (or `--prev-label`),
we can automatically find the files from the previous step to input.
For example, Level2, in which we combine a set of consistently calibrated files:

```bash
edges-analysis level 2 example_level2_settings.yaml -L example_level1_settings.yaml
```

Notice that we passed the settings file of Level1 to `-L`: the code knows how to create
the "unique label" that those settings would generate, and automatically find the files
in that directory. If you had explicitly set a label in level 1, you should pass that
to `-L` instead. Alternatively, you can pass an explicit directory in which to search
for the files to combine (or, for other levels, the one file to process):

```bash
edges-analysis level 2 example_level2_settings.yaml -i /path/to/directory/of/level1
```

This is fairly smart: if the path is an absolute directory, it will search there. If not,
it will first search relative to the current working directory, and then relative to
the default cache location for the level you're trying to read in, and then relative to
the root directory of the level cache.

Furthermore, the path can be a glob-style pattern if you wish to only read some certain files,
eg:

```bash
edges-analysis level 2 example_level2_settings.yaml -i /path/to/directory/of/level1/2020_076*
```

If the level you're converting to only allows a single file at a time (eg. Level 3 and 4),
then if you pass a directory (either via the `-L` or `-i`) it must include only one file.
You may specify a single file by combining `-L` and `-i`, eg.:

```bash
edges-analysis level 3 example_level2_settings.yaml -i my_unique_filename.h5 -L example_level2_settings.yaml
```

This will automatically search the default cache location for Level2, and pick the file
`my_unique_filename.h5` from that directory.
