==============
edges-analysis
==============

.. image:: https://github.com/edges-collab/edges-analysis/actions/workflows/test_suite.yaml/badge.svg
  :target: https://github.com/edges-collab/edges-analysis/actions/workflows/test_suite.yaml
.. image:: https://readthedocs.org/projects/edges-analysis/badge/?version=stable
  :target: https://edges-analysis.readthedocs.io/en/stable/?badge=stable
.. image:: https://codecov.io/gh/edges-collab/edges-analysis/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/edges-collab/edges-analysis
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
  :target: https://github.com/psf/black

**Code for analysing EDGES field data.**

``edges-analysis`` has methods for averaging, filtering and calibrating EDGES data
from the field. It also includes classes for describing the products of these analysis
steps. The primary goal of the code is to allow the analysis to be fully reproducible,
efficient, and clear.

.. note:: This code originated from Raul Monsalve's private repo.
          The original code before modification is tagged as ``original_version``.

Features
========

``edges-analysis`` includes the following features:

* A well-defined class-based interface to generic single-dish data for global-signal
  measurements.
* Full tracking of input metadata across all analysis steps.
* Compact command-line interface for running each step.
* Beam models
* Loss models
* Various filters that are able to flag out potentially bad data.
* Sky Models



Installation
============

This package can be installed with ``pip`` either by cloning the repo first, or directly
from github::

    git clone https://github.com/edges-collab/edges-analysis
    cd edges-analysis
    pip install [-e] .

or::

    pip install git+git://github.com/edges-collab/edges-analysis.git

Documentation
=============

Documentation is hosted on `ReadTheDocs <https://edges-analysis.readthedocs.org>`_.

Usage Guide
===========

CLI
---
There is a very basic CLI set up for running a full calibration pipeline
over a set of data. To use it, do

    $ edges cal run --help

Multiple options exist, but the only ones required are ``CONFIG`` and
``PATH``. The first should point to a YAML configuration for the run, and
the second should point to a directory in which exists ``S11``,
``Resistance`` and ``Spectra`` folders. Thus::

    $ edges cal run ~/config.yaml .

will work if you are in such a directory.

The ``config.yaml`` consists of a set of parameters passed to
``edges.cal.CalibrationObservation``. See its docstring for more details.

In addition, you can run a "term sweep" over a given calibration,
iterating over number of Cterms and Wterms until some threshold is met.
This uses the same configuration as ``edges.cal run``, but you can pass a
maximum number of C and W-terms, along with a threshold at which to stop
the sweep (this is a threshold in absolute RMS over degrees of freedom).
This will write out a ``Calibration`` file for the "best" set of
parameters.

You can also create full Jupyter notebook reports (and convert them to
PDF!) using the CLI. To get this to work, you must install ``edges``
with ``pip install edges[report]``. Then you must do the following:

1.  Activate the environment you wish to use to generate the reports
    (usually ``conda activate edges``)
2.  Run
    ``python -m ipykernel install --user --name edges --display-name "edges"``

Note that in the second command, calling it "edges" is necessary
(regardless of the name of your environment!).

Now you can run::

    $ edges cal report PATH --config ~/config.yaml

(obviously there are other parameters -- use ``edges cal report --help``
for help). The ``PATH`` should again be a calibration observation
directory. The config can be the same file as in ``edges cal run``, and is
optional. By default, both a notebook and a PDF will be produced, in the
``outputs/`` directory of the observation. You can turn off the PDF
production with a ``-R`` flag.

Similarly, you can *compare* two observations as a report notebook with::

    $ edges cal compare PATH COMPARE --config ~/config.yaml --config-cmp ~/config.yaml

This is intended to more easily show up what might be wrong in an
observation, when compared to a "golden" observation, for example.
