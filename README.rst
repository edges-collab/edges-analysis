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
          The original code before modification is tagged as `original_version`.

Features
========

``edges-analysis`` includes the following features:

* Well-defined objects describing the data at various stages of analysis, backed by
  structured HDF5 files.
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
