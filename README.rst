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

**Analysis and Calibration Code for the EDGES experiment**

``edges-analysis`` has methods for I/O, receiver calibration, averaging, filtering and
calibrating EDGES (and other global 21cm experiment) data. The primary goal of the code
is to allow the analysis to be fully reproducible, efficient, and clear.

Features
========

``edges-analysis`` includes the following features:

* Methods for reading/writiing EDGES-specific datasets/data products.
* A full set of methods for receiver calibration, applicable to *most*
  current global 21cm experiments (based on Dicke-switch calibration +
  the noise-wave formalism).
* Many algorithms/routines for flagging bad data, including RFI, outliers
  in time/frequency, poor auxiliary data etc.
* A full-featured interface for linear modelling and fitting.
* Algorithms for consistent averaging of data over nights/times/frequencies,
  either in specified bins or complete averaging of the dataset.
* Works with ``pygsdata`` objects for a consistent interface all the way through
  an analysis pipeline, including maintenance of metadata about the operations
  applied to particular data (and propagation of metadata like the number
  of averaged samples).
* Simulation algorithms, including beam models and sky models.

Documentation
=============

Documentation is hosted on `ReadTheDocs <https://edges-analysis.readthedocs.org>`_.


Installation
============

This package can be installed with ``pip``::

   pip install edges-analysis

If you want all the extras (for development etc), use the ``[dev]`` extra, like so::

  pip install edges-analysis[dev]


You can also install directly from github. Either cloning first::

    git clone https://github.com/edges-collab/edges-analysis
    cd edges-analysis
    pip install [-e] .

or directly::

    pip install git+git://github.com/edges-collab/edges-analysis.git
