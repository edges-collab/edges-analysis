I/O
---

The ``io`` submodule provides functions both for reading/writing EDGES data formats (for
different datasets such as spectra, S11's, thermistor readings, etc) as well as the
ability to quickly specify the location of sets of files required to define a full
"calibration observation", with support for file layouts used for EDGES 2 and 3
(but also arbitrary file locations).

High-Level File Specification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    edges.io.calobsdef
    edges.io.calobsdef3
    edges.io.ants11

Reading/Writing Data Formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    edges.io.auxiliary
    edges.io.spectra
    edges.io.templogs
    edges.io.thermistor
    edges.io.vna

General Utilities
~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    edges.io.serialization
    edges.io.time_formats
