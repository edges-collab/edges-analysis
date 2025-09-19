I/O
---

The ``io`` module provides functions both for reading/writing EDGES data formats (for
different datasets such as spectra, S11's, thermistor readings, etc) as well as the
ability to quickly specify the location of sets of files required to define a full
"calibration observation", with support for file layouts used for EDGES 2 and 3
(but also arbitrary file locations).

.. autosummary::
    :toctree: _autosummary
    :recursive:

    edges.io
