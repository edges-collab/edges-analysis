API Reference
=============

Extensions to the GSData Interface
----------------------------------
.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    edges.analysis.datamodel

I/O Subpackage
--------------
.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    edges.io.io
    edges.io.h5
    edges.io.auxiliary
    edges.io.logging
    edges.io.utils


High Level Methods
------------------
The following modules predominantly provide high-level Python interfaces to update
and visualise GSData objects.

Averaging
~~~~~~~~~
.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    edges.analysis.averaging.combiners
    edges.analysis.averaging.freqbin
    edges.analysis.averaging.lstbin

Calibration
~~~~~~~~~~~
.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    edges.analysis.calibration.calibrate

Filters
~~~~~~~
.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    edges.filters.filters
    edges.filters.lst_model

Auxiliary Data
~~~~~~~~~~~~~~
.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    edges.analysis.aux_data

Visualization
~~~~~~~~~~~~~
.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    edges.analysis.plots


Lower Level Methods
-------------------
The following modules provide lower-level methods that typically underlie the high-level
interface.

.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    edges.analysis.averaging.averaging
    edges.analysis.averaging.utils
    edges.analysis.calibration.labcal
    edges.analysis.calibration.loss
    edges.analysis.calibration.s11

Sky and Beam Modelling
----------------------
.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    edges.analysis.sky_models
    edges.analysis.beams

Package Config and Utilities
----------------------------
.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    edges.analysis.const
    edges.analysis.config
    edges.analysis.tools
    edges.analysis.coordinates
    edges.analysis.groupdays
