API Reference
=============

Extensions to the GSData Interface
----------------------------------
.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    edges_analysis.datamodel

High Level Methods
------------------
The following modules predominantly provide high-level Python interfaces to update
and visualise GSData objects.

Averaging
~~~~~~~~~
.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    edges_analysis.averaging.combiners
    edges_analysis.averaging.freqbin
    edges_analysis.averaging.lstbin

Calibration
~~~~~~~~~~~
.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    edges_analysis.calibration.calibrate

Filters
~~~~~~~
.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    edges_analysis.filters.filters
    edges_analysis.filters.lst_model

Auxiliary Data
~~~~~~~~~~~~~~
.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    edges_analysis.aux_data

Visualization
~~~~~~~~~~~~~
.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    edges_analysis.plots


Lower Level Methods
-------------------
The following modules provide lower-level methods that typically underlie the high-level
interface.

.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    edges_analysis.averaging.averaging
    edges_analysis.averaging.utils
    edges_analysis.calibration.labcal
    edges_analysis.calibration.loss
    edges_analysis.calibration.s11

Sky and Beam Modelling
----------------------
.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    edges_analysis.sky_models
    edges_analysis.beams

Package Config and Utilities
----------------------------
.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    edges_analysis.const
    edges_analysis.config
    edges_analysis.tools
    edges_analysis.coordinates
    edges_analysis.groupdays
