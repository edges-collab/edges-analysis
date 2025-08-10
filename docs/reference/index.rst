API Reference
=============

High Level Analysis Methods
---------------------------
The following modules predominantly provide high-level Python interfaces to update
and visualise GSData objects. These should form the primary user interface for
operations on data.

Averaging and Combining Files/Objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    edges.averaging.combiners
    edges.averaging.freqbin
    edges.averaging.lstbin
    edges.analysis.groupdays

Calibration
~~~~~~~~~~~
.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    edges.analysis.calibrate

Flagging and Filtering
~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    edges.filters.filters
    edges.filters.lst_model
    edges.filters.runners

Auxiliary Data
~~~~~~~~~~~~~~
.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    edges.analysis.aux_data

Data Modelling/Inpainting
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    edges.analysis.datamodel

Visualization
~~~~~~~~~~~~~
.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    edges.analysis.plots


Low-Level Analysis Methods
---------------------------
The following modules provide lower-level analysis methods that typically underlie the
high-level interfaces listed above, suited for applying averaging/filtering/calibration
to data.

.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    edges.averaging.averaging
    edges.averaging.utils
    edges.analysis.loss
    edges.filters.xrfi

Receiver Calibration
--------------------
To determine the calibration solutions for the receiver (so that it can be later applied
to the data), use functions from the following modules.

High-Level Interface
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    edges.cal.calobs
    edges.cal.calibrator
    edges.cal.load_data
    edges.cal.receiver_cal
    edges.cal.spectra
    edges.cal.apply
    edges.cal.plots

Low-Level Functions
~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    edges.cal.dicke
    edges.cal.ee
    edges.cal.loss
    edges.cal.noise_waves
    edges.cal.reflection_coefficient
    edges.cal.thermistor
    edges.cal.s11.base
    edges.cal.s11.s11model

I/O
---
The ``io`` module provides functions both for reading/writing EDGES data formats (for
different datasets such as spectra, S11's, thermistor readings, etc) as well as the
ability to quickly specify the location of sets of files required to define a full
"calibration observation", with support for file layouts used for EDGES 2 and 3
(but also arbitrary file locations).

.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    edges.io.serialization
    edges.io.calobsdef
    edges.io.calobsdef3
    edges.io.spectra
    edges.io.templogs
    edges.io.thermistor
    edges.io.time_formats
    edges.io.vna
    edges.io.auxiliary


Simulation
----------
Modules containing simulation routines -- for simulating observations given sky models,
beam models, receiver models, etc.

.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    edges.sim.sky_models
    edges.sim.beams
    edges.sim.antenna_beam_factor
    edges.sim.receivercal
    edges.sim.simulate

Linear Models
-------------
Linear modelling forms the backbone of many algorithms within the ``edges`` package.
The ``modelling`` subpackage contains a nice interface for defining and fitting
composite linear models.

.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    edges.modelling.core
    edges.modelling.composite
    edges.modelling.data_transforms
    edges.modelling.xtransforms
    edges.modelling.fitting
    edges.modelling.models


High-Level Interface For Reproducing Legacy Results
---------------------------------------------------
We provide the ``alanmode`` sub-package with included utilities and interfaces that
are geared towards reproducing past EDGES results (e.g. Bowman+2018), by mimicking the
interface of original C-code used for those analyses.

.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    edges.alanmode.alanio
    edges.alanmode.alanmode

Package Config and Utilities
----------------------------
.. autosummary::
    :toctree: _autosummary
    :template: module.rst

    edges.const
    edges.config
    edges.tools
    edges.frequencies
    edges.testing
    edges.types
    edges.units
