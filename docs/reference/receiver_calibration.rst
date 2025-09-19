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
