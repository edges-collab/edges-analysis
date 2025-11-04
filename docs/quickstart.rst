
Quickstart for the ``edges`` library
------------------------------------

The library is split into subpackages for different steps of the calibration and analysis
pipeline for the EDGES telescope, for example the ``io``, ``cal``, ``averaging``
and ``analysis`` sub-packages. There are also submodules for calculating beam
corrections, sky models, and other useful functions such as plotting.
There is no "one way" to use the ``edges`` package -- it is a *collection* of
routines that can be put together to form a data pipeline (the actual pipelining
is not defined in this package). Nevertheless, there are aspects of the package
that are unified, and make it much easier to work with global spectrum data.

Data Objects
~~~~~~~~~~~~
The primary data object that all functionality is built around is the
:class:`pygsdata.GSData` object. This object stores all relevant data from
a global-signal measurement, and has useful methods for accessing parts of the data,
as well as reading/writing the data to standard formats (including HDF5 and ACQ).

All high-level processing functions in ``edges`` operate on the ``GSData`` object,
and generally return a new (updated) ``GSData`` object. To first order, a pipeline
backed by the ``edges`` package can look very simple because of this::

    data = GSData.from_file("datafile.gsh5")
    data = edges.analysis.do_foo(data, **foo_params)
    data = edges.analysis.do_bar(data, **bar_params)

At each step, the ``data`` object is of the *same type*, making it very easy to write,
make plots etc. Also, no function ever modifies the ``data`` in-place, so you can always
be sure of the current state of any object.

Since the ``GSData`` objects carry a history of each processing function that has
been applied to it, the full history of processing can always be read later::

    print(data.history.pretty)

See the `pygsdata docs <https://pygsdata.readthedocs.io>`_ for more information
about these objects.

Almost all of the functions in the ``edges.analysis`` sub-package are functions like we
just described -- taking in a ``GSData`` object and returning another, keeping track
of history. These are so-called "high-level" analysis functions. Generally, these
are built on top of lower-level functions that can be applied to more general/simple
data interfaces like ``numpy`` arrays. If you visit the
`API Documentation <reference/index.rst>`_, we try to make it clear where to find these
high-level functions (and where to find the low-level counterparts as well).

Example Analysis Session
~~~~~~~~~~~~~~~~~~~~~~~~
While it's impossible to provide a "quintessential" quickstart example, because there are
so many different ways to use the ``edges`` package, here we provide a simple example
of reading some data on-disk, averaging it over times, identifying and flagging RFI,
and applying some pre-computed calibration solutions::

    from pygsdata import GSData, plots
    from edges.cal.dicke import dicke_calibration
    from edges.filters import sun_filter, negative_power_filter, rfi_model_filter
    from edges.averaging import lst_bin
    from edges.cal.s11 import CalibratedS11
    from edges.analysis.calibrate import apply_noise_wave_calibration
    from edges.modelling import LinLog

    data = GSData.from_file("my_multi_integration_raw_file.acq")

    # Flag integrations with negative power
    daa = negative_power_filter(data)

    # Perform dicke-switch calibration
    data = dicke_calibration(data)

    # Filter integrations where the sun is above the horizon
    data = sun_filter(data, elevation_range=(-90, -10))

    # Average data within LST bins
    data = lst_bin(
        data,
        binsize=1.0,     # hour
        first_edge=4.0,  # hours
        max_edge=16.0,   # hours,
    )

    # Now we want to apply calibration solutions.
    # First we need to specify the S11 of the antenna
    ants11 = CalibratedS11.from_calibrated_file("my_antenna_s11.csv")

    # Now calibrate, pointing to pre-computed solutions.
    data = apply_noise_wave_calibration(
        data,
        calibrator="path_to_solutions.h5",
        antenna_s11=ants11
    )

    # Flag RFI by finding outliers compared to a smooth model
    data = rfi_model_filter(data, model=LinLog(n_terms=5), max_iter=1, increase_order=False)

    # Plot a waterfall of the data
    plots.plot_waterfall(data)

    # Write the calibrated, filtered and averaged data to a file
    data.write_gsh5("final_data.gsh5")
