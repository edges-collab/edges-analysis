Library Usage
=============
The library is split into two main packages: :mod:`~edges_analysis.analysis` and
:mod:`~edges_analysis.simulation`.
Most of the functionality is contained in ``analysis``, which has routines
for correction of reflection measurements, loss corrections, filtering, beams, sky models
and averaging.

While these routines may be used ad hoc, they are joined into a pipeline in the ``levels``
module, which takes data through a process from raw data to calibrated, filtered and
averaged data in a series of discrete steps. Each of these steps is represented
as an HDF5 file, with an associated Python class which makes it easier to access the
data of each step (and make plots etc). It is recommended that you use the higher level
classes unless absolutely necessary. In order of application, these are

* :class:`~edges_analysis.analysis.levels.CalibratedData`
* :class:`~edges_analysis.analysis.levels.FilteredData`
* :class:`~edges_analysis.analysis.levels.ModelData`
* :class:`~edges_analysis.analysis.levels.CombinedData`
* :class:`~edges_analysis.analysis.levels.DayAverageData`
* :class:`~edges_analysis.analysis.levels.BinnedData`

These have a uniform interface for "promoting" one to the next. For instance, to promote
a :class:`~edges_analysis.analysis.levels.CalibratedData` object to a
:class:`~edges_analysis.analysis.levels.FilteredData`::

    filtered_data = FilteredData.promote(calibrated_data, **other_arguments)

However, we typically encourage you to use the command-line interface to promote one
dataset to another, as it will write out each file as a proper HDF5 dataset, and keep
all the data in a nice file layout.

On the other hand, the library is very useful for *inspecting* and *visualising* these
datasets. The most important function here is :func:`~edges_analysis.analysis.levels.read_step`::

    from edges_analysis import read_step, BinnedData
    binned_data = read_step("my_output_binned_data.h5")
    assert isinstance(binned_data, BinnedData)  # Passes
    binned_data.plot_resids()

Look at the tutorials for further instructions.
