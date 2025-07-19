"""Various plotting functions."""

from __future__ import annotations

import matplotlib.pyplot as plt

from . import calobs as cc
from .thermistor import get_temperature_thermistor


def plot_raw_spectrum(
    spectrum, freq=None, fig=None, ax=None, xlabel=True, ylabel=True, **kwargs
):
    """
    Make a plot of the averaged uncalibrated spectrum associated with this load.

    Parameters
    ----------
    thermistor : bool
        Whether to plot the thermistor temperature on the same axis.
    fig : Figure
        Optionally, pass a matplotlib figure handle which will be used to plot.
    ax : Axis
        Optional, pass a matplotlib Axis handle which will be added to.
    xlabel : bool
        Whether to make an x-axis label.
    ylabel : bool
        Whether to plot the y-axis label
    kwargs :
        All other arguments are passed to `plt.subplots()`.
    """
    if isinstance(spectrum, cc.LoadSpectrum):
        freq = spectrum.freq
        spectrum = spectrum.averaged_spectrum
    else:
        assert freq is not None

    if fig is None:
        fig, ax = plt.subplots(1, 1, **kwargs)

    ax.plot(freq, spectrum)
    if ylabel:
        ax.set_ylabel("$T^*$ [K]")

    ax.grid(True)
    if xlabel:
        ax.set_xlabel("Frequency [MHz]")


def plot_resistance_measurements(
    resistance: QTable,
    quantity="thermistor_temp",
    ax=None,
    xlabel="Time",
) -> plt.Axes:
    """Plot thermistor measurements against time.

    Parameters
    ----------
    resistance
        Either a structured array, dictionary of standard arrays, or Resistance object
        from which such data can be read.
    quantity
        The quantity in the array to plot. Must be one of the keys in the dict (or names
        of the array columns), or if `thermistor_temp`, it will be automatically
        calculated from the given input.
    ax
        The axis to make the plot on.
    xlabel
        The xlabel to set.

    Returns
    -------
    ax
        The axis on which the plot was made.
    """
    if ax is None:
        _fig, ax = plt.subplots(1, 1)

    try:
        out = resistance[quantity]
    except (KeyError, ValueError):
        if quantity == "thermistor_temp":
            out = get_temperature_thermistor(resistance["load_resistance"])
        else:
            raise

    ax.plot(out)

    if xlabel:
        ax.set_xlabel(xlabel)

    return ax
