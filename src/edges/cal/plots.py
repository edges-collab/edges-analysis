"""Various plotting functions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .calobs import CalibrationObservation
from .calibrator import Calibrator
from .load_data import Load
from .spectra import LoadSpectrum
from astropy.table import QTable
from .thermistor import get_temperature_thermistor
from edges.averaging.averaging import bin_array_unweighted


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
    if isinstance(spectrum, LoadSpectrum):
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

def plot_raw_spectra(
    calobs: CalibrationObservation, 
    fig=None, ax=None
) -> plt.Figure:
    """
    Plot raw uncalibrated spectra for all calibrator sources.

    Parameters
    ----------
    fig : :class:`plt.Figure`
        A matplotlib figure on which to make the plot. By default creates a new one.
    ax : :class:`plt.Axes`
        A matplotlib Axes on which to make the plot. By default creates a new one.

    Returns
    -------
    fig : :class:`plt.Figure`
        The figure on which the plot was made.
    """
    if fig is None and ax is None:
        fig, ax = plt.subplots(
            len(calobs.loads), 1, sharex=True, gridspec_kw={"hspace": 0.05}
        )

    for i, (name, load) in enumerate(calobs.loads.items()):
        ax[i].plot(load.freq, calobs.averaged_spectrum(load))
        ax[i].set_ylabel("$T^*$ [K]")
        ax[i].set_title(name)
        ax[i].grid(True)
    ax[-1].set_xlabel("Frequency [MHz]")

    return fig

def plot_s11_models(calobs: CalibrationObservation, **kwargs):
    """
    Plot residuals of S11 models for all sources.

    Returns
    -------
    dict:
        Each entry has a key of the source name, and the value is a matplotlib fig.
    """
    fig, ax = plt.subplots(
        4,
        len(calobs.loads) + 1,
        figsize=((len(calobs.loads) + 1) * 4, 6),
        sharex=True,
        gridspec_kw={"hspace": 0.05},
        layout="constrained",
    )

    for i, (name, source) in enumerate(calobs.loads.items()):
        source.reflections.plot_residuals(ax=ax[:, i], title=False, **kwargs)
        ax[0, i].set_title(name)

    calobs.receiver.plot_residuals(ax=ax[:, -1], title=False, **kwargs)
    ax[0, -1].set_title("Receiver")
    return ax


def plot_calibrated_temp(
    calobs: CalibrationObservation,
    calibrator: Calibrator,
    load: Load | str,
    bins: int = 2,
    fig=None,
    ax=None,
    xlabel=True,
    ylabel=True,
    label: str = "",
    as_residuals: bool = False,
    load_in_title: bool = False,
    rms_in_label: bool = True,
):
    """
    Make a plot of calibrated temperature for a given source.

    Parameters
    ----------
    load : :class:`~LoadSpectrum` instance
        Source to plot.
    bins : int
        Number of bins to smooth over (std of Gaussian kernel)
    fig : Figure
        Optionally provide a matplotlib figure to add to.
    ax : Axis
        Optionally provide a matplotlib Axis to add to.
    xlabel : bool
        Whether to write the x-axis label
    ylabel : bool
        Whether to write the y-axis label

    Returns
    -------
    fig :
        The matplotlib figure that was created.
    """
    load = calobs._load_str_to_load(load)

    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1, facecolor="w")

    # binning
    temp_calibrated = calibrator.calibrate_load(load)
    if bins > 0:
        freq_ave_cal = bin_array_unweighted(temp_calibrated, size=bins)
        f = bin_array_unweighted(calobs.freq.to_value("MHz"), size=bins)
    else:
        freq_ave_cal = temp_calibrated
        f = calobs.freq.to_value("MHz")

    freq_ave_cal[np.isinf(freq_ave_cal)] = np.nan

    rms = np.sqrt(np.mean((freq_ave_cal - np.mean(freq_ave_cal)) ** 2))

    ax.plot(
        f,
        freq_ave_cal,
        label=f"Calibrated {load.load_name} [RMS = {rms:.3f}]",
    )

    temp_ave = calobs.source_thermistor_temps.get(load.load_name, load.temp_ave)

    if np.isscalar(temp_ave):
        temp_ave = np.ones(calobs.freq.size) * temp_ave

    ax.plot(
        calobs.freq,
        temp_ave,
        color="C2",
        label="Average thermistor temp",
    )

    ax.set_ylim([np.nanmin(freq_ave_cal), np.nanmax(freq_ave_cal)])
    if xlabel:
        ax.set_xlabel("Frequency [MHz]")

    if ylabel:
        ax.set_ylabel("Temperature [K]")

    plt.ticklabel_format(useOffset=False)
    ax.grid()
    ax.legend()

    return plt.gcf()


def plot_calibrated_temps(
    calobs: CalibrationObservation, 
    calibrator: Calibrator,
    bins: int = 64, fig=None, ax=None, **kwargs
):
    """
    Plot all calibrated temperatures in a single figure.

    Parameters
    ----------
    bins : int
        Number of bins in the smoothed spectrum

    Returns
    -------
    fig :
        Matplotlib figure that was created.
    """
    if fig is None or ax is None or len(ax) != len(calobs.loads):
        fig, ax = plt.subplots(
            len(calobs.loads),
            1,
            sharex=True,
            gridspec_kw={"hspace": 0.05},
            figsize=(10, 12),
        )

    for i, source in enumerate(calobs.loads):
        plot_calibrated_temp(
            source,
            calibrator=calibrator,
            bins=bins,
            fig=fig,
            ax=ax[i],
            xlabel=i == (len(calobs.loads) - 1),
        )

    fig.suptitle("Calibrated Temperatures for Calibration Sources", fontsize=15)
    return fig


def plot_cal_coefficients(
    calibrator: Calibrator, 
    fig=None, 
    ax=None
):
    """
    Make a plot of the calibration models, C1, C2, Tunc, Tcos and Tsin.

    Parameters
    ----------
    fig : Figure
        Optionally pass a matplotlib figure to add to.
    ax : Axis
        Optionally pass a matplotlib axis to pass to. Must have 5 axes.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(
            5, 1, facecolor="w", gridspec_kw={"hspace": 0.05}, figsize=(10, 9)
        )

    labels = [
        "Scale ($C_1$)",
        "Offset ($C_2$) [K]",
        r"$T_{\rm unc}$ [K]",
        r"$T_{\rm cos}$ [K]",
        r"$T_{\rm sin}$ [K]",
    ]
    for i, (kind, label) in enumerate(
        zip(["C1", "C2", "Tunc", "Tcos", "Tsin"], labels, strict=False)
    ):
        ax[i].plot(calibrator.freq, getattr(calibrator, kind)())
        ax[i].set_ylabel(label, fontsize=13)
        ax[i].grid()
        plt.ticklabel_format(useOffset=False)

        if i == 4:
            ax[i].set_xlabel("Frequency [MHz]", fontsize=13)

    fig.suptitle("Calibration Parameters", fontsize=15)
    return fig