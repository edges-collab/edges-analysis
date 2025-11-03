"""Various plotting functions."""

import matplotlib.pyplot as plt
import numpy as np

from edges.averaging.averaging import bin_array_unweighted

from .calibrator import Calibrator
from .calobs import CalibrationObservation
from .load_data import Load
from .s11 import CalibratedS11, S11ModelParams
from .spectra import LoadSpectrum


def plot_raw_spectrum(
    spectrum: np.ndarray | LoadSpectrum,
    freq: np.ndarray | None = None,
    fig=None,
    ax=None,
    xlabel: bool = True,
    ylabel: bool = True,
    **kwargs,
):
    """
    Make a plot of the averaged uncalibrated spectrum associated with this load.

    Parameters
    ----------
    spectrum
        The LoadSpectrum object to plot.
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
        freq = spectrum.freqs
        spectrum = spectrum.q.data.squeeze()
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


def plot_raw_spectra(calobs: CalibrationObservation, fig=None, ax=None) -> plt.Figure:
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
        ax[i].plot(load.freqs, load.averaged_q)
        ax[i].set_ylabel("$Q$")
        ax[i].set_title(name)
        ax[i].grid(True)
    ax[-1].set_xlabel("Frequency [MHz]")

    return fig


def plot_s11_residual(
    raw_s11: CalibratedS11,
    s11_model_params: S11ModelParams,
    load_name: str | None = None,
    fig=None,
    ax=None,
    color_abs="C0",
    color_diff="g",
    label=None,
    title=None,
    decade_ticks=True,
    ylabels=True,
) -> plt.Figure:
    """
    Plot the residuals of the S11 model compared to un-smoothed corrected data.

    Returns
    -------
    fig :
        Matplotlib Figure handle.
    """
    if ax is None or len(ax) != 4:
        fig, ax = plt.subplots(
            4, 1, sharex=True, gridspec_kw={"hspace": 0.05}, facecolor="w"
        )
    if fig is None:
        fig = ax[0].get_figure()

    if decade_ticks:
        for axx in ax:
            axx.grid(True)
    ax[-1].set_xlabel("Frequency [MHz]")

    modelled = raw_s11.smoothed(s11_model_params)

    fq = raw_s11.freqs

    ax[0].plot(fq, 20 * np.log10(np.abs(modelled.s11)), color=color_abs, label=label)
    if ylabels:
        ax[0].set_ylabel(r"$|S_{11}|$")

    ax[1].plot(fq, np.abs(modelled.s11) - np.abs(raw_s11.s11), color_diff)
    if ylabels:
        ax[1].set_ylabel(r"$\Delta  |S_{11}|$")

    ax[2].plot(fq, np.unwrap(np.angle(modelled.s11)) * 180 / np.pi, color=color_abs)
    if ylabels:
        ax[2].set_ylabel(r"$\angle S_{11}$")

    ax[3].plot(
        fq,
        np.unwrap(np.angle(modelled.s11)) - np.unwrap(np.angle(raw_s11.s11)),
        color_diff,
    )
    if ylabels:
        ax[3].set_ylabel(r"$\Delta \angle S_{11}$")

    lname = load_name or ""

    if title is None:
        title = f"{lname} Reflection Coefficient Models"

    if title:
        fig.suptitle(f"{lname} Reflection Coefficient Models", fontsize=14)
    if label:
        ax[0].legend()

    return fig


def plot_s11_models(
    calobs: CalibrationObservation,
    s11_model_params: S11ModelParams,
    receiver_model_params: S11ModelParams,
    **kwargs,
):
    """
    Plot residuals of S11 models for all sources.

    Returns
    -------
    dict:
        Each entry has a key of the source name, and the value is a matplotlib fig.
    """
    _fig, ax = plt.subplots(
        4,
        len(calobs.loads) + 1,
        figsize=((len(calobs.loads) + 1) * 4, 6),
        sharex=True,
        gridspec_kw={"hspace": 0.05},
        layout="constrained",
    )

    for i, (name, source) in enumerate(calobs.loads.items()):
        plot_s11_residual(
            source._raw_s11,
            s11_model_params=s11_model_params,
            load_name=name,
            ax=ax[:, i],
            title=False,
            **kwargs,
        )
        ax[0, i].set_title(name)

    plot_s11_residual(
        calobs._raw_receiver,
        s11_model_params=receiver_model_params,
        ax=ax[:, -1],
        title=False,
        **kwargs,
    )
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
        f = bin_array_unweighted(calobs.freqs.to_value("MHz"), size=bins)
    else:
        freq_ave_cal = temp_calibrated
        f = calobs.freqs.to_value("MHz")

    freq_ave_cal[np.isinf(freq_ave_cal)] = np.nan

    rms = np.sqrt(np.mean((freq_ave_cal - np.mean(freq_ave_cal)) ** 2))

    ax.plot(
        f,
        freq_ave_cal,
        label=f"Calibrated {load.load_name} [RMS = {rms:.3f}]",
    )

    temp_ave = calobs.source_thermistor_temps.get(load.load_name, load.temp_ave)

    if temp_ave.isscalar:
        temp_ave = np.ones(calobs.freqs.size) * temp_ave

    ax.plot(
        calobs.freqs,
        temp_ave,
        color="C2",
        label="Average thermistor temp",
    )

    ax.set_ylim([np.nanmin(freq_ave_cal).value, np.nanmax(freq_ave_cal).value])
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
    bins: int = 64,
    fig=None,
    ax=None,
    **kwargs,
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
            calobs=calobs,
            calibrator=calibrator,
            load=source,
            bins=bins,
            fig=fig,
            ax=ax[i],
            xlabel=i == (len(calobs.loads) - 1),
        )

    fig.suptitle("Calibrated Temperatures for Calibration Sources", fontsize=15)
    return fig


def plot_cal_coefficients(calibrator: Calibrator, fig=None, ax=None):
    """
    Make a plot of the calibration coefficents, Tsca, Tof, Tunc, Tcos and Tsin.

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
        zip(["Tsca", "Toff", "Tunc", "Tcos", "Tsin"], labels, strict=False)
    ):
        ax[i].plot(calibrator.freqs, getattr(calibrator, kind))
        ax[i].set_ylabel(label, fontsize=13)
        ax[i].grid()
        plt.ticklabel_format(useOffset=False)

        if i == 4:
            ax[i].set_xlabel("Frequency [MHz]", fontsize=13)

    fig.suptitle("Calibration Parameters", fontsize=15)
    return fig
