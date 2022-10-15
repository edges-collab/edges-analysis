"""Plotting utilities."""
from __future__ import annotations

import edges_cal.modelling as mdl
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as apu

from .averaging import averaging
from .averaging.lstbin import lst_bin
from .gsdata import GSData, add_model


def plot_waterfall(
    data: GSData,
    load: int = 0,
    pol: int = 0,
    which_flags: tuple[str] = None,
    ignore_flags: tuple[str] = (),
    ax: plt.Axes | None = None,
    cbar: bool = True,
    xlab: bool = True,
    ylab: bool = True,
    title: bool | str = True,
    attribute: str = "spectra",
    **imshow_kwargs,
):
    """Plot a waterfall from a GSData object.

    Parameters
    ----------
    data
        The GSData object to plot.
    load
        The index of the load to plot (only one load is plotted).
    pol
        The polarization to plot (only one polarization is plotted).
    which_flags
        A tuple of flag names to use in order to mask the data. If None, all flags are
        used. Send an empty tuple to ignore all flags.
    ignore_flags
        A tuple of flag names to ignore.
    ax
        The axis to plot on. If None, a new axis is created.
    cbar
        Whether to plot a colorbar.
    xlab
        Whether to plot an x-axis label.
    ylab
        Whether to plot a y-axis label.
    title
        Whether to plot a title. If True, the title is the year-day representation
        of the dataset. If a string, use that as the title.
    attribute
        The attribute to actually plot. Can be any attribute of the data object that has
        the same array shape as the primary data array. This includes "data", "spectra",
        "resids", "complete_flags", "nsamples".
    """
    q = getattr(data, attribute)
    if q.shape != data.data.shape:
        raise ValueError(
            f"Cannot use attribute '{attribute}' as it doesn't have "
            "the same shape as data."
        )

    q = np.where(data.get_flagged_nsamples(which_flags, ignore_flags) == 0, np.nan, q)
    q = q[load, pol, :, :]

    if ax is None:
        ax = plt.subplots(1, 1)[1]

    if attribute == "resids":
        cmap = imshow_kwargs.pop("cmap", "coolwarm")
    else:
        cmap = imshow_kwargs.pop("cmap", "magma")

    times = data.time_array

    img = ax.imshow(
        q,
        origin="lower",
        extent=(
            data.freq_array.min().to_value("MHz"),
            data.freq_array.max().to_value("MHz"),
            0,
            (
                (times.max() - times.min()).hour
                if data.in_lst
                else (times.max() - times.min()).to_value("hour")
            ),
        ),
        cmap=cmap,
        aspect="auto",
        interpolation="none",
        **imshow_kwargs,
    )

    if xlab:
        ax.set_xlabel("Frequency [MHz]")
    if ylab:
        if data.in_lst:
            ax.set_ylabel("LST")
        else:
            ax.set_ylabel("Hours into Observation")

    if title and not isinstance(title, str):
        if not data.in_lst:
            ax.set_title(
                f"{data.get_initial_yearday()}. LST0={data.lst_array[0][0]:.2f}"
            )

    if cbar:
        cb = plt.colorbar(img, ax=ax)
        cb.set_label(data.loads[load])

    return ax


def plot_time_average(
    data: GSData,
    ax: plt.Axes | None = None,
    logy=None,
    lst_min: float = 0,
    lst_max: float = 24,
    load: int = 0,
    pol: int = 0,
    attribute: str = "spectra",
    offset: float = 0.0,
):
    """Make a 1D plot of the time-averaged data.

    Parameters
    ----------
    data
        The GSData object to plot.
    ax
        The axis to plot on. If None, a new axis is created.
    logy
        Whether to plot a logarithmic y-axis. If None, the y-axis is logarithmic if all
        the plotted data is positive.
    lst_min
        The minimum LST to average together.
    lst_max
        The maximum LST to average together.
    load
        The index of the load to plot (only one load is plotted).
    pol
        The polarization to plot (only one polarization is plotted).
    attribute
        The attribute to actually plot. Can be any attribute of the data object that has
        the same array shape as the primary data array. This includes "data", "spectra",
        "resids", "complete_flags", "nsamples".
    offset
        The offset to add to the data before plotting. Useful if plotting multiple
        averages on the same axis.
    """
    if ax is not None:
        fig = ax.figure
    else:
        fig, ax = plt.subplots(1, 1)

    if lst_min > 0 or lst_max < 24:
        data = data.select_times(range=(lst_min, lst_max))

    data = lst_bin(data, binsize=24.0)

    q = getattr(data, attribute)
    if q.shape != data.data.shape:
        raise ValueError(
            f"Cannot use attribute '{attribute}' as it doesn't "
            "have the same shape as data."
        )

    ax.plot(data.freq_array, q[load, pol, 0] - offset)
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Average Spectrum")

    if logy is None:
        logy = np.all(q > 0)

    if logy:
        ax.set_yscale("log")

    return ax, data


def plot_daily_residuals(
    objs: list[GSData],
    model: mdl.Model | None = None,
    separation: float = 20.0,
    ax: plt.Axes | None = None,
    load: int = 0,
    pol: int = 0,
    **kw,
) -> plt.Axes:
    """
    Make a single plot of residuals for each object.

    Parameters
    ----------
    objs
        A list of objects to plot.
    separation
        The separation between residuals in K (on the plot).

    Other Parameters
    ----------------
    All other parameters are passed through to :func:`plot_time_average`.

    Returns
    -------
    ax
        The matplotlib Axes on which the plot is made.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    for i, data in enumerate(objs):
        if data.data_model is None and model is None:
            raise ValueError("If data has no model, must provide one!")

        if data.data_model is None:
            data = add_model(data, model=model)

        ax, d = plot_time_average(
            data, attribute="resids", offset=separation * i, ax=ax, **kw
        )

        rms = np.sqrt(
            averaging.weighted_mean(
                data=d.resids[load, pol, 0] ** 2, weights=d.nsamples[load, pol, 0]
            )[0]
        )
        title = data.filename.name if data.in_lst else data.get_initial_yearday()
        ax.text(
            data.freq_array.max() + 5 * apu.MHz,
            -i * separation,
            f"{title} RMS={rms:.2f}",
        )

    return ax
