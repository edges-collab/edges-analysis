"""Plotting utilities."""

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as apu
from pygsdata import GSData
from pygsdata.select import select_lsts

from .. import modeling as mdl
from ..averaging import averaging
from ..averaging.lstbin import average_over_times
from .datamodel import add_model


def plot_time_average(
    data: GSData,
    ax: plt.Axes | None = None,
    logy=None,
    lst_min: float = 0,
    lst_max: float = 24,
    load: int = 0,
    pol: int = 0,
    attribute: str = "data",
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
        the same array shape as the primary data array. This includes "data",
        "residuals", "complete_flags", "nsamples".
    offset
        The offset to add to the data before plotting. Useful if plotting multiple
        averages on the same axis.
    """
    if ax is not None:
        pass
    else:
        _fig, ax = plt.subplots(1, 1)

    if lst_min > 0 or lst_max < 24:
        data = select_lsts(data, lst_range=(lst_min, lst_max))
    data = average_over_times(data)

    q = getattr(data, attribute)
    if not hasattr(q, "shape") or q.shape != data.data.shape:
        raise ValueError(
            f"Cannot use attribute '{attribute}' as it doesn't "
            "have the same shape as data."
        )

    ax.plot(data.freqs, q[load, pol, 0] - offset)
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
        _fig, ax = plt.subplots(1, 1)

    for i, data in enumerate(objs):
        if data.residuals is None and model is None:
            raise ValueError("If data has no model, must provide one!")

        if data.residuals is None:
            data = add_model(data, model=model)

        ax, d = plot_time_average(
            data, attribute="residuals", offset=separation * i, ax=ax, **kw
        )

        rms = np.sqrt(
            averaging.weighted_mean(
                data=d.residuals[load, pol, 0] ** 2, weights=d.nsamples[load, pol, 0]
            )[0]
        )
        title = data.get_initial_yearday()
        ax.text(
            data.freqs.max() + 5 * apu.MHz,
            -i * separation,
            f"{title} RMS={rms:.2f}",
        )

    return ax
