"""Tools for dealing with frequency arrays."""

import numpy as np
from astropy import units
from numpy.typing import NDArray

from . import types as tp


@units.quantity_input
def edges_raw_freqs(
    f_low: tp.FreqType = 0 * units.MHz,
    f_high: tp.FreqType = np.inf * units.MHz,
) -> units.Quantity[units.MHz]:
    """Get the raw frequency array of the EDGES spectrometer.

    Parameters
    ----------
    f_low, f_high
        A frequency range to keep.

    Returns
    -------
    freqs
        The raw frequencies of the spectrometer.

    Notes
    -----
    This is correct. The channel width is the important thing.
    The channel width is given by the FFT. We actually take
    32678*2 samples of data at 400 Mega-samples per second.
    We only use the first half of the samples (since it's real input).
    Regardless, the frequency channel width is thus
    400 MHz / (32678*2) == 200 MHz / 32678 ~ 6.103 kHz

    """
    n_channels: int = 16384 * 2
    max_freq: float = 200.0  # MHz

    df = max_freq / n_channels

    # The final frequency here will be slightly less than 200 MHz. 200 MHz
    # corresponds to the centre of the N+1 bin, which doesn't actually exist.
    f = np.arange(0, max_freq, df) * units.MHz

    return clip_freqs(f, f_low, f_high)


def get_mask(
    freq: tp.FreqType,
    low: tp.FreqType = 0 * units.MHz,
    high: tp.FreqType = np.inf * units.MHz,
) -> NDArray[np.bool]:
    """Get a mask from a frequency array between a given range."""
    return (freq >= low) & (freq <= high)


@units.quantity_input
def edges_freq_mask(low: tp.FreqType, high: tp.FreqType) -> NDArray[np.bool]:
    """Create a mask for the raw EDGES spectrum frequencies."""
    raw = edges_raw_freqs()
    return get_mask(raw, low, high)


@units.quantity_input
def clip_freqs(
    freq: tp.FreqType, low: tp.FreqType, high: tp.FreqType
) -> units.Quantity[units.MHz]:
    """Clip an array of frequencies within a range."""
    return freq[(freq >= low) & (freq <= high)]
