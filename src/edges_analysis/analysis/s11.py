import numpy as np
from edges_io import io
from edges_cal.reflection_coefficient import de_embed
from edges_cal.s11_correction import get_switch_correction
from typing import Sequence, Union
from pathlib import Path


def get_corrected_s11(
    files: Sequence[Union[str, Path]],
    switch_state_dir: [str, Path],
    switch_state_repeat_num: [None, int] = None,
    n_fit_terms: int = 23,
):
    assert len(files) == 4

    standards = [io.S1P.read(fl)[0] for fl in sorted(files)]
    f = io.S1P.read(files[0])[1]

    sw = {
        "o": 1 * np.ones(len(f)),
        "s": -1 * np.ones(len(f)),
        "l": 0 * np.ones(len(f)),
    }

    # Correction at switch
    a_sw_c, x1, x2, x3 = de_embed(sw["o"], sw["s"], sw["l"], *standards)

    switch_state = io.SwitchingState(
        switch_state_dir, repeat_num=switch_state_repeat_num, fix=False
    )

    # Correction at receiver input
    return (
        get_switch_correction(a_sw_c, internal_switch=switch_state, f_in=f, n_terms=n_fit_terms)[0],
        f,
    )


def antenna_s11_remove_delay(
    s11_files: Sequence[Union[str, Path]],
    switch_state_dir: [str, Path],
    f_low: float = -np.inf,
    f_high: float = np.inf,
    delay_0: float = 0.17,
    n_fit: int = 10,
    n_fourier: int = 23,
    switch_state_repeat_num: [int, None] = None,
):
    """
    Remove delay from antenna S11.

    Parameters
    ----------
    s11_files
        Paths to four files with the S11 data in them.
    switch_state_dir
        The directory containing the switching_state measurements to use.
    f_low, f_high
        The min/max frequencies for which to perform the fit.
    delay_0
        Delay of the antenna (at 1 MHz?)
    n_fit
        Number of terms in polynomial fit to the S11, in order to recast at new
        frequencies.
    n_fourier
        Number of terms to use in getting switch correction.
    switch_state_repeat_num
        The repeat number to use when getting the switching state measurements.

    Returns
    -------
    array-like :
        An array of the same shape as `f`, containing the S11 with delay removed.
    """
    gamma, f_orig = get_corrected_s11(
        s11_files,
        switch_state_dir,
        n_fit_terms=n_fourier,
        switch_state_repeat_num=switch_state_repeat_num,
    )

    mask = (f_orig >= f_low) & (f_orig <= f_high)
    gamma = gamma[mask]
    f_orig = f_orig[mask]

    # Removing delay from S11
    delay = delay_0 * f_orig
    gamma *= np.exp(delay * 1j)

    re = np.polyfit(f_orig, np.real(gamma), n_fit - 1)
    im = np.polyfit(f_orig, np.imag(gamma), n_fit - 1)

    def model(f):
        return (np.polyval(re, f) + 1j * np.polyval(im, f)) * np.exp(-1j * delay_0 * f)

    return model, gamma * np.exp(-1j * delay_0 * f_orig), f_orig
