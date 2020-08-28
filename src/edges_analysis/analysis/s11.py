import numpy as np
from edges_io import io
from edges_cal.reflection_coefficient import de_embed
from edges_cal.s11_correction import get_switch_correction


def get_corrected_s11(files, switch_state_dir, switch_state_run_num=None, n_fit_terms=23):
    assert len(files) == 4

    standards = [io.S1P.read(fl)[0] for fl in files]
    f = io.S1P.read(files[0])[1]

    sw = {
        "o": 1 * np.ones(len(f)),
        "s": -1 * np.ones(len(f)),
        "l": 0 * np.ones(len(f)),
    }

    # Correction at switch
    a_sw_c, x1, x2, x3 = de_embed(sw["o"], sw["s"], sw["l"], *standards)

    switch_state = io.SwitchingState(switch_state_dir, run_num=switch_state_run_num, fix=False)

    # Correction at receiver input
    return (
        get_switch_correction(a_sw_c, internal_switch=switch_state, f_in=f, poly_order=n_fit_terms)[
            0
        ],
        f,
    )


def antenna_s11_remove_delay(
    s11_files,
    f,
    switch_state_dir,
    delay_0=0.17,
    n_fit=10,
    n_fourier=23,
    switch_state_run_num=None,
):
    """
    Remove delay from antenna S11.

    Parameters
    ----------
    s11_path : str
        Path to a file with the S11 data in it.
    f : array-like
        Frequencies at which to return *output*, in MHz.
    delay_0 : float, optional
        Delay of the antenna (at 1 MHz?)
    n_fit : int, optional
        Number of terms in polynomial fit to the S11, in order to recast at new
        frequencies.

    Returns
    -------
    array-like :
        An array of the same shape as `f`, containing the S11 with delay removed.
    """
    gamma, f_orig = get_corrected_s11(
        s11_files,
        switch_state_dir,
        n_fit_terms=n_fourier,
        switch_state_run_num=switch_state_run_num,
    )

    f_low = np.min(f)
    f_high = np.max(f)

    if f_orig.min() > f_low:
        raise ValueError("Would be extrapolating beyond low end of frequency.")
    if f_orig.max() < f_high:
        raise ValueError("Would be extrapolating beyond high end of frequency.")

    mask = (f_orig >= f_low) & (f_orig <= f_high)
    gamma = gamma[mask]
    f_orig = f_orig[mask]

    # Removing delay from S11
    delay = delay_0 * f_orig

    def get_model(fnc):
        re_wd = np.abs(gamma) * fnc(delay + np.unwrap(np.angle(gamma)))
        par_re_wd = np.polyfit(f_orig, re_wd, n_fit - 1)
        return np.polyval(par_re_wd, f)

    model_re_wd = get_model(np.cos)
    model_im_wd = get_model(np.sin)

    model_s11_wd = model_re_wd + 1j * model_im_wd
    return model_s11_wd * np.exp(-1j * delay_0 * f)
