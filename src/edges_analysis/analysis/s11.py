import numpy as np
from edges_cal import modelling as mdl
from edges_io import io


def antenna_s11_remove_delay(s11_path, f, delay_0=0.17, n_fit=10):
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
    gamma, f_orig = io.S1P.read(s11_path)

    f_low = np.min(f)
    f_high = np.max(f)

    if f_orig.min() > f_low:
        raise ValueError("Would be extrapolating beyond low end of frequency.")
    if f_orig.max() < f_high:
        raise ValueError("Would be extrapolating beyond high end of frequency.")

    print(f_low, f_high, f_orig.min(), f_orig.max())
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
