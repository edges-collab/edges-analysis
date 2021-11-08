"""Corrections for S11 measurements."""
import numpy as np
from edges_io import io
from edges_cal.s11_correction import InternalSwitch
from typing import Sequence, Union, Callable, Optional
from pathlib import Path
from edges_cal import reflection_coefficient as rc
from edges_cal import modelling as mdl


def get_corrected_s11(
    files: Sequence[Union[str, Path]],
    internal_switch: Optional[InternalSwitch] = None,
    internal_switch_s11: Optional[Callable] = None,
    internal_switch_s12: Optional[Callable] = None,
    internal_switch_s22: Optional[Callable] = None,
):
    """Correct a measured S11 with the internal switch."""
    if len(files) == 4:
        if internal_switch is None:
            assert internal_switch_s11 is not None
            assert internal_switch_s12 is not None
            assert internal_switch_s22 is not None
        else:
            assert isinstance(internal_switch, InternalSwitch)
            internal_switch_s11 = internal_switch.s11_model
            internal_switch_s12 = internal_switch.s12_model
            internal_switch_s22 = internal_switch.s22_model

        standards = [io.S1P.read(fl)[0] for fl in sorted(files)]
        f = io.S1P.read(files[0])[1]

        sw = {
            "o": 1 * np.ones(len(f)),
            "s": -1 * np.ones(len(f)),
            "l": 0 * np.ones(len(f)),
        }

        # Correction at switch
        a_sw_c = rc.de_embed(sw["o"], sw["s"], sw["l"], *standards)[0]

        return (
            rc.gamma_de_embed(
                internal_switch_s11(f),
                internal_switch_s12(f),
                internal_switch_s22(f),
                a_sw_c,
            ),
            f,
        )
    elif len(files) == 1:
        return get_s11_from_file(files[0])


def get_s11_from_file(s11_file_name):
    """Function to read the csv file that has the corrected S11."""
    f_orig, gamm_real, gamma_imag = np.loadtxt(
        s11_file_name, skiprows=1, delimiter=",", unpack=True
    )
    return gamm_real + 1j * gamma_imag, f_orig / 10 ** 6


def antenna_s11_remove_delay(
    s11_files: Sequence[Union[str, Path]],
    f_low: float = -np.inf,
    f_high: float = np.inf,
    delay_0: float = 0.17,
    internal_switch: Optional[InternalSwitch] = None,
    internal_switch_s11: Optional[Callable] = None,
    internal_switch_s12: Optional[Callable] = None,
    internal_switch_s22: Optional[Callable] = None,
    model: mdl.Model = None,
):
    """
    Remove delay from antenna S11.

    Parameters
    ----------
    s11_files
        Paths to four files with the S11 data in them.
    f_low, f_high
        The min/max frequencies for which to perform the fit.
    delay_0
        Delay of the antenna (at 1 MHz?)
    model
        The model to fit to the real and imaginary parts of the antenna S11
        (separately).

    Returns
    -------
    array-like :
        An array of the same shape as `f`, containing the S11 with delay removed.
    """
    gamma, f_orig = get_corrected_s11(
        s11_files,
        internal_switch,
        internal_switch_s11,
        internal_switch_s12,
        internal_switch_s22,
    )

    mask = (f_orig >= f_low) & (f_orig <= f_high)
    gamma = gamma[mask]
    f_orig = f_orig[mask]

    # Removing delay from S11
    delay = delay_0 * f_orig
    gamma *= np.exp(delay * 1j)

    if model is None:
        model = mdl.Polynomial(
            n_terms=10, transform=mdl.UnitTransform(range=(f_orig.min(), f_orig.max()))
        )

    model = mdl.ComplexRealImagModel(real=model, imag=model)
    fit = model.fit(xdata=f_orig, ydata=gamma)

    def model(f):
        return fit(x=f) * np.exp(-1j * delay_0 * f)

    return model, gamma * np.exp(-1j * delay_0 * f_orig), f_orig
