"""Models for loss through cables."""

from collections.abc import Sequence

import attrs
import numpy as np
from numpy import typing as npt
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from . import sparams as sp


def compute_cable_loss_from_scattering_params(
    input_s11: sp.ReflectionCoefficient, sparams: sp.SParams
) -> npt.NDArray[float]:
    """Compute loss from a cable, given the S-params of the cable, and input S11.

    This function operates in the context of a cable (or balun, etc.) that is attached
    to an input (generally the antenna, or a calibration load). It computes the loss
    due to the cable, given the scattering matrix of the cable itself, and the S11
    of the input antenna/load.

    The equations here are described in MIT EDGES Memo #132:
    https://www.haystack.mit.edu/wp-content/uploads/2020/07/memo_EDGES_132.pdf.
    We use two of the equations, to be specific: the equation for Gamma_a (the antenna
    reflection coefficient at the input of the cable), and the equation for the loss,
    L.

    SGM: as far as I can tell, this function *doesn't* assume that S12 == S21, though
    actual calls to this function generally throughout our calibration do make this
    assumption.
    """
    s12 = sparams.s12
    s21 = sparams.s21
    s22 = sparams.s22

    T = input_s11.de_embed(sparams).reflection_coefficient

    return (
        np.abs(s12 * s21)
        * (1 - np.abs(T) ** 2)
        / (
            (1 - np.abs(input_s11.reflection_coefficient) ** 2)
            * np.abs(1 - s22 * T) ** 2
        )
    )


def get_cable_loss_model(
    cable: sp.CoaxialCable | str | Sequence[sp.CoaxialCable | str],
) -> callable:
    """Return a callable loss model for a particular cable or series of cables.

    The returned function is suitable for passing to a :class:`Load`
    as the loss_model.

    You can pass a single cable (i.e. a :class:`edges.cal.ee.CoaxialCable`) or a list
    of such cables, each of which is assumed to be joined in a cascade. Each should
    be equipped with a cable length.

    Parameters
    ----------
    cable
        Either a string, or a CoaxialCable instance, or a list of such. If a string,
        it should be a name present in `ee.KNOWN_CABLES`.
    """
    if isinstance(cable, sp.CoaxialCable | str):
        cable = [cable]

    cable = [c if isinstance(c, sp.CoaxialCable) else sp.KNOWN_CABLES[c] for c in cable]

    def loss_model(s11a: sp.ReflectionCoefficient) -> npt.NDArray:
        s0 = cable[0].scattering_parameters(s11a.freqs)
        if len(cable) > 1:
            for cbl in cable[1:]:
                ss = cbl.scattering_parameters(s11a.freqs)
                s0 = s0.cascade_with(ss)

        return compute_cable_loss_from_scattering_params(s11a, s0)

    return loss_model


def get_loss_model_from_file(fname):
    """Simply read a loss model directly from a file.

    The file must have two columns separated by whitespace. The first is frequency
    in MHz and the second should be the loss.
    """
    with fname.open("r") as fl:
        data = np.genfromtxt(fl)

    spl = Spline(data[:, 0], data[:, 1])
    return lambda s11a: spl(s11a.freqs)


@attrs.define(slots=False, frozen=True)
class LossFunctionGivenSparams:
    """
    A callable that satisfies the signature for a loss function, from given sparams.

    Measurements required to define the HotLoad temperature, from Monsalve et al.
    (2017), Eq. 8+9.

    """

    sparams: sp.SParams = attrs.field(
        validator=attrs.validators.instance_of(sp.SParams)
    )

    def __call__(self, gamma: sp.ReflectionCoefficient) -> np.ndarray:
        """
        Calculate the power gain.

        Parameters
        ----------
        freq : np.ndarray
            The frequencies.
        hot_load_s11 : array
            The S11 of the hot load.

        Returns
        -------
        gain : np.ndarray
            The power gain as a function of frequency.
        """
        if self.sparams.freqs.size != len(gamma.freqs):
            raise ValueError(
                "Given gamma doesn't have the same length as the S-params."
            )

        return compute_cable_loss_from_scattering_params(gamma, self.sparams)
