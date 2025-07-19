"""Models for loss through cables."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property

import attrs
import numpy as np
from astropy import units as un
from numpy import typing as npt
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from .. import frequencies as fqtools
from .. import modelling as mdl
from .. import types as tp
from . import ee, get_data_path
from . import reflection_coefficient as rc


def compute_cable_loss_from_scattering_params(
    s11a: npt.NDArray[complex], smatrix: rc.SMatrix
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
    s12 = smatrix.s12
    s21 = smatrix.s21
    s22 = smatrix.s22

    T = rc.gamma_de_embed(s11a, smatrix)
    return (
        np.abs(s12 * s21)
        * (1 - np.abs(T) ** 2)
        / ((1 - np.abs(s11a) ** 2) * np.abs(1 - s22 * T) ** 2)
    )


def get_cable_loss_model(
    cable: ee.CoaxialCable | str | Sequence[ee.CoaxialCable | str],
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
    if isinstance(cable, str):
        cable = ee.KNOWN_CABLES[cable]

    if isinstance(cable, ee.CoaxialCable):
        cable = [cable]

    cable = [c if isinstance(c, ee.CoaxialCable) else ee.KNOWN_CABLES[c] for c in cable]

    def loss_model(freq, s11a):
        s0 = cable[0].scattering_parameters(freq)
        if len(cable) > 1:
            for cbl in cable[1:]:
                ss = cbl.scattering_parameters(freq)
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
    return lambda freq, s11a: spl(freq)


@attrs.define(slots=False, frozen=True, kw_only=True)
class HotLoadCorrection:
    """
    Corrections for the hot load.

    Measurements required to define the HotLoad temperature, from Monsalve et al.
    (2017), Eq. 8+9.

    Parameters
    ----------
    path
        Path to a file containing measurements of the semi-rigid cable reflection
        parameters. A preceding colon (:) indicates to prefix with DATA_PATH.
        The default file was measured in 2015, but there is also a file included
        that can be used from 2017: ":semi_rigid_s_parameters_2017.txt".
    f_low, f_high
        Lowest/highest frequency to retain from measurements.
    n_terms
        The number of terms used in fitting S-parameters of the cable.
    """

    freq: tp.FreqType = attrs.field()
    raw_s11: np.ndarray = attrs.field(eq=attrs.cmp_using(eq=np.array_equal))
    raw_s12s21: np.ndarray = attrs.field(eq=attrs.cmp_using(eq=np.array_equal))
    raw_s22: np.ndarray = attrs.field(eq=attrs.cmp_using(eq=np.array_equal))

    model: mdl.Model = attrs.field(default=mdl.Polynomial(n_terms=21))
    complex_model: type[mdl.ComplexRealImagModel] | type[mdl.ComplexMagPhaseModel] = (
        attrs.field(default=mdl.ComplexMagPhaseModel)
    )
    use_spline: bool = attrs.field(default=False)
    model_method: str = attrs.field(default="lstsq")

    @classmethod
    def from_file(
        cls,
        path: tp.PathLike = ":semi_rigid_s_parameters_WITH_HEADER.txt",
        f_low: tp.FreqType = 0 * un.MHz,
        f_high: tp.FreqType = np.inf * un.MHz,
        set_transform_range: bool = True,
        **kwargs,
    ):
        """Instantiate the HotLoadCorrection from file.

        Parameters
        ----------
        path
            Path to the S-parameters file.
        f_low, f_high
            The min/max frequencies to use in the modelling.
        """
        path = get_data_path(path)

        data = np.genfromtxt(path)
        mask = fqtools.get_mask(data[:, 0] * un.MHz, low=f_low, high=f_high)
        data = data[mask]
        freq = data[:, 0] * un.MHz

        if data.shape[1] == 7:  # Original file from 2015
            data = data[:, 1::2] + 1j * data[:, 2::2]
        elif data.shape[1] == 6:  # File from 2017
            data = np.array([
                data[:, 1] + 1j * data[:, 2],
                data[:, 3],
                data[:, 4] + 1j * data[:, 5],
            ]).T

        model = kwargs.pop(
            "model",
            mdl.Polynomial(
                n_terms=21,
                transform=mdl.UnitTransform(
                    range=(freq.min().to_value("MHz"), freq.max().to_value("MHz"))
                ),
            ),
        )

        if hasattr(model.xtransform, "range") and set_transform_range:
            model = attrs.evolve(
                model,
                transform=attrs.evolve(
                    model.xtransform,
                    range=(freq.min().to_value("MHz"), freq.max().to_value("MHz")),
                ),
            )

        return cls(
            freq=freq,
            raw_s11=data[:, 0],
            raw_s12s21=data[:, 1],
            raw_s22=data[:, 2],
            model=model,
            **kwargs,
        )

    def _get_model(self, raw_data: np.ndarray, **kwargs):
        model = self.complex_model(self.model, self.model)
        return model.fit(
            xdata=self.freq.to_value("MHz"),
            ydata=raw_data,
            method=self.model_method,
        )

    def _get_splines(self, data):
        if self.complex_model == mdl.ComplexRealImagModel:
            return (
                Spline(self.freq.to_value("MHz"), np.real(data)),
                Spline(self.freq.to_value("MHz"), np.imag(data)),
            )
        return (
            Spline(self.freq.to_value("MHz"), np.abs(data)),
            Spline(self.freq.to_value("MHz"), np.angle(data)),
        )

    def _ev_splines(self, splines):
        rl, im = splines
        if self.complex_model == mdl.ComplexRealImagModel:
            return lambda freq: rl(freq) + 1j * im(freq)
        return lambda freq: rl(freq) * np.exp(1j * im(freq))

    @cached_property
    def s11_model(self):
        """The reflection coefficient."""
        if not self.use_spline:
            return self._get_model(self.raw_s11)
        splines = self._get_splines(self.raw_s11)
        return self._ev_splines(splines)

    @cached_property
    def s12s21_model(self):
        """The transmission coefficient."""
        if not self.use_spline:
            return self._get_model(self.raw_s12s21)
        splines = self._get_splines(self.raw_s12s21)
        return self._ev_splines(splines)

    @cached_property
    def s22_model(self):
        """The reflection coefficient from the other side."""
        if not self.use_spline:
            return self._get_model(self.raw_s22)
        splines = self._get_splines(self.raw_s22)
        return self._ev_splines(splines)

    def power_gain(self, freq: tp.FreqType, hot_load_s11: np.ndarray) -> np.ndarray:
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
        nu = freq.to_value("MHz")
        s12 = np.sqrt(self.s12s21_model(nu))

        smatrix = rc.SMatrix(
            np.array([[self.s11_model(nu), s12], [s12, self.s22_model(nu)]])
        )
        return compute_cable_loss_from_scattering_params(hot_load_s11, smatrix)
