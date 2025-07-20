"""Classes representing S11 measurements.

There are classes here for S11s of external loads, the Receiver and also the
Internal Switch (necessary to correct the S11 of external loads).

The S11s of each "device" are assumed to be measured with a VNA and calibrated
using a Calkit containing Open/Short/Load standards. The formalism for this calibration
is defined in Monsalve et al., 2016. Methods for performing this S11 calibration are
in the :mod:`~.reflection_coefficient` module.

We attempt to keep the interface to each of the devices relatively consistent. Each
provides a `s11_model` method which is a function of frequency, outputting the
calibrated and smoothed S11, according to some smooth model.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
from typing import Any, ClassVar, Self

import attrs
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as un
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from edges.io import calobsdef

from ... import types as tp
from ...io.serialization import hickleable
from ...modelling import (
    ComplexMagPhaseModel,
    ComplexRealImagModel,
    Fourier,
    Model,
    Modelable,
    UnitTransform,
    XTransform,
    get_mdl,
)
from .. import reflection_coefficient as rc
from edges.frequencies import get_mask


@hickleable
@attrs.define(kw_only=True, frozen=True)
class S11Model:
    """
    An abstract base class for representing calibrated S11 measurements of a device.

    Parameters
    ----------
    device
        An instance of the basic ``io`` S11 folder.
    f_low : float
        Minimum frequency to use. Default is all frequencies.
    f_high : float
        Maximum frequency to use. Default is all frequencies.
    n_terms : int
        The number of terms to use in fitting a model to the S11 (used to both
        smooth and interpolate the data). Must be odd.
    """

    _default_nterms: ClassVar[int] = 55
    _complex_model_type_default: ClassVar[type[Model]] = ComplexMagPhaseModel
    _model_type_default: ClassVar[type[Model]] = Fourier

    raw_s11: np.ndarray = attrs.field(eq=attrs.cmp_using(eq=np.array_equal))
    freq: tp.FreqType = attrs.field()

    n_terms: int = attrs.field(converter=int)
    model_type: Modelable = attrs.field()
    complex_model_type: type[ComplexMagPhaseModel] | type[ComplexRealImagModel] = (
        attrs.field()
    )
    model_delay: tp.TimeType = attrs.field(default=0 * un.s)
    model_transform: XTransform = attrs.field(default=UnitTransform(range=(0, 1)))
    set_transform_range: bool = attrs.field(default=True, converter=bool)
    model_kwargs: dict[str, Any] = attrs.field(factory=dict)
    use_spline: bool = attrs.field(default=False)
    metadata: dict = attrs.field(factory=dict, eq=False)
    fit_kwargs: dict[str, Any] = attrs.field(factory=dict)

    @freq.validator
    def _fv(self, att, val):
        if val.size != len(self.raw_s11):
            raise ValueError(
                f"len(freq) != len(raw_s11) [{len(val)},{len(self.raw_s11)}]"
            )

    @model_type.default
    def _mdl_type_default(self):
        return self._model_type_default

    @complex_model_type.default
    def _cmt_default(self):
        return self._complex_model_type_default

    @n_terms.default
    def _nt_default(self):
        return int(self._default_nterms)

    @n_terms.validator
    def _nt_vld(self, att, val):
        if get_mdl(self.model_type) == Fourier and not val % 2:
            raise ValueError(
                f"n_terms must be odd for Fourier models. For {self} got n_terms={val}."
            )

    def clone(self, **kwargs):
        """Clone with new parameters."""
        return attrs.evolve(self, **kwargs)

    def get_s11_model(
        self,
        raw_s11: np.ndarray,
        *,
        freq: tp.FreqType | None = None,
        n_terms: int | None = None,
        model_type: Modelable | None = None,
    ) -> ComplexMagPhaseModel | ComplexRealImagModel:
        """Generate a callable model for the S11.

        This should closely match :meth:`s11_correction`.

        Parameters
        ----------
        raw_s11
            The raw s11 of the

        Returns
        -------
        callable :
            A function of one argument, f, which should be a frequency in the same units
            as `self.freq`.

        Raises
        ------
        ValueError
            If n_terms is not an integer, or not odd.
        """
        if freq is None:
            freq = self.freq

        n_terms = n_terms or self.n_terms
        model_type = get_mdl(model_type or self.model_type)

        transform = self.model_transform

        if self.set_transform_range:
            if hasattr(transform, "range"):
                transform = attrs.evolve(
                    transform,
                    range=(
                        self.freq.min().to_value("MHz"),
                        self.freq.max().to_value("MHz"),
                    ),
                )
            if hasattr(transform, "scale"):
                transform = attrs.evolve(
                    transform,
                    scale=(
                        self.freq.min().to_value("MHz")
                        + self.freq.max().to_value("MHz")
                    )
                    / 2,
                )

        model = model_type(n_terms=n_terms, transform=transform, **self.model_kwargs)
        emodel = model.at(x=freq.to_value("MHz"))

        cmodel = self.complex_model_type(emodel, emodel)

        return cmodel.fit(
            ydata=raw_s11
            * np.exp(2 * np.pi * 1j * self.model_delay * freq).to_value(""),
            **self.fit_kwargs,
        )

    @cached_property
    def _s11_model(self) -> callable:
        """The S11 model."""
        return self.get_s11_model(self.raw_s11)

    @cached_property
    def _splines(self) -> callable:
        if self.complex_model_type == ComplexRealImagModel:
            return (
                Spline(self.freq.to_value("MHz"), np.real(self.raw_s11)),
                Spline(self.freq.to_value("MHz"), np.imag(self.raw_s11)),
            )
        return (
            Spline(self.freq.to_value("MHz"), np.abs(self.raw_s11)),
            Spline(self.freq.to_value("MHz"), np.angle(self.raw_s11)),
        )

    def s11_model(self, freq: np.ndarray | tp.FreqType) -> np.ndarray:
        """Compute the S11 at a specific set of frequencies."""
        if hasattr(freq, "unit"):
            freq = freq.to_value("MHz")

        if not self.use_spline:
            return self._s11_model(freq) * np.exp(
                -1j * 2 * np.pi * self.model_delay.to_value("microsecond") * freq
            )
        if self.complex_model_type == ComplexRealImagModel:
            return self._splines[0](freq) + 1j * self._splines[1](freq)
        return self._splines[0](freq) * np.exp(1j * self._splines[1](freq))

    def plot_residuals(
        self,
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

        corr = self.raw_s11
        model = self.s11_model(self.freq.to_value("MHz"))

        fq = self.freq.to_value("MHz")
        ax[0].plot(fq, 20 * np.log10(np.abs(model)), color=color_abs, label=label)
        if ylabels:
            ax[0].set_ylabel(r"$|S_{11}|$")

        ax[1].plot(fq, np.abs(model) - np.abs(corr), color_diff)
        if ylabels:
            ax[1].set_ylabel(r"$\Delta  |S_{11}|$")

        ax[2].plot(fq, np.unwrap(np.angle(model)) * 180 / np.pi, color=color_abs)
        if ylabels:
            ax[2].set_ylabel(r"$\angle S_{11}$")

        ax[3].plot(
            fq,
            np.unwrap(np.angle(model)) - np.unwrap(np.angle(corr)),
            color_diff,
        )
        if ylabels:
            ax[3].set_ylabel(r"$\Delta \angle S_{11}$")

        lname = (
            self.load_name if hasattr(self, "load_name") else self.__class__.__name__
        )

        if title is None:
            title = f"{lname} Reflection Coefficient Models"

        if title:
            fig.suptitle(f"{lname} Reflection Coefficient Models", fontsize=14)
        if label:
            ax[0].legend()

        return fig

    def with_model_delay(self, delay: tp.Time | None = None) -> S11Model:
        """Get a new S11Model with a different model delay."""
        if delay is None:
            delay = rc.get_delay(self.freq, self.raw_s11)
        return attrs.evolve(self, model_delay=delay)

    # Constructor Methods
    @classmethod
    def from_receiver_filespec(cls, **kwargs) -> Self:
        from .receiver import get_receiver_s11model_from_filespec
        return get_receiver_s11model_from_filespec(**kwargs)


    @classmethod
    def from_load_and_switch(cls, **kwargs) -> Self:
        from .cal_loads import get_loads11_from_load_and_switch
        return get_loads11_from_load_and_switch(**kwargs)

    @classmethod
    def from_edges2_loaddef(cls, **kwargs) -> Self:
        from .cal_loads import get_loads11_from_edges2_loaddef
        return get_loads11_from_edges2_loaddef(**kwargs)

    @classmethod
    def from_edges3_loaddef(cls, **kwargs) -> Self:
        from .cal_loads import get_loads11_from_edges3_loaddef
        return get_loads11_from_edges3_loaddef(**kwargs)

    @classmethod
    def from_s1p_files(cls, **kwargs) -> Self:
        """Generate from a list of four S1P files.

        The files are interpreted as the (open, short, match) then (external).
        """
        raise NotImplementedError(
            "Use from_load_and_switch instead, and pack the four files into correct objects."
        )

    @classmethod
    def from_calibrated_file(
        cls,
        path: tp.PathLike,
        f_low: tp.FreqType = 0 * un.MHz,
        f_high: tp.FreqType = np.inf * un.MHz,
        **kwargs,
    ):
        """Generate from a single pre-calibrated file."""
        delimiter = "," if path.endswith(".csv") else " "

        f_orig, gamma_real, gamma_imag = np.loadtxt(
            path,
            skiprows=1,
            delimiter=delimiter,
            unpack=True,
            comments=["BEGIN", "END", "#"],
        )

        mask = get_mask(f_orig * un.Hz, low=f_low, high=f_high)

        return cls(
            raw_s11=gamma_real[mask] + 1j * gamma_imag[mask],
            freq=f_orig[mask] * un.Hz,
            **kwargs,
        )

