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

from collections.abc import Callable, Sequence
from functools import cached_property
from typing import Any, ClassVar

import attrs
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as un
from astropy.constants import c as speed_of_light
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from edges.io import SParams, calobsdef, calobsdef3

from .. import types as tp
from ..io.serialization import hickleable
from ..modelling import (
    ComplexMagPhaseModel,
    ComplexRealImagModel,
    Fourier,
    Model,
    Modelable,
    Polynomial,
    UnitTransform,
    XTransform,
    get_mdl,
)
from ..units import vld_unit
from . import reflection_coefficient as rc


def _tuplify(x):
    if not hasattr(x, "__len__"):
        return (int(x), int(x), int(x))
    return tuple(int(xx) for xx in x)


@attrs.define
class StandardsReadings:
    open: SParams = attrs.field(validator=attrs.validators.instance_of(SParams))
    short: SParams = attrs.field(validator=attrs.validators.instance_of(SParams))
    match: SParams = attrs.field(validator=attrs.validators.instance_of(SParams))

    @short.validator
    def _short_vld(self, att, val):
        if np.any(val.freq != self.open.freq):
            raise ValueError(
                "short standard does not have same frequencies as open standard!"
            )

    @match.validator
    def _match_vld(self, att, val):
        if np.any(val.freq != self.open.freq):
            raise ValueError(
                "match standard does not have same frequencies as open standard!"
            )

    @property
    def freq(self) -> tp.FreqType:
        """Frequencies of the standards measurements."""
        return self.open.freq

    @classmethod
    def from_io(cls, paths: calobsdef.Calkit, **kwargs) -> StandardsReadings:
        """Instantiate from a given edges-io object.

        Parameters
        ----------
        device
            The device for which the standards were measured.

        Other Parameters
        ----------------
        kwargs
            Everything else is passed to the :class:`SParams` objects. This includes
            f_low and f_high.
        """
        return cls(
            open=SParams.from_s1p_file(paths.open, **kwargs),
            short=SParams.from_s1p_file(paths.short, **kwargs),
            match=SParams.from_s1p_file(paths.match, **kwargs),
        )


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
    model_delay: tp.Time = attrs.field(default=0 * un.s)
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
    def from_edges2_receiver(
        cls,
        pathspec: calobsdef.ReceiverS11 | Sequence[calobsdef.ReceiverS11],
        calkit: rc.Calkit = rc.AGILENT_85033E,
        resistance: float | None = None,
        f_low=0.0 * un.MHz,
        f_high=np.inf * un.MHz,
        **kwargs,
    ) -> Self:
        """
        Create an instance from a given path.

        Parameters
        ----------
        path : str or Path
            Path to overall Calibration Observation.
        run_num_load : int
            The run to use for the LNA (default latest available).
        run_num_switch : int
            The run to use for the switching state (default lastest available).
        kwargs
            All other arguments passed through to :class:`S11Model`.

        Returns
        -------
        receiver
            The Receiver object.
        """
        if resistance is not None:
            calkit = rc.get_calkit(calkit, resistance_of_match=resistance)

        if not hasattr(pathspec, "__len__"):
            pathspec = [pathspec]

        s11s = []
        for dv in pathspec:
            standards = StandardsReadings.from_io(dv.calkit, f_low=f_low, f_high=f_high)
            receiver_reading = SParams.from_s1p_file(
                dv.device, f_low=f_low, f_high=f_high
            )
            freq = standards.freq

            smatrix = rc.SMatrix.from_calkit_and_vna(calkit, standards)
            s11s.append(rc.gamma_de_embed(receiver_reading.s11, smatrix))

        metadata = {"pathspecs": pathspec, "calkit": calkit}

        return cls(
            raw_s11=np.mean(s11s, axis=0), freq=freq, metadata=metadata, **kwargs
        )

    @classmethod
    def from_edges3_receiver(
        cls,
        obs: calobsdef3.ReceiverS11,
        calkit: rc.Calkit = rc.AGILENT_ALAN,
        f_low=0.0 * un.MHz,
        f_high=np.inf * un.MHz,
        cable_length: tp.LengthType = 0.0 * un.cm,
        cable_loss_percent: float = 0.0,
        cable_dielectric_percent: float = 0.0,
        **kwargs,
    ):
        """Create a Receiver object from the EDGES-3 receiver."""
        standards = StandardsReadings.from_io(obs.calkit, f_low=f_low, f_high=f_high)
        receiver_reading = SParams.from_s1p_file(obs.device, f_low=f_low, f_high=f_high)

        freq = standards.freq
        smatrix = rc.SMatrix.from_calkit_and_vna(calkit, standards)
        calibrated_s11_raw = rc.gamma_de_embed(receiver_reading.s11, smatrix)

        _T, s11, s12 = rc.path_length_correction_edges3(
            freq=freq,
            delay=cable_length / speed_of_light,
            gamma_in=0,
            lossf=1 + cable_loss_percent * 0.01,
            dielf=1 + cable_dielectric_percent * 0.01,
        )
        smatrix = rc.SMatrix([[s11, s12], [s12, s11]])
        Ta = calibrated_s11_raw

        if cable_length > 0.0:
            Ta = rc.gamma_embed(smatrix, Ta)
        elif cable_length < 0.0:
            Ta = rc.gamma_de_embed(smatrix, Ta)

        metadata = {"calkit": calkit}

        return cls(raw_s11=Ta, freq=freq, metadata=metadata, **kwargs)

    @classmethod
    def from_edges2_load_and_internal_switch(
        cls,
        load_s11: np.ndarray,
        internal_switch: InternalSwitch,
        base: LoadS11 | None = None,
        **kwargs,
    ) -> LoadS11:
        """Generate the LoadS11 from an uncalibrated load and internal switch."""
        if not hasattr(load_s11, "__len__"):
            load_s11 = [load_s11]

        freq = internal_switch.freq

        s11s = []
        nu = freq.to_value("MHz")

        for load in load_s11:
            gamma = rc.gamma_de_embed(
                load.get_calibrated_s11(), internal_switch.smatrix(nu)
            )
            s11s.append(gamma)

        # metadata = {"load_s11s": load_s11}
        # metadata.update(getattr(internal_switch, "metadata", {}))

        if base is None:
            return cls(
                freq=freq,
                raw_s11=np.mean(s11s, axis=0),
                # metadata=metadata,
                # load_name=load_s11[0].load_name,
                # internal_switch=internal_switch,
                **kwargs,
            )
        return attrs.evolve(
            base,
            freq=freq,
            raw_s11=np.mean(s11s, axis=0),
            # metadata=metadata,
            # load_name=load_s11[0].load_name,
        )

    @classmethod
    def from_load_and_switch(
        cls,
        loaddef: calobsdef.LoadS11,
        switchdef: calobsdef.SwitchingState | None = None,
        internal_switch_kwargs: dict | None = None,
        calkit: rc.Calkit | None = None,
        load_kw: dict | None = None,
        f_low: tp.FreqType = 0 * un.MHz,
        f_high: tp.FreqType = np.inf * un.MHz,
        **kwargs,
    ) -> Self:
        """Instantiate from an :class:`edges.io.io.S11Dir` object."""
        internal_switch_kwargs = internal_switch_kwargs or {}
        internal_switch_kwargs["f_low"] = f_low
        internal_switch_kwargs["f_high"] = f_high

        load_kw = load_kw or {}
        load_kw["f_low"] = f_low
        load_kw["f_high"] = f_high

        standards = (
            StandardsReadings.from_io(loaddef.calkit, f_low=f_low, f_high=f_high),
        )
        external_match = (
            SParams.from_s1p_file(loaddef.external, f_low=f_low, f_high=f_high),
        )
        freq = standards.freq

        # Historically we use (1, -1, 0) in EDGES2, and proper calkit in EDGES3
        smatrix = rc.get_sparams_from_osl(
            1 if calkit is None else calkit.open.reflection_coefficient(freq),
            -1 if calkit is None else calkit.short.reflection_coefficient(freq),
            0.0 if calkit is None else calkit.match.reflection_coefficient(freq),
            standards.open.s11,
            standards.short.s11,
            standards.match.s11,
        )
        loads11 = rc.gamma_de_embed(external_match.s11, smatrix)

        if switchdef is not None:
            internal_switch = InternalSwitch.from_io(
                switchdef, **internal_switch_kwargs
            )

            return cls.from_edges2_load_and_internal_switch(
                load_s11=loads11, internal_switch=internal_switch, **kwargs
            )
        return cls(raw_s11=loads11, freq=freq, **kwargs)

    @classmethod
    def from_edges2_loaddef(
        cls, caldef: calobsdef.CalObsDefEDGES2, load: str, **kwargs
    ):
        return cls.from_load_and_switch(
            loaddef=getattr(caldef, load), switchdef=caldef.switching_state, **kwargs
        )

    @classmethod
    def from_edges3_loaddef(
        cls,
        caldef: calobsdef3.CalObsDefEDGES3,
        load: str,
        calkit: rc.Calkit = rc.AGILENT_ALAN,
        **kwargs,
    ):
        """Create a LoadS11 object from the EDGES-3 CalibrationObservation."""
        return cls.from_load_and_switch(
            loaddef=getattr(caldef, load), calkit=calkit, **kwargs
        )

    @classmethod
    def from_s1p_files(
        cls,
        files: tuple[tp.PathLike, tp.PathLike, tp.PathLike, tp.PathLike],
        internal_switch: InternalSwitch,
        f_low: float = 0 * un.MHz,
        f_high: float = np.inf * un.MHz,
        **kwargs,
    ) -> AntennaS11:
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
        f_low: units.Quantity[frequency] = 0 * un.MHz,
        f_high: float = np.inf * un.MHz,
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


@hickleable
@attrs.define
class InternalSwitch:
    freq: tp.FreqType = attrs.field(
        validator=vld_unit("frequency"), eq=attrs.cmp_using(eq=np.array_equal)
    )
    s11_data: np.ndarray = attrs.field(eq=attrs.cmp_using(eq=np.array_equal))
    s12_data: np.ndarray = attrs.field(eq=attrs.cmp_using(eq=np.array_equal))
    s22_data: np.ndarray = attrs.field(eq=attrs.cmp_using(eq=np.array_equal))
    model: Model = attrs.field()
    n_terms: tuple[int, int, int] | int = attrs.field(
        default=(7, 7, 7), converter=_tuplify
    )
    metadata: dict = attrs.field(factory=dict, eq=False)
    fit_kwargs: dict = attrs.field(factory=dict, eq=False)

    @model.default
    def _mdl_default(self):
        return Polynomial(
            n_terms=7,
            transform=UnitTransform(
                range=(self.freq.min().to_value("MHz"), self.freq.max().to_value("MHz"))
            ),
        )

    @classmethod
    def from_io(
        cls,
        internal_switch: calobsdef.InternalSwitch | Sequence[calobsdef.InternalSwitch],
        calkit=rc.AGILENT_85033E,
        resistance=None,
        f_low=0 * un.MHz,
        f_high=np.inf * un.MHz,
        **kwargs,
    ) -> InternalSwitch:
        """Initiate from an edges-io object."""
        if not hasattr(internal_switch, "__len__"):
            internal_switch = [internal_switch]

        if resistance is not None:
            calkit = rc.get_calkit(calkit, resistance_of_match=resistance)

        smatrices = []
        corrections = []
        for isw in internal_switch:
            internal = StandardsReadings.from_io(
                isw.internal, f_low=f_low, f_high=f_high
            )
            external = StandardsReadings.from_io(
                isw.external, f_low=f_low, f_high=f_high
            )
            freq = internal.freq

            # TODO: not clear why we use the ideal values of 1,-1,0 instead of the physical
            # expected values of calkit.match.intrinsic_gamma etc.
            smtrx = rc.get_sparams_from_osl(
                1, -1, 0, internal.open.s11, internal.short.s11, internal.match.s11
            )

            corr = {
                kind: rc.gamma_de_embed(getattr(external, kind).s11, smtrx)
                for kind in ("open", "short", "match")
            }

            smatrices.append(smtrx)
            corrections.append(corr)

        s11, s12, s22 = cls.get_sparams_from_corrections(freq, corrections, calkit)

        metadata = {
            "calkit": calkit,
            "internal_switches": internal_switch,
            "corrections": corrections,
        }

        return cls(
            freq=freq,
            s11_data=s11,
            s12_data=s12,
            s22_data=s22,
            metadata=metadata,
            **kwargs,
        )

    @staticmethod
    def get_sparams_from_corrections(freq, corrections, calkit):
        """Get S-parameters from a set of measured corrections."""
        s11s, s12s, s22s = [], [], []

        for cc in corrections:
            smatrix = rc.get_sparams_from_osl(
                calkit.open.reflection_coefficient(freq),
                calkit.short.reflection_coefficient(freq),
                calkit.match.reflection_coefficient(freq),
                cc["open"],
                cc["short"],
                cc["match"],
            )
            s11s.append(smatrix.s11)
            s12s.append(smatrix.s12 * smatrix.s21)
            s22s.append(smatrix.s22)

        return np.mean(s11s, axis=0), np.mean(s12s, axis=0), np.mean(s22s, axis=0)

    def with_new_calkit(self, calkit: rc.Calkit):
        """Obtain a new InternalSwitch using a new calkit."""
        if "corrections" not in self.metadata:
            raise RuntimeError(
                "Cannot update calkit when the object was not made with a calkit."
            )

        s11, s12, s22 = self.get_sparams_from_corrections(
            self.freq, self.metadata["corrections"], calkit
        )
        return attrs.evolve(self, s11_data=s11, s12_data=s12, s22_data=s22)

    @n_terms.validator
    def _n_terms_val(self, att, val):
        if len(val) != 3:
            raise TypeError(
                f"n_terms must be an integer or tuple of three integers "
                f"(for s11, s12, s22). Got {val}."
            )
        if any(v < 1 for v in val):
            raise ValueError(f"n_terms must be >0, got {val}.")

    @cached_property
    def calkit(self) -> rc.Calkit:
        """The calkit used for the InternalSwitch."""
        if "calkit" in self.metadata:
            return self.metadata["calkit"]
        raise AttributeError("calkit not known!")

    @cached_property
    def _s11_model(self):
        """The input unfit S11 model."""
        model = self.model.with_nterms(n_terms=self.n_terms[0])
        return ComplexRealImagModel(real=model, imag=model)

    @cached_property
    def _s12_model(self):
        """The input unfit S12 model."""
        model = self.model.with_nterms(n_terms=self.n_terms[1])
        return ComplexRealImagModel(real=model, imag=model)

    @cached_property
    def _s22_model(self):
        """The input unfit S22 model."""
        model = self.model.with_nterms(n_terms=self.n_terms[2])
        return ComplexRealImagModel(real=model, imag=model)

    @cached_property
    def s11_model(self) -> Callable:
        """The fitted S11 model."""
        return self._get_reflection_model("s11")

    @cached_property
    def s12_model(self) -> Callable:
        """The fitted S12 model."""
        return self._get_reflection_model("s12")

    @cached_property
    def s22_model(self) -> Callable:
        """The fitted S22 model."""
        return self._get_reflection_model("s22")

    def _get_reflection_model(self, kind: str) -> Model:
        # 'kind' should be 's11', 's12' or 's22'
        data = getattr(self, f"{kind}_data")
        return getattr(self, f"_{kind}_model").fit(
            xdata=self.freq.to_value("MHz"), ydata=data, **self.fit_kwargs
        )

    def smatrix(self, freq) -> rc.SMatrix:
        """Compute an S-Matrix from the internal switch."""
        s12 = np.sqrt(self.s12_model(freq))
        return rc.SMatrix([[self.s11_model(freq), s12], [s12, self.s22_model(freq)]])


# @hickleable
# @attrs.define(kw_only=True, frozen=True)
# class LoadPlusSwitchS11:
#     """S11 for a lab calibration load including the internal switch.

#     Note that this class is generally not used directly, as we require the S11 of the
#     load after *correcting* for the switch. See :class:`LoadS11` for this.

#     Parameters
#     ----------
#     standards
#         The internal VNA standards readings of the full system (input + internalswitch).
#     external_match
#         The reading of the VNA from the match standard as applied externally.
#     load_name
#         Optional name for the input device.

#     Other Parameters
#     ----------------
#     Passed through to :class:`_S11Base`.
#     """

#     standards: StandardsReadings = attrs.field()
#     external_match: SParams = attrs.field(
#         validator=attrs.validators.instance_of(SParams)
#     )
#     load_name: str | None = attrs.field(default=None)

#     @classmethod
#     def from_io(
#         cls,
#         load_io: calobsdef.LoadS11,
#         f_low=0 * un.MHz,
#         f_high=np.inf * un.MHz,
#     ):
#         """
#         Create a new object from a given path and load name.

#         Parameters
#         ----------
#         load_io
#             The io.LoadS11 that this will be based off.
#         f_low, f_high
#             Min/max frequencies to keep in the modelling.

#         Returns
#         -------
#         s11 : :class:`LoadPlusSwitchS11`
#             The S11 of the load + internal switch.
#         """
#         return cls(
#             standards=StandardsReadings.from_io(load_io.calkit, f_low=f_low, f_high=f_high),
#             external_match=SParams.from_s1p(load_io.external, f_low=f_low, f_high=f_high),
#         )

#     def get_calibrated_s11(self):
#         """The measured S11 of the load, calculated from raw internal standards."""
#         # TODO: It's not clear exactly why we use the completely ideal values for the
#         # OSL standards here, instead of their predicted values from physical parameters
#         # eg. calkit.match.intrinsic_gamma.
#         smatrix = rc.get_sparams_from_osl(
#             1,
#             -1,
#             0.0,
#             self.standards.open.s11,
#             self.standards.short.s11,
#             self.standards.match.s11,
#         )
#         return rc.gamma_de_embed(self.external_match.s11, smatrix)

#     @property
#     def freq(self):
#         """Frequencies of the observation."""
#         return self.standards.open.freq


# @hickleable
# @attrs.define
# class LoadS11(S11Model):
#     """S11 of an input Load."""

#     internal_switch: InternalSwitch = attrs.field()
#     load_name: str | None = attrs.field(default=None)

#     @classmethod
#     def from_load_and_internal_switch(
#         cls,
#         load_s11: LoadPlusSwitchS11 | Sequence[LoadPlusSwitchS11],
#         internal_switch: InternalSwitch,
#         base: LoadS11 | None = None,
#         **kwargs,
#     ) -> LoadS11:
#         """Generate the LoadS11 from an uncalibrated load and internal switch."""
#         if not hasattr(load_s11, "__len__"):
#             load_s11 = [load_s11]

#         freq = load_s11[0].freq

#         default_nterms = {
#             "ambient": 37,
#             "hot_load": 37,
#             "open": 105,
#             "short": 105,
#         }

#         if "n_terms" not in kwargs:
#             kwargs["n_terms"] = default_nterms.get(
#                 load_s11[0].load_name, cls._default_nterms
#             )

#         s11s = []
#         nu = freq.to_value("MHz")
#         s12 = np.sqrt(internal_switch.s12_model(nu))
#         smatrix = rc.SMatrix(
#             [[internal_switch.s11_model(nu), s12], [s12, internal_switch.s22_model(nu)]]
#         )
#         for load in load_s11:
#             gamma = rc.gamma_de_embed(load.get_calibrated_s11(), smatrix)
#             s11s.append(gamma)

#         metadata = {"load_s11s": load_s11}
#         metadata.update(getattr(internal_switch, "metadata", {}))

#         if base is None:
#             return cls(
#                 freq=freq,
#                 raw_s11=np.mean(s11s, axis=0),
#                 metadata=metadata,
#                 load_name=load_s11[0].load_name,
#                 internal_switch=internal_switch,
#                 **kwargs,
#             )
#         return attrs.evolve(
#             base,
#             freq=freq,
#             raw_s11=np.mean(s11s, axis=0),
#             metadata=metadata,
#             load_name=load_s11[0].load_name,
#         )

#     @classmethod
#     def from_caldef(
#         cls,
#         s11_io: calobsdef.LoadS11,
#         switch_io: calobsdef.SwitchingState,
#         internal_switch_kwargs=None,
#         load_kw=None,
#         f_low: tp.FreqType = 0 * un.MHz,
#         f_high: tp.FreqType = np.inf * un.MHz,
#         **kwargs,
#     ) -> LoadS11:
#         """Instantiate from an :class:`edges.io.io.S11Dir` object."""
#         internal_switch_kwargs = internal_switch_kwargs or {}
#         internal_switch_kwargs["f_low"] = f_low
#         internal_switch_kwargs["f_high"] = f_high

#         load_kw = load_kw or {}
#         load_kw["f_low"] = f_low
#         load_kw["f_high"] = f_high

#         internal_switch = InternalSwitch.from_io(switch_io, **internal_switch_kwargs)

#         load_s11s = [
#             LoadPlusSwitchS11.from_io(s11_io, **load_kw)
# #            for xx in getattr(s11_io, load_name)
#         ]

#         return cls.from_load_and_internal_switch(
#             load_s11=load_s11s, internal_switch=internal_switch, **kwargs
#         )

#     @classmethod
#     def from_edges3(
#         cls,
#         obs: calobsdef3.CalibrationObservation,
#         load_name: str,
#         calkit: rc.Calkit = rc.AGILENT_ALAN,
#         f_low: tp.FreqType = 0 * un.MHz,
#         f_high: tp.FreqType = np.inf * un.MHz,
#         **kwargs,
#     ):
#         """Create a LoadS11 object from the EDGES-3 CalibrationObservation."""
#         files = obs.s11_files[load_name]
#         standards = StandardsReadings(
#             open=SParams.from_s1p(files["open"], f_low=f_low, f_high=f_high),
#             short=SParams.from_s1p(files["short"], f_low=f_low, f_high=f_high),
#             match=SParams.from_s1p(files["match"], f_low=f_low, f_high=f_high),
#         )

#         load = SParams.from_s1p(files["input"], f_low=f_low, f_high=f_high)

#         freq = standards.freq

#         smatrix = rc.SMatrix.from_calkit_and_vna(calkit, standards)
#         calibrated_s11_raw = rc.gamma_de_embed(load.s11, smatrix)

#         metadata = {"calkit": calkit}
#         return cls(
#             freq=freq,
#             raw_s11=calibrated_s11_raw,
#             metadata=metadata,
#             load_name=load_name,
#             internal_switch=None,
#             **kwargs,
#         )

#     def get_k_matrix(self, receiver: Receiver, freq: tp.FreqType | None = None):
#         """Compute the K matrix for this source."""
#         if freq is None:
#             freq = self.freq

#         return rcf.get_K(
#             gamma_rec=receiver.s11_model(freq), gamma_ant=self.s11_model(freq)
#         )


# @hickleable
# @attrs.define
# class AntennaS11(LoadS11):
#     """Class to represent the S11 of an antenna."""

#     _complex_model_type_default = ComplexRealImagModel
#     _default_nterms = 10
#     _model_type_default = Polynomial

#     model_delay: tp.Time = attrs.field(default=0 * un.ns)
