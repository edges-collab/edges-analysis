from edges.io.serialization import hickleable
import attrs
from collections.abc import Sequence, Callable
import numpy as np
from functools import cached_property
from astropy import units as un
from typing import Self

from edges.units import vld_unit
from edges.modelling import Model, UnitTransform, Polynomial, ComplexRealImagModel
from edges import types as tp
from edges.io import calobsdef
from .. import reflection_coefficient as rc
from .calkit_standards import StandardsReadings


def _tuplify(x):
    if not hasattr(x, "__len__"):
        return (int(x), int(x), int(x))
    return tuple(int(xx) for xx in x)

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
        internal_switch: calobsdef.SwitchingState | Sequence[calobsdef.SwitchingState],
        calkit=rc.AGILENT_85033E,
        resistance=None,
        f_low=0 * un.MHz,
        f_high=np.inf * un.MHz,
        **kwargs,
    ) -> Self:
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
