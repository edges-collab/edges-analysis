"""Functions for creating CalibratedS11 objects from internal switch data (EDGES 2)."""

from collections.abc import Sequence

import attrs
import numpy as np
from astropy import units as un

from edges import types as tp
from edges.io import calobsdef
from edges.io.serialization import hickleable

from .. import reflection_coefficient as rc
from .base import CalibratedS11, CalibratedSParams
from .calkit_standards import StandardsReadings
from .s11model import S11ModelParams, new_s11_modelled


def get_calibrated_sparams_from_switchdef(
    internal_switch: calobsdef.SwitchingState | Sequence[calobsdef.SwitchingState],
    calkit=rc.AGILENT_85033E,
    resistance=None,
    f_low=0 * un.MHz,
    f_high=np.inf * un.MHz,
) -> CalibratedSParams:
    """Initiate from an edges-io object."""
    if not hasattr(internal_switch, "__len__"):
        internal_switch = [internal_switch]

    if resistance is not None:
        calkit = rc.get_calkit(calkit, resistance_of_match=resistance)

    smatrices = []
    corrections = []
    for isw in internal_switch:
        internal = StandardsReadings.from_io(isw.internal, f_low=f_low, f_high=f_high)
        external = StandardsReadings.from_io(isw.external, f_low=f_low, f_high=f_high)
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

    s11, s12, s22 = get_sparams_from_corrections(freq, corrections, calkit)

    return CalibratedSParams(freqs=freq, s11=s11, s12=s12, s22=s22)


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


@hickleable
@attrs.define
class InternalSwitch:
    s11: CalibratedS11 = attrs.field()
    s12: CalibratedS11 = attrs.field()
    s22: CalibratedS11 = attrs.field()

    def smoothed(
        self,
        params: S11ModelParams | tuple[S11ModelParams, S11ModelParams, S11ModelParams],
        freqs: tp.FreqType | None = None,
    ):
        """Return a new InternalSwitch, smoothed and interpolated onto new frequencies.

        Parameters
        ----------
        params : ~s11model.S11ModelParams
            The set of parameters to use to construct the smoothing model.
        freqs
            The frequencies to interpolate to. By default, the same frequencies
            as in this object (i.e. only smoothing, no interpolation).
        """
        if isinstance(params, S11ModelParams):
            params = (params,) * 3

        s11 = new_s11_modelled(self.s11, params[0], freqs)
        s12 = new_s11_modelled(self.s12, params[0], freqs)
        s22 = new_s11_modelled(self.s22, params[0], freqs)

        return InternalSwitch(s11=s11, s12=s12, s22=s22)

    def smatrix(self) -> rc.SMatrix:
        """Compute an S-Matrix from the internal switch."""
        return rc.SMatrix([[self.s11.s11, self.s12.s11], [self.s12.s11, self.s22.s11]])
