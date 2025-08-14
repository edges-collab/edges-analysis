"""Functions for creating CalibratedS11 objects from internal switch data (EDGES 2)."""

from collections.abc import Sequence

import numpy as np
from astropy import units as un

from edges.io import calobsdef

from .. import reflection_coefficient as rc
from .base import CalibratedSParams
from .calkit_standards import StandardsReadings


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
        internal = StandardsReadings.from_filespec(
            isw.internal, f_low=f_low, f_high=f_high
        )
        external = StandardsReadings.from_filespec(
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
