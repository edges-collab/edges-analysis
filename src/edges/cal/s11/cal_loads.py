import numpy as np
from typing import Self

from . import StandardsReadings, InternalSwitch

from .s11model import S11Model
from .. import reflection_coefficient as rc
from edges.io import CalObsDefEDGES2, LoadS11, CalObsDefEDGES3, SwitchingState, SParams
from edges import types as tp
from astropy import units as un


def calibrate_loads11_with_switch(
    load_s11: np.ndarray,
    internal_switch: InternalSwitch,
    **kwargs,
) -> S11Model:
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

    return S11Model(
        freq=freq,
        raw_s11=np.mean(s11s, axis=0),
        **kwargs,
    )

def get_loads11_from_load_and_switch(
    loaddef: LoadS11,
    switchdef: SwitchingState | None = None,
    internal_switch_kwargs: dict | None = None,
    calkit: rc.Calkit | None = None,
    load_kw: dict | None = None,
    f_low: tp.FreqType = 0 * un.MHz,
    f_high: tp.FreqType = np.inf * un.MHz,
    **kwargs,
) -> S11Model:
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

        return calibrate_loads11_with_switch(
            load_s11=loads11, internal_switch=internal_switch, **kwargs
        )
    return S11Model(raw_s11=loads11, freq=freq, **kwargs)

@classmethod
def get_loads11_from_edges2_loaddef(
    caldef: CalObsDefEDGES2, load: str, **kwargs
) -> S11Model:
    return get_loads11_from_load_and_switch(
        loaddef=getattr(caldef, load), switchdef=caldef.switching_state, **kwargs
    )

@classmethod
def get_loads11_from_edges3_loaddef(
    cls,
    caldef: CalObsDefEDGES3,
    load: str,
    calkit: rc.Calkit = rc.AGILENT_ALAN,
    **kwargs,
) -> S11Model:
    """Create a LoadS11 object from the EDGES-3 CalibrationObservation."""
    return get_loads11_from_load_and_switch(
        loaddef=getattr(caldef, load), calkit=calkit, **kwargs
    )