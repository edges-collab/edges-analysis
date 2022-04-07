"""Corrections for S11 measurements."""
from __future__ import annotations
import numpy as np
from edges_cal.s11 import (
    InternalSwitch,
    StandardsReadings,
    VNAReading,
    LoadPlusSwitchS11,
    LoadS11,
)
from typing import Sequence
from edges_cal import modelling as mdl
from astropy import units as u
from edges_cal.tools import FrequencyRange
from edges_cal import types as tp
import attr


@attr.s
class AntennaS11(LoadS11):
    _complex_model_type_default = mdl.ComplexRealImagModel
    _default_nterms = 10
    _model_type_default = mdl.Polynomial

    model_delay: tp.Time = attr.ib(170 * u.ns)

    @classmethod
    def from_s1p_files(
        cls,
        files: Sequence[tp.PathLike],
        internal_switch: InternalSwitch,
        f_low: float = 0 * u.MHz,
        f_high: float = np.inf * u.MHz,
        **kwargs,
    ) -> AntennaS11:
        """Generate from a list of four S1P files."""
        files = sorted(files)
        assert len(files) == 4
        standards = StandardsReadings(
            *[VNAReading.from_s1p(fl, f_low=f_low, f_high=f_high) for fl in files[:3]]
        )
        external = VNAReading.from_s1p(files[-1], f_low=f_low, f_high=f_high)

        # Note that here the
        load_s11 = (
            LoadPlusSwitchS11(
                standards=standards,
                external_match=external,
                load_name="antenna",
            ),
        )

        return cls.from_load_and_internal_switch(
            load_s11=load_s11, internal_switch=internal_switch, **kwargs
        )

    @classmethod
    def from_single_file(
        cls,
        path,
        internal_switch: InternalSwitch,
        f_low: float = 0 * u.MHz,
        f_high: float = np.inf * u.MHz,
        **kwargs,
    ):
        """Generate from a single pre-calibrated file."""
        if path.endswith(".csv"):
            delimiter = ","
        else:
            delimiter = " "

        f_orig, gamma_real, gamma_imag = np.loadtxt(
            path,
            skiprows=1,
            delimiter=delimiter,
            unpack=True,
            comments=["BEGIN", "END", "#"],
        )

        return cls(
            raw_s11=gamma_real + 1j * gamma_imag,
            freq=FrequencyRange(f_orig * u.Hz, f_low=f_low, f_high=f_high),
            internal_switch=internal_switch,
            **kwargs,
        )
