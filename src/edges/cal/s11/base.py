"""Classes representing S11 and full S-parameter measurements."""

from collections.abc import Sequence
from typing import Self

import attrs
import numpy as np
from astropy import units as un
from pygsdata.attrs import npfield

from edges.frequencies import get_mask
from edges.io import calobsdef
from edges.io.calobsdef3 import CalObsDefEDGES3

from ... import types as tp
from ...io.serialization import hickleable
from .. import reflection_coefficient as rc


@hickleable
@attrs.define(kw_only=True, frozen=True)
class CalibratedS11:
    """A class for representing calibrated S11 measurements of a device."""

    s11: np.ndarray = npfield(
        dtype=complex,
        possible_ndims=(1,),
    )
    freqs: tp.FreqType = npfield(dtype=float, possible_ndims=(1,), unit=un.MHz)

    @freqs.validator
    def _fv(self, att, val):
        if val.size != len(self.s11):
            raise ValueError(f"len(freq) != len(raw_s11) [{len(val)},{len(self.s11)}]")

    # Constructor Methods
    @classmethod
    def from_receiver_filespec(
        cls, pathspec: calobsdef.ReceiverS11 | Sequence[calobsdef.ReceiverS11], **kwargs
    ) -> Self:
        """Create a CalibratedS11 from the file-specification of a receiver."""
        from .receiver import get_receiver_s11model_from_filespec

        return get_receiver_s11model_from_filespec(pathspec, **kwargs)

    @classmethod
    def from_load_and_switch(
        cls,
        loaddef: calobsdef.LoadS11,
        switchdef: calobsdef.SwitchingState | None,
        **kwargs,
    ) -> Self:
        """Construct a CalibratedS11 given a loaddef spec and internal switch spec."""
        from .cal_loads import get_loads11_from_load_and_switch

        return get_loads11_from_load_and_switch(
            loaddef=loaddef, switchdef=switchdef, **kwargs
        )

    @classmethod
    def from_edges2_loaddef(
        cls, caldef: calobsdef.CalObsDefEDGES2, load: str, **kwargs
    ) -> Self:
        """Create a CalibratedS11 object from an EDGES-2 loaddef spec."""
        from .cal_loads import get_loads11_from_edges2_loaddef

        return get_loads11_from_edges2_loaddef(caldef, load, **kwargs)

    @classmethod
    def from_edges3_loaddef(
        cls,
        caldef: CalObsDefEDGES3,
        load: str,
        calkit: rc.Calkit = rc.AGILENT_ALAN,
        **kwargs,
    ) -> Self:
        """Create a CalibratedS11 object from an EDGES-3 load definition spec."""
        from .cal_loads import get_loads11_from_edges3_loaddef

        return get_loads11_from_edges3_loaddef(
            caldef, load=load, calkit=calkit, **kwargs
        )

    @classmethod
    def from_s1p_files(cls, **kwargs) -> Self:
        """Generate from a list of four S1P files.

        The files are interpreted as the (open, short, match) then (external).
        """
        raise NotImplementedError(
            "Use from_load_and_switch instead, and pack the four files into correct "
            "objects."
        )

    @classmethod
    def from_calibrated_file(
        cls,
        path: tp.PathLike,
        f_low: tp.FreqType = 0 * un.MHz,
        f_high: tp.FreqType = np.inf * un.MHz,
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
        )

    def smoothed(self, params, freqs: tp.FreqType | None = None):
        """Return a new CalibratedS11, smoothed and interpolated onto new frequencies.

        Parameters
        ----------
        params : ~s11model.S11ModelParams
            The set of parameters to use to construct the smoothing model.
        freqs
            The frequencies to interpolate to. By default, the same frequencies
            as in this object (i.e. only smoothing, no interpolation).
        """
        from .s11model import new_s11_modelled

        return new_s11_modelled(self, params, freqs)


@hickleable
@attrs.define
class CalibratedSParams:
    """A class representing calibrated S-parameters.

    This is similar to :class:`CalibratedS11`, except that it includes all four
    S-parameters instead of just S11.

    Parameters
    ----------
    freqs
        The frequencies of the S-parameters.
    s11
        The S11 parameter.
    s12
        The S12 parameter.
    s22
        The S22 parameter.
    s21
        The S21 parameter. By default, this is equal to S12.
    """

    freqs: tp.FreqType = npfield(dtype=float, possible_ndims=(1,), unit=un.MHz)
    s11: np.ndarray = npfield(dtype=complex, possible_ndims=(1,))
    s12: np.ndarray = npfield(dtype=complex, possible_ndims=(1,))
    s22: np.ndarray = npfield(dtype=complex, possible_ndims=(1,))
    s21: np.ndarray = npfield(dtype=complex, possible_ndims=(1,))

    @s21.default
    def _s21_default(self):
        return self.s12

    def smoothed(self, params, freqs: tp.FreqType | None = None):
        """Return a new CalibratedSparams, smoothed and interpolated to new frequencies.

        Parameters
        ----------
        params : ~s11model.S11ModelParams
            The set of parameters to use to construct the smoothing model.
        freqs
            The frequencies to interpolate to. By default, the same frequencies
            as in this object (i.e. only smoothing, no interpolation).
        """
        from .s11model import S11ModelParams, new_s11_modelled

        if isinstance(params, S11ModelParams):
            params = {"s11": params, "s12": params, "s21": params, "s22": params}

        s11 = new_s11_modelled(
            self.s11, params["s11"], freqs=self.freqs, new_freqs=freqs
        )
        s12 = new_s11_modelled(
            self.s12, params["s12"], freqs=self.freqs, new_freqs=freqs
        )
        s21 = new_s11_modelled(
            self.s21,
            params.get("s21", params["s12"]),
            freqs=self.freqs,
            new_freqs=freqs,
        )
        s22 = new_s11_modelled(
            self.s22, params["s22"], freqs=self.freqs, new_freqs=freqs
        )

        return CalibratedSParams(
            freqs=freqs, s11=s11.s11, s12=s12.s11, s21=s21.s11, s22=s22.s11
        )

    @property
    def smatrix(self) -> rc.SMatrix:
        """Represent the data as an S-Matrix."""
        return rc.SMatrix([[self.s11, self.s12], [self.s21, self.s22]])

    def as_s11(self) -> CalibratedS11:
        """Return just the S11 component."""
        return CalibratedS11(freqs=self.freqs, s11=self.s11)

    @classmethod
    def from_internal_switchdef(cls, switchdef, **kwargs):
        """Create calibrated SParams from an internal switch definition class."""
        from .internal_switch import get_calibrated_sparams_from_switchdef

        return get_calibrated_sparams_from_switchdef(switchdef, **kwargs)

    @classmethod
    def from_hot_load_semi_rigid(
        cls,
        path: tp.PathLike = ":semi_rigid_s_parameters_WITH_HEADER.txt",
        f_low: tp.FreqType = 0 * un.MHz,
        f_high: tp.FreqType = np.inf * un.MHz,
    ):
        """Create a CalibratedSParams object by reading a file with semi-rigid S11s."""
        from .hot_load_cable import read_semi_rigid_cable_sparams_file

        return read_semi_rigid_cable_sparams_file(path, f_low, f_high)
