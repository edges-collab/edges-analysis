"""Classes for dealing with measurements from 'calkits'."""

from typing import Self

import attrs
import numpy as np

from edges import types as tp
from edges.io import SParams, calobsdef


@attrs.define
class StandardsReadings:
    """A class representing the full set of calkit measurements.

    This includes the open, short, and match standards.

    Parameters
    ----------
    open
        The open standard S-parameters.
    short
        The short standard S-parameters.
    match
        The match standard S-parameters.
    """

    open: SParams = attrs.field(validator=attrs.validators.instance_of(SParams))
    short: SParams = attrs.field(validator=attrs.validators.instance_of(SParams))
    match: SParams = attrs.field(validator=attrs.validators.instance_of(SParams))

    @short.validator
    @match.validator
    def _vld(self, att, val):
        if val.freq.size != self.open.freq.size:
            raise ValueError(f"{att.name} standard does not have same frequencies")

        if np.any(val.freq != self.open.freq):
            raise ValueError(
                f"{att.name} standard does not have same frequencies as open standard!"
            )

    @property
    def freq(self) -> tp.FreqType:
        """Frequencies of the standards measurements."""
        return self.open.freq

    @classmethod
    def from_filespec(cls, paths: calobsdef.CalkitFileSpec, **kwargs) -> Self:
        """Instantiate from a given Calkit I/O object.

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
