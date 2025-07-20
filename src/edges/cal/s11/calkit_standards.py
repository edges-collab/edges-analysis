
import attrs
from edges.io import SParams, calobsdef
from typing import Self
from edges import types as tp
import numpy as np

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
    def from_io(cls, paths: calobsdef.Calkit, **kwargs) -> Self:
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
