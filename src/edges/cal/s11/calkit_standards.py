
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
    def from_io(cls, paths: calobsdef.Calkit, **kwargs) -> Self:
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
