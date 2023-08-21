"""Register functions as processors for GSData objects."""
from __future__ import annotations

import datetime
import functools
from attrs import define
from typing import Callable, Literal

from .gsdata import GSData


class _Register:
    def __init__(self, func: Callable, kind: str) -> None:
        self.func = func
        self.kind = kind
        functools.update_wrapper(self, func, updated=())

    def __call__(
        self, data: GSData, *args, message: str = "", **kw
    ) -> GSData | list[GSData]:
        now = datetime.datetime.now()
        newdata = self.func(data, *args, **kw)

        from .gsdata import GSData

        history = {
            "message": message,
            "function": self.func.__name__,
            "parameters": kw,
            "timestamp": now,
        }
        if isinstance(newdata, GSData):
            return newdata.update(history=history)
        try:
            return [nd.update(history=history) for nd in newdata]
        except Exception as e:
            raise TypeError(
                f"{self.func.__name__} returned {type(newdata)} "
                f"instead of GSData or list thereof."
            ) from e


GSDATA_PROCESSORS = {}


@define
class gsregister:  # noqa: N801
    kind: Literal["gather", "calibrate", "filter", "reduce", "supplement"]

    def __call__(self, func: Callable) -> Callable:
        """Register a function as a processor for GSData objects."""
        out = _Register(func, self.kind)
        GSDATA_PROCESSORS[func.__name__] = out
        return out
