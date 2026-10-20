"""Backwards-compatible alias of :mod:`edges.serialization`.

The serialization machinery (including :func:`~edges.serialization.hickleable`) now
lives in :mod:`edges.serialization`, which has no dependency on :mod:`edges.io`, so
that low-level sub-packages (e.g. :mod:`edges.modeling`) can use it without importing
all of the I/O layer.

Importing ``edges.io.serialization`` returns the very same module object as
``edges.serialization``, so module-level state (e.g. the ``converter`` and its
registered hooks) is shared and monkeypatching either name affects both.
"""

import sys

from .. import serialization as _serialization

sys.modules[__name__] = _serialization
