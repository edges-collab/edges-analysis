"""Bit-exact regression tests of the alanmode calibration against golden snapshots.

The other tests in this directory compare alanmode to Alan's C code with tolerances,
which cannot detect a last-bit change of the results. These tests rerun the pipeline
on the data in the repository and check that every intermediate and final array is
*exactly* the same as in snapshots written by ``make_golden.py`` into the directory
given by the ``EDGES_ALANMODE_GOLDEN_DIR`` environment variable (they are skipped if it
is not set). There is no tolerance: refactors of alanmode (and of the parts of
edges.cal it uses) must leave the results bit-identical. The snapshots are written from
the code before a change, on the same machine, since the last bits of the results
depend on the platform; the ``alanmode-golden`` CI workflow does this for every pull
request. See ``make_golden.py`` for details.
"""

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def _import_make_golden() -> ModuleType:
    """Import make_golden.py from this directory (which is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "alanmode_make_golden", Path(__file__).with_name("make_golden.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mg = _import_make_golden()

_GOLDEN_DIR_ENV = os.environ.get("EDGES_ALANMODE_GOLDEN_DIR")

pytestmark = pytest.mark.skipif(
    not _GOLDEN_DIR_ENV,
    reason="EDGES_ALANMODE_GOLDEN_DIR is not set (see make_golden.py)",
)

GOLDEN_DIR = Path(_GOLDEN_DIR_ENV) if _GOLDEN_DIR_ENV else None


@pytest.fixture(scope="module", params=list(mg.CASES))
def case(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="module")
def golden(case: str) -> dict[str, np.ndarray]:
    arrays, fingerprint = mg.load_golden(case, GOLDEN_DIR)

    here = mg.platform_fingerprint()
    if diff := sorted(
        key
        for key in fingerprint.keys() | here.keys()
        if fingerprint.get(key) != here.get(key)
    ):
        pytest.skip(
            f"The golden snapshots in {GOLDEN_DIR} were written on another platform "
            f"(differing in {', '.join(diff)}), on which the results are not "
            "bit-identical. See make_golden.py for how to check on this platform."
        )
    return arrays


@pytest.fixture(scope="module")
def computed(
    case: str, golden: dict[str, np.ndarray], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, np.ndarray]:
    # Depends on `golden` so that the pipeline isn't run if the comparison is skipped.
    return mg.compute_case(case, tmp_path_factory.mktemp(f"golden_{case}"))


def test_same_arrays(golden: dict[str, np.ndarray], computed: dict[str, np.ndarray]):
    """The pipeline produces exactly the arrays in the snapshot."""
    assert sorted(computed) == sorted(golden)


def test_arrays_bit_identical(
    golden: dict[str, np.ndarray], computed: dict[str, np.ndarray]
):
    """Every array is exactly equal to the snapshot, with the same dtype and shape."""
    failures = []
    for key in sorted(golden.keys() & computed.keys()):
        try:
            np.testing.assert_array_equal(computed[key], golden[key], strict=True)
        except AssertionError as err:
            failures.append(f"{key}:\n{err}")

    assert not failures, (
        f"{len(failures)} of {len(golden)} arrays differ from the golden snapshot:\n\n"
        + "\n\n".join(failures)
    )
