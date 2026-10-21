"""Tests of the package's import-time layering and side effects.

Most of these run in a fresh interpreter (via subprocess), since what we care about
is which modules get imported, and the test session has already imported everything.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import edges

# Makes ``edges`` an empty namespace-like package, so that ``import edges.<sub>``
# only imports what ``edges.<sub>`` itself needs, not ``edges/__init__.py``.
_STUB_ROOT = """
import importlib.util, sys, types
_spec = importlib.util.find_spec("edges")
_pkg = types.ModuleType("edges")
_pkg.__path__ = list(_spec.submodule_search_locations)
sys.modules["edges"] = _pkg
"""


def _modules_after(code: str, stub_root: bool = False) -> set[str]:
    """Run ``code`` in a fresh interpreter and return the imported module names."""
    prog = (_STUB_ROOT if stub_root else "") + textwrap.dedent(code)
    prog += "\nimport json, sys\nprint(json.dumps(sorted(sys.modules)))\n"
    proc = subprocess.run(
        [sys.executable, "-c", prog], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    return set(json.loads(proc.stdout.strip().splitlines()[-1]))


def _run(code: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_modeling_does_not_import_io():
    # edges.modeling sits below edges.io in the layering: it may only depend on
    # the low-level helpers (serialization, tools, types).
    mods = _modules_after("import edges.modeling", stub_root=True)
    edges_mods = {m for m in mods if m.startswith("edges.")}
    assert "edges.modeling" in edges_mods
    assert not {m for m in edges_mods if m.startswith(("edges.io", "edges.cal"))}
    assert edges_mods <= {
        "edges.serialization",
        "edges.tools",
        "edges.types",
        *{m for m in edges_mods if m.startswith("edges.modeling")},
    }
    assert "pygsdata" not in mods


def test_serialization_is_low_level():
    mods = _modules_after("import edges.serialization", stub_root=True)
    assert {m for m in mods if m.startswith("edges.")} == {
        "edges.serialization",
        "edges.types",
    }


@pytest.mark.parametrize(
    "module", ["edges.io.calobsdef", "edges.cal.spectra", "edges.filters.filters"]
)
def test_subpackages_do_not_import_root_package(module):
    # These modules used to import DATA_PATH/__version__ from ``edges`` itself,
    # which only worked because of the import order in edges/__init__.py. With a
    # stub root package, they must still import cleanly.
    mods = _modules_after(f"import {module}", stub_root=True)
    assert module in mods


def test_import_edges_does_not_import_pyplot():
    mods = _modules_after("import edges")
    assert "matplotlib.pyplot" not in mods


def test_import_edges_does_not_import_lazy_subpackages():
    mods = _modules_after("import edges")
    for name in ("edges.alanmode", "edges.inference", "emcee", "pooch"):
        assert name not in mods


def test_import_edges_registers_all_gsdata_processors():
    # ``import edges`` must register every processor that any edges module
    # registers with pygsdata (users look them up by name in GSDATA_PROCESSORS).
    out = _run(
        """
        import importlib, pkgutil
        import edges
        from pygsdata import GSDATA_PROCESSORS
        before = set(GSDATA_PROCESSORS)
        for m in pkgutil.walk_packages(edges.__path__, "edges."):
            if not m.name.endswith("cqrd"):  # the C-extension is loaded via ctypes
                importlib.import_module(m.name)
        print(sorted(set(GSDATA_PROCESSORS) - before))
        """
    )
    assert out == "[]"


@pytest.mark.parametrize("name", ["alanmode", "inference"])
def test_lazy_subpackages_accessible(name):
    out = _run(
        f"""
        import edges, sys
        assert "edges.{name}" not in sys.modules
        assert "{name}" in dir(edges)
        mod = edges.{name}
        from edges import {name} as mod2
        import edges.{name} as mod3
        print(mod is mod2 is mod3 is sys.modules["edges.{name}"])
        """
    )
    assert out == "True"


def test_root_dir_lists_lazy_subpackages():
    names = dir(edges)
    assert {"alanmode", "inference"} <= set(names)
    assert names == sorted(names)


def test_root_getattr_unknown():
    with pytest.raises(AttributeError, match="no attribute 'not_a_thing'"):
        _ = edges.not_a_thing


def test_public_names_present():
    for name in edges.__all__:
        assert getattr(edges, name) is not None


def test_version_consistent():
    from edges._version import __version__

    assert edges.__version__ == __version__


def test_data_paths():
    from edges import data, filters

    assert edges.DATA_PATH == data.DATA_PATH
    assert Path(edges.__file__).parent / "data" == edges.DATA_PATH
    assert data.BEAM_PATH == data.DATA_PATH / "beams"
    assert data.LOSS_PATH == data.DATA_PATH / "loss"
    # The filters data path is a different directory.
    assert Path(filters.__file__).parent / "data" == filters.DATA_PATH
    assert filters.DATA_PATH != data.DATA_PATH
    assert edges.get_data_path is data.get_data_path
    assert edges.get_data_path(":beams") == data.BEAM_PATH
    assert edges.get_data_path("beams") == Path("beams")
    assert edges.get_data_path(Path(":x")) == Path(":x")


def test_data_package_is_light():
    mods = _modules_after("import edges.data", stub_root=True)
    assert "pooch" not in mods
    assert "edges.data.cli" not in mods


def test_data_package_lazy_attributes():
    out = _run(
        """
        import sys
        import edges.data as d
        assert "pooch" not in sys.modules
        f = d.fetch_b18cal_full
        from edges.data import _pooch
        print(f is _pooch.fetch_b18cal_full, d.cli is sys.modules["edges.data.cli"])
        """
    )
    assert out == "True True"

    import edges.data

    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        _ = edges.data.nope
    assert "fetch_b18cal_full" in dir(edges.data)


def test_cli_registers_data_command_without_import_side_effects():
    out = _run(
        """
        from edges.cli import app
        print("data" in app)
        """
    )
    assert out == "True"


def test_testing_sky_model_is_lazy():
    # The sky-model file is not read on import, only when first needed.
    out = _run(
        """
        import edges.testing as t
        n0 = t._sky_model.cache_info().currsize
        t.create_mock_edges_data(ntime=3)
        print(n0, t._sky_model.cache_info().currsize)
        """
    )
    assert out.splitlines()[-1] == "0 1"

    from edges import testing

    assert testing.msky.shape == (72,)
    assert testing.mgha.shape == (72,)
    assert set(testing.model) == {"freqs", "sky_temp", "lsts"}
    assert testing.spl75(testing.mgha[3]) == pytest.approx(testing.msky[3], rel=1e-12)
    with pytest.raises(AttributeError):
        _ = testing.not_a_thing
