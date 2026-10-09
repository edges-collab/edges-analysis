"""Tests of the edges.io.spectra module."""

import logging
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
from astropy import units as un
from astropy.time import Time

from edges.const import KNOWN_TELESCOPES
from edges.io import TEST_DATA_PATH
from edges.io.spectra import (
    acq_time_ranges,
    get_acq_integration_time,
    infer_telescope_from_acq_header,
    read_acq_header,
    read_spectra,
)

# GSData stores times as float64 Julian dates, precise to ~50 microseconds.
TIME_ATOL = 1e-4  # seconds

# read_acq warns about empty header items (e.g. --output_file) in these files.
pytestmark = pytest.mark.filterwarnings(
    "ignore:(?s)In file .* has no value:UserWarning"
)

EDGES3_AMB = TEST_DATA_PATH / "edges3-mock-root/mro/amb/2023/2023_070_12_43_06_amb.acq"


def _write_variant(tmp_path: Path, name: str, **header) -> Path:
    """Copy the EDGES-3 mock file, replacing (or removing, if None) header items."""
    txt = EDGES3_AMB.read_text()
    for key, val in header.items():
        pattern = rf"^;--{key}: .*\n"
        repl = "" if val is None else f";--{key}: {val}\n"
        txt = re.sub(pattern, repl, txt, flags=re.MULTILINE)
    out = tmp_path / name
    out.write_text(txt)
    return out


def _write_pxspec(tmp_path: Path) -> Path:
    """Copy the EDGES-3 mock file without any header, as written by pxspec."""
    lines = EDGES3_AMB.read_text().splitlines(keepends=True)
    out = tmp_path / "pxspec.acq"
    out.write_text("".join(ln for ln in lines if not ln.startswith(";")))
    return out


# Real (site, instrument, datadir) combinations from a survey of the EDGES archive.
@pytest.mark.parametrize(
    ("site", "instrument", "datadir", "expected"),
    [
        ("mro", "ant", "/home/edges/edges3/MRO", "edges3"),
        ("mro", "amb", "/home/edges/edges3/MRO", "edges3"),
        ("hmp", "ant", "/home/edges/edges3/devon", "edges3-devon"),
        # EDGES-3 systems without a preset.
        ("hmp", "loads", "/home/edges/edges3/devon", None),
        ("adak", "loads", "/home/edges/edges3_adak/ADAK", None),
        ("mro", "loads", "/home/edges/edges3_adak/MRO", None),
        # The Adak system reported site=mro during its first days at Adak.
        ("mro", "ant", "/home/edges/edges3_adak/MRO", None),
        # EDGES-2 bands: no preset distinguishes them.
        ("mro", "low2", "/home/loco/edges/data", None),
        ("mro", "low2-128k", "/home/loco/edges/data", None),
        ("mro", "mid", "/home/loco/edges/data", None),
        # Lab tests and RFI surveys.
        ("hay", "loads", "/home/edges/quick_test", None),
        ("asulab", "lab", "/home/edges-0/Desktop/DATA/", None),
        ("lab", "razormax", "/home/jdbowman/data", None),
        ("meteor-crater-1", "rfi", "/home/jdbowman/data", None),
    ],
)
def test_infer_telescope(site, instrument, datadir, expected):
    meta = {"site": site, "instrument": instrument, "datadir": datadir}
    assert infer_telescope_from_acq_header(meta) == expected


def test_infer_telescope_no_header():
    assert infer_telescope_from_acq_header({}) is None


def test_integration_time_from_header():
    meta = read_acq_header(EDGES3_AMB)
    intg = get_acq_integration_time(meta)
    np.testing.assert_allclose(intg.to_value(un.s), 2684354560 / 400e6, rtol=1e-12)


def test_integration_time_missing():
    assert get_acq_integration_time({}) is None


def test_read_edges3_acq(caplog):
    """Regression test for #298: EDGES-3 data was read as edges-low with 13s."""
    with caplog.at_level(logging.INFO, logger="edges.io.spectra"):
        data = read_spectra([EDGES3_AMB])
    assert "Inferred telescope 'edges3'" in caplog.text
    assert data.telescope.name == "edges3"
    np.testing.assert_allclose(
        data.effective_integration_time.to_value(un.s), 6.7108864, rtol=1e-12
    )
    # Time ranges come from the time stamps, not the integration time. The mock file
    # has cycles starting at 12:43:06, 12:43:30 and 12:43:53.
    dt = (data.time_ranges[..., 1] - data.time_ranges[..., 0]).to_value(un.s)
    np.testing.assert_allclose(dt[:, 0], [24 / 3, 23 / 3, 23.5 / 3], atol=TIME_ATOL)
    np.testing.assert_allclose(data.time_ranges[0, 0, 0].jd, data.times[0, 0].jd)
    # LST ranges must be consistent with the new time ranges.
    np.testing.assert_allclose(
        data.lst_ranges.hour,
        data.time_ranges.sidereal_time("apparent", data.telescope.location).hour,
    )


def _cycle_times(offsets_s, nloads=3):
    t0 = Time("2025:100:04:01:19", format="yday", scale="utc")
    starts = t0 + np.asarray(offsets_s, dtype=float) * un.s
    return starts[:, None][:, np.zeros(nloads, dtype=int)]


def test_acq_time_ranges_tile_the_observation():
    times = _cycle_times([0, 24, 47, 71])
    tr = acq_time_ranges(times)
    assert tr.shape == (4, 3, 2)
    # Each switch position starts where the previous one ended.
    flat = tr.jd.reshape(-1, 2)
    np.testing.assert_allclose(flat[1:, 0], flat[:-1, 1], rtol=0, atol=1e-9)
    # Positions are an equal share of their cycle.
    dt = (tr[..., 1] - tr[..., 0]).to_value(un.s)
    np.testing.assert_allclose(
        dt[:3], np.array([[8.0] * 3, [23 / 3] * 3, [8.0] * 3]), atol=TIME_ATOL
    )
    # The last cycle gets the median duration.
    np.testing.assert_allclose(dt[3], 8.0, atol=TIME_ATOL)


def test_acq_time_ranges_gap():
    # A gap (e.g. between files) doesn't stretch the cycle before it.
    times = _cycle_times([0, 23, 46, 3600, 3623])
    tr = acq_time_ranges(times)
    dt = (tr[..., 1] - tr[..., 0]).to_value(un.s)
    np.testing.assert_allclose(dt, 23 / 3, atol=TIME_ATOL)


def test_acq_time_ranges_file_gaps():
    # Two short files an hour apart: the median must not be set by the gap.
    times = _cycle_times([0, 23, 3600, 3623])
    tr = acq_time_ranges(times, max_cycle_duration=40 * un.s)
    dt = (tr[..., 1] - tr[..., 0]).to_value(un.s)
    np.testing.assert_allclose(dt, 23 / 3, atol=TIME_ATOL)

    # Single-cycle files an hour apart have no plausible cycle at all.
    assert (
        acq_time_ranges(_cycle_times([0, 3600]), max_cycle_duration=40 * un.s) is None
    )


def test_acq_time_ranges_single_cycle():
    assert acq_time_ranges(_cycle_times([0])) is None


def test_read_single_cycle_warns(tmp_path):
    # Keep the header and the first cycle (three switch positions) only.
    keep, ndata = [], 0
    for ln in EDGES3_AMB.read_text().splitlines(keepends=True):
        keep.append(ln)
        ndata += not ln.startswith((";", "#"))
        if ndata == 3:
            break
    fl = tmp_path / "one_cycle.acq"
    fl.write_text("".join(keep))
    with pytest.warns(UserWarning, match="Cannot determine the cycle duration"):
        data = read_spectra([fl])
    dt = (data.time_ranges[..., 1] - data.time_ranges[..., 0]).to_value(un.s)
    np.testing.assert_allclose(dt, 2684354560 / 400e6, atol=TIME_ATOL)


def test_read_devon_acq(tmp_path):
    fl = _write_variant(
        tmp_path, "devon.acq", site="hmp", datadir="/home/edges/edges3/devon"
    )
    assert read_spectra([fl]).telescope.name == "edges3-devon"


def test_read_acq_explicit_telescope(tmp_path):
    # No warning when the telescope is given, even if it can't be inferred.
    fl = _write_variant(tmp_path, "low2.acq", instrument="low2")
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Could not determine")
        data = read_spectra([fl], telescope="edges-low")
    assert data.telescope.name == "edges-low"

    data = read_spectra([EDGES3_AMB], telescope="edges-low")
    assert data.telescope.name == "edges-low"

    data = read_spectra([EDGES3_AMB], telescope=KNOWN_TELESCOPES["edges3-devon"])
    assert data.telescope.name == "edges3-devon"


def test_read_edges2_fastspec(tmp_path):
    fl = _write_variant(
        tmp_path,
        "low2.acq",
        instrument="low2",
        samples_per_accumulation=4294967296,
    )
    with pytest.warns(UserWarning, match="Could not determine the telescope"):
        data = read_spectra([fl])
    assert data.telescope.name == "edges-low"
    np.testing.assert_allclose(
        data.effective_integration_time.to_value(un.s), 2**32 / 400e6, rtol=1e-12
    )


def test_read_pxspec_uses_telescope_integration_time(tmp_path):
    with pytest.warns(UserWarning, match="Could not determine the telescope"):
        data = read_spectra([_write_pxspec(tmp_path)])
    assert data.telescope.name == "edges-low"
    np.testing.assert_allclose(
        data.effective_integration_time.to_value(un.s),
        KNOWN_TELESCOPES["edges-low"].integration_time.to_value(un.s),
    )


def test_read_mixed_telescopes_raises(tmp_path):
    low = _write_variant(tmp_path, "low2.acq", instrument="low2")
    with pytest.raises(ValueError, match="different telescopes"):
        read_spectra([EDGES3_AMB, low])


def test_read_mixed_integration_times_raises(tmp_path):
    other = _write_variant(tmp_path, "other.acq", samples_per_accumulation=4294967296)
    with pytest.raises(ValueError, match="different integration times"):
        read_spectra([EDGES3_AMB, other])


def test_read_bad_format():
    with pytest.raises(ValueError, match="not supported"):
        read_spectra([Path("file.txt")])


def test_read_no_files():
    with pytest.raises(ValueError, match="No spectrum files given"):
        read_spectra([])


def test_read_mixed_formats_raises(tmp_path):
    with pytest.raises(ValueError, match="same format"):
        read_spectra([EDGES3_AMB, tmp_path / "other.gsh5"])


@pytest.mark.parametrize("as_list", [True, False])
def test_read_str_paths(as_list: bool):
    expected = read_spectra([EDGES3_AMB])
    data = read_spectra([str(EDGES3_AMB)] if as_list else str(EDGES3_AMB))
    np.testing.assert_array_equal(data.data, expected.data)
    assert data.telescope.name == "edges3"
