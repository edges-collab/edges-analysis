from pathlib import Path

import numpy as np
import pytest
from astropy import units as un

from edges.io.vna import SParams, read_s1p


def test_s1p_read(datadir: Path):
    fl = (
        datadir / "Receiver01_25C_2019_11_26_040_to_200MHz/S11/Ambient01/External01.s1p"
    )
    s1p = SParams.from_s1p_file(fl)

    assert np.all(np.iscomplex(s1p.s11))
    assert len(s1p.s11) == len(s1p.freq)


def test_s1_read_db(datadir: Path):
    fl = datadir / "s11_db.s1p"
    s1p = SParams.from_s1p_file(fl)

    assert np.all(np.iscomplex(s1p.s11))
    assert len(s1p.s11) == len(s1p.freq)


def test_bad_s11_input_to_vna():
    with pytest.raises(ValueError, match="s-parameters must be complex"):
        SParams(freq=np.linspace(50, 100, 100) * un.MHz, s11=np.linspace(0, 1, 100))

    with pytest.raises(ValueError, match="Shape of s11 does not match"):
        SParams(freq=np.linspace(50, 100, 100) * un.MHz, s11=np.linspace(0, 1, 70) + 0j)


# ---------------------------------------------------------------------------------
# Touchstone parsing on synthetic files with known values.
# ---------------------------------------------------------------------------------
FREQS_HZ = np.array([40e6, 50e6, 60e6, 70e6])
S11 = 0.3 * np.exp(1j * np.linspace(0, 3, 4))
S21 = 0.9 * np.exp(-1j * np.linspace(0.5, 2.5, 4))
S12 = 0.1 * np.exp(1j * np.linspace(-1, 1, 4))
S22 = 0.2 * np.exp(1j * np.linspace(2, -2, 4))


def _pair(z: complex, fmt: str) -> tuple[float, float]:
    if fmt == "RI":
        return z.real, z.imag
    ang = np.angle(z, deg=True)
    if fmt == "MA":
        return np.abs(z), ang
    return 20 * np.log10(np.abs(z)), ang


def _write_touchstone(
    path: Path,
    params: list[np.ndarray],
    fmt: str = "RI",
    unit: str = "Hz",
    freqs_hz: np.ndarray = FREQS_HZ,
    option_line: str | None = None,
    header: str = "! Synthetic test file\n",
    footer: str = "",
    sep: str = " ",
):
    scale = {"hz": 1, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}[unit.lower()]
    if option_line is None:
        option_line = f"# {unit} S {fmt} R 50"
    lines = [header + option_line]
    for i, f in enumerate(freqs_hz):
        vals = [f"{f / scale:.15g}"]
        for p in params:
            vals += [f"{v:.15g}" for v in _pair(p[i], fmt)]
        lines.append(sep.join(vals))
    path.write_text("\n".join(lines) + "\n" + footer)
    return path


@pytest.mark.parametrize("fmt", ["RI", "MA", "DB"])
def test_s2p_read_column_order(tmp_path: Path, fmt: str):
    """Touchstone v1 s2p columns are f, S11, S21, S12, S22 (all formats)."""
    fl = _write_touchstone(tmp_path / "net.s2p", [S11, S21, S12, S22], fmt=fmt)
    sp = SParams.from_s1p_file(fl)

    np.testing.assert_allclose(sp.freq.to_value("Hz"), FREQS_HZ, rtol=1e-15)
    np.testing.assert_allclose(sp.s11, S11, rtol=1e-12)
    np.testing.assert_allclose(sp.s21, S21, rtol=1e-12)
    np.testing.assert_allclose(sp.s12, S12, rtol=1e-12)
    np.testing.assert_allclose(sp.s22, S22, rtol=1e-12)


@pytest.mark.parametrize("fmt", ["RI", "MA", "DB"])
def test_s1p_formats_agree(tmp_path: Path, fmt: str):
    fl = _write_touchstone(tmp_path / "a.s1p", [S11], fmt=fmt)
    t = read_s1p(fl)
    np.testing.assert_allclose(t["s11"], S11, rtol=1e-12)
    np.testing.assert_allclose(t["frequency"].to_value("Hz"), FREQS_HZ, rtol=1e-15)
    assert "s21" not in t.colnames


@pytest.mark.parametrize("unit", ["Hz", "kHz", "MHz", "GHz", "mhz", "GHZ"])
def test_frequency_unit_from_option_line(tmp_path: Path, unit: str):
    fl = _write_touchstone(tmp_path / "a.s1p", [S11], unit=unit)
    t = read_s1p(fl)
    np.testing.assert_allclose(t["frequency"].to_value("Hz"), FREQS_HZ, rtol=1e-14)
    np.testing.assert_allclose(t["s11"], S11, rtol=1e-12)


@pytest.mark.parametrize(
    "option_line",
    ["# mhz s ri r 50", "#MHz S RI R 50", "# MHz  S  RI  R  50", "# MHz RI S R 50"],
)
def test_option_line_case_and_spacing(tmp_path: Path, option_line: str):
    fl = _write_touchstone(
        tmp_path / "a.s1p", [S11], unit="MHz", option_line=option_line
    )
    t = read_s1p(fl)
    np.testing.assert_allclose(t["frequency"].to_value("Hz"), FREQS_HZ, rtol=1e-14)
    np.testing.assert_allclose(t["s11"], S11, rtol=1e-12)


def test_db_lowercase_flag(tmp_path: Path):
    fl = _write_touchstone(
        tmp_path / "a.s1p", [S11], fmt="DB", option_line="# Hz S dB R 50"
    )
    np.testing.assert_allclose(read_s1p(fl)["s11"], S11, rtol=1e-12)


def test_reference_impedance(tmp_path: Path):
    fl = _write_touchstone(tmp_path / "a.s1p", [S11])
    assert read_s1p(fl).meta["reference_impedance"] == 50 * un.ohm

    fl = _write_touchstone(tmp_path / "b.s1p", [S11], option_line="# Hz S RI R 75")
    with pytest.warns(UserWarning, match="reference impedance of 75"):
        t = read_s1p(fl)
    assert t.meta["reference_impedance"] == 75 * un.ohm


def test_single_row(tmp_path: Path):
    fl = _write_touchstone(
        tmp_path / "a.s1p", [S11[:1]], freqs_hz=FREQS_HZ[:1], fmt="MA"
    )
    t = read_s1p(fl)
    assert len(t) == 1
    np.testing.assert_allclose(t["s11"], S11[:1], rtol=1e-12)


def test_trailing_end_and_blank_lines(tmp_path: Path):
    fl = _write_touchstone(tmp_path / "a.s1p", [S11], footer="END\n\n\n")
    t = read_s1p(fl)
    assert len(t) == len(FREQS_HZ)
    np.testing.assert_allclose(t["s11"], S11, rtol=1e-12)


def test_inline_comments(tmp_path: Path):
    fl = _write_touchstone(tmp_path / "a.s1p", [S11])
    lines = fl.read_text().splitlines()
    # Add an inline comment to the option line and to every data line,
    # and a blank line and a full-line comment in the middle of the data.
    lines[1] += " ! option line comment"
    lines = [*lines[:2], *(ln + "  ! a comment" for ln in lines[2:])]
    lines.insert(3, "")
    lines.insert(4, "! full line comment")
    fl.write_text("\n".join(lines) + "\n")

    t = read_s1p(fl)
    np.testing.assert_allclose(t["frequency"].to_value("Hz"), FREQS_HZ, rtol=1e-15)
    np.testing.assert_allclose(t["s11"], S11, rtol=1e-12)


def test_comma_separated(tmp_path: Path):
    fl = _write_touchstone(tmp_path / "a.s1p", [S11], sep=",")
    np.testing.assert_allclose(read_s1p(fl)["s11"], S11, rtol=1e-12)


@pytest.mark.parametrize("fmt", ["RI", "DB", "db", "MA"])
def test_begin_end_format(tmp_path: Path, fmt: str):
    """Files with a BEGIN/<FMT>/.../END wrapper (frequencies in Hz)."""
    rows = [
        " ".join(f"{v:.15g}" for v in (f, *_pair(s, fmt.upper())))
        for f, s in zip(FREQS_HZ, S11, strict=True)
    ]
    fl = tmp_path / "a.s1p"
    fl.write_text("BEGIN\n" + fmt + "\n" + "\n".join(rows) + "\nEND\n")
    t = read_s1p(fl)
    np.testing.assert_allclose(t["frequency"].to_value("Hz"), FREQS_HZ, rtol=1e-15)
    np.testing.assert_allclose(t["s11"], S11, rtol=1e-12)


def test_frequency_range(tmp_path: Path):
    fl = _write_touchstone(tmp_path / "a.s2p", [S11, S21, S12, S22], unit="MHz")
    t = read_s1p(fl, f_low=45 * un.MHz, f_high=65 * un.MHz)
    np.testing.assert_allclose(t["frequency"].to_value("MHz"), [50, 60])
    np.testing.assert_allclose(t["s21"], S21[1:3], rtol=1e-12)


def test_no_option_line_raises(tmp_path: Path):
    fl = tmp_path / "a.s1p"
    fl.write_text("! only a comment\n")
    with pytest.raises(OSError, match="incorrect format"):
        read_s1p(fl)


def test_unsupported_number_of_columns(tmp_path: Path):
    fl = tmp_path / "a.s1p"
    fl.write_text("# Hz S RI R 50\n1 2 3 4 5\n")
    with pytest.raises(ValueError, match="columns"):
        read_s1p(fl)


def test_pseudo_option_line_ignored(tmp_path: Path):
    """Some VNAs write '#<number>!...' lines before the real option line."""
    fl = _write_touchstone(
        tmp_path / "a.s1p",
        [S11],
        unit="MHz",
        header="BEGIN\n#48062!Agilent N9923A: A.07.51\n!Date: today\n",
        footer="END\n",
    )
    t = read_s1p(fl)
    np.testing.assert_allclose(t["frequency"].to_value("Hz"), FREQS_HZ, rtol=1e-14)
    np.testing.assert_allclose(t["s11"], S11, rtol=1e-12)
