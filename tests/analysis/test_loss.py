"""Test the loss module."""

from pathlib import Path

import numpy as np
import pytest
from astropy import units as un

from edges.analysis import loss
from edges.cal.sparams import ReflectionCoefficient
from edges.data import DATA_PATH


def test_no_band():
    with pytest.raises(ValueError, match="you must provide 'band'"):
        loss.ground_loss(filename=":", freq=np.linspace(50, 100, 100))


class TestLow2BalunConnectorLoss:
    @pytest.mark.parametrize("use_approx_eps0", [True, False])
    def test_happy_path(self, use_approx_eps0):
        fq = np.linspace(50, 100, 51) * un.MHz
        ants11 = np.ones(51) / 2

        bcloss = loss.low2_balun_connector_loss(
            ReflectionCoefficient(reflection_coefficient=ants11, freqs=fq),
            use_approx_eps0=use_approx_eps0,
        )

        assert bcloss.shape == fq.shape

    def test_other_s11_inputs(self, tmp_path: Path):
        s11 = ReflectionCoefficient(
            freqs=np.linspace(50, 100, 100) * un.MHz,
            reflection_coefficient=np.zeros(100, dtype=complex),
        )

        # S11Model object
        bcloss = loss.low2_balun_connector_loss(ants11=s11)
        assert bcloss.shape == s11.freqs.shape

        # Hickled file
        s11.write(tmp_path / "tmp-ant.h5")
        bcloss2 = loss.low2_balun_connector_loss(tmp_path / "tmp-ant.h5")
        assert np.allclose(bcloss, bcloss2)


def _read_tabulated_loss(instrument: str, name: str) -> np.ndarray:
    return np.genfromtxt(DATA_PATH / "loss" / instrument / f"{name}.txt")


@pytest.mark.parametrize(
    ("instrument", "configuration"), [("low", ""), ("low", "45deg"), ("mid", "")]
)
def test_builtin_ground_loss_matches_tabulated(instrument: str, configuration: str):
    name = f"ground_{configuration}" if configuration else "ground"
    table = _read_tabulated_loss(instrument, name)

    gl = loss.ground_loss(
        table[:, 0], ":", instrument=instrument, configuration=configuration
    )
    assert not np.allclose(gl, 1.0)
    # Tabulated losses are ~1e-3; the polynomial fit reproduces them to <3e-5.
    np.testing.assert_allclose(gl, 1 - table[:, 1], atol=5e-5, rtol=0)


def test_builtin_antenna_loss_reads_antenna_file():
    table = _read_tabulated_loss("mid", "antenna")
    al = loss.antenna_loss(table[:, 0], ":", instrument="mid")
    np.testing.assert_allclose(al, 1 - table[:, 1], atol=5e-5, rtol=0)


def test_antenna_loss_uses_n_terms():
    fname = DATA_PATH / "loss" / "mid" / "antenna.txt"
    table = np.genfromtxt(fname)
    freq = np.linspace(50, 150, 101)

    expected = 1 - np.polyval(np.polyfit(table[:, 0], table[:, 1], 10), freq)
    al = loss.antenna_loss(freq, fname)
    np.testing.assert_allclose(al, expected, rtol=0, atol=1e-12)


def test_ground_loss_is_degree_8_fit():
    """The ground loss is a 9-term (degree-8) polynomial fit to the tabulated loss."""
    fname = DATA_PATH / "loss" / "low" / "ground.txt"
    table = np.genfromtxt(fname)
    freq = np.linspace(50, 120, 71)

    expected = 1 - np.polyval(np.polyfit(table[:, 0], table[:, 1], 8), freq)
    gl = loss.ground_loss(freq, fname)
    np.testing.assert_allclose(gl, expected, rtol=0, atol=1e-12)


def test_builtin_loss_file_missing_raises():
    # There is no built-in antenna-loss file for the low-band instrument.
    with pytest.raises(FileNotFoundError, match="No built-in antenna loss file"):
        loss.antenna_loss(np.linspace(50, 100, 11), ":", instrument="low")

    with pytest.raises(FileNotFoundError, match="ground_90deg"):
        loss.ground_loss(
            np.linspace(50, 100, 11), ":", instrument="low", configuration="90deg"
        )
