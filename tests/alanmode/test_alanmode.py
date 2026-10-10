"""Unit-tests of functions in alanmode."""

from pathlib import Path

import attrs
import numpy as np
import pytest
from astropy import units as un
from astropy.time import Time
from pygsdata import KNOWN_TELESCOPES, GSData, GSFlag
from read_acq.gsdata import write_gsdata_to_acq

from edges import alanmode as am
from edges import modeling as mdl
from edges.alanmode.alanmode import _average_spectra
from edges.cal import ReflectionCoefficient
from edges.config import config
from edges.data import fetch_b18_cal_outputs
from edges.frequencies import edges_raw_freqs


def test_read_write_spec_loop(alanmode_data_path: Path, tmpdir: Path):
    spamb = am.read_spec_txt(alanmode_data_path / "edges3-2022-316-raw/spambient.txt")

    am.write_spec_txt(
        spamb.freqs,
        n=int(np.mean(spamb.nsamples)),
        spec=spamb.data.squeeze(),
        fname=tmpdir / "spamb.txt",
    )
    spamb2 = am.read_spec_txt(tmpdir / "spamb.txt")
    np.testing.assert_allclose(spamb.freqs, spamb2.freqs)
    np.testing.assert_allclose(spamb.data, spamb2.data)
    np.testing.assert_allclose(spamb.nsamples, spamb2.nsamples)


def _make_spec_gsd(nfreq: int = 64, flagged=(3, 10, 11), scale: float = 300.0):
    freqs = np.linspace(50, 100, nfreq) * un.MHz
    rng = np.random.default_rng(3)
    data = scale + rng.uniform(-1, 1, nfreq)
    flags = np.zeros(nfreq, dtype=bool)
    flags[list(flagged)] = True
    return GSData(
        data=data[None, None, None],
        freqs=freqs,
        times=Time([[2459856.0]], format="jd"),
        telescope=KNOWN_TELESCOPES["edges-low"],
        nsamples=np.full((1, 1, 1, nfreq), 450.0),
        flags={"flags": GSFlag(flags, axes=("freq",))},
        data_unit="uncalibrated_temp",
    )


def test_spec_txt_roundtrip_keeps_flags(tmp_path: Path):
    """Regression test: written weights were always 1."""
    gsd = _make_spec_gsd()
    am.write_spec_txt_gsd(gsd, tmp_path / "sp.txt")
    new = am.read_spec_txt(tmp_path / "sp.txt")

    np.testing.assert_array_equal(new.complete_flags, gsd.complete_flags)
    np.testing.assert_allclose(new.freqs, gsd.freqs, atol=1e-6 * un.MHz)
    np.testing.assert_allclose(new.data, gsd.data, atol=1e-6, rtol=0)
    np.testing.assert_allclose(new.flagged_nsamples, gsd.flagged_nsamples)


def test_write_spec_txt_weights(tmp_path: Path):
    freqs = np.linspace(50, 100, 5)
    weights = np.array([1, 0, 1, 1, 0])
    am.write_spec_txt(freqs, n=10, spec=np.ones(5), fname=tmp_path / "sp.txt")
    am.write_spec_txt(
        freqs, n=10, spec=np.ones(5), fname=tmp_path / "spw.txt", weights=weights
    )
    np.testing.assert_array_equal(
        np.genfromtxt(tmp_path / "sp.txt", comments="/", usecols=2), 1
    )
    np.testing.assert_array_equal(
        np.genfromtxt(tmp_path / "spw.txt", comments="/", usecols=2), weights
    )


def test_read_spec_txt_wide_values(tmp_path: Path):
    """Regression test: n was read from a fixed column of the first line."""
    gsd = _make_spec_gsd(flagged=(), scale=1.0e7)
    am.write_spec_txt_gsd(gsd, tmp_path / "sp.txt")
    new = am.read_spec_txt(tmp_path / "sp.txt")
    np.testing.assert_allclose(new.nsamples, 450.0)
    np.testing.assert_allclose(new.data, gsd.data, atol=1e-6, rtol=0)


NTIME = 24
FREQS = edges_raw_freqs()


@pytest.fixture(scope="module")
def unity_acq(tmpdir):
    data = np.ones((3, 1, NTIME, len(FREQS))) / 1000.0
    data[0] *= 4.0
    data[2] *= 4.0

    times = np.linspace(2459856, 2459857, NTIME + 1)[:-1]
    times = np.array([times, times + 0.1, times + 0.2]).T
    times = Time(times, format="jd", scale="utc")

    data = GSData(
        data=data,
        freqs=FREQS,
        times=times,
        telescope=KNOWN_TELESCOPES["edges-low"],
        auxiliary_measurements={
            "adcmax": np.zeros_like(times),
            "adcmin": np.zeros_like(times),
        },
    )
    write_gsdata_to_acq(data, tmpdir / "unity.acq")

    return tmpdir / "unity.acq"


def test_read_all_spec_text():
    datadir = fetch_b18_cal_outputs() / "H2Case-fittpfix"
    freqs, specs = am.read_all_spec_txt(datadir, flow=50 * un.MHz, fhigh=100 * un.MHz)
    assert freqs.min() >= 50 * un.MHz
    assert freqs.max() <= 100 * un.MHz
    assert len(specs) == 4  # four files in that directory
    assert all(spec.shape == (len(freqs),) for spec in specs.values())


class TestACQPlot7AMoon:
    def test_unity_default(self, unity_acq):
        meanspec = am.acqplot7amoon(
            unity_acq, fstart=0, fstop=np.inf, smooth=0, tload=300, tcal=1000
        )

        assert meanspec.nfreqs == FREQS.size
        assert meanspec.nsamples.max() == NTIME
        np.testing.assert_allclose(meanspec.data, 1300)

    def test_unity_one_hour(self, unity_acq):
        meanspec = am.acqplot7amoon(
            unity_acq,
            fstart=0,
            fstop=np.inf,
            smooth=0,
            tload=300,
            tcal=1000,
            tstart=12.0,
            tstop=12.0,
        )

        assert meanspec.nfreqs == FREQS.size
        assert meanspec.nsamples.max() == 1
        np.testing.assert_allclose(meanspec.data, 1300)

    def test_unity_delaystart(self, unity_acq):
        meanspec = am.acqplot7amoon(
            unity_acq,
            fstart=0,
            fstop=np.inf,
            smooth=0,
            tload=300,
            tcal=1000,
            delaystart=1.0,  # second
        )

        assert meanspec.nfreqs == FREQS.size
        assert meanspec.nsamples.max() == NTIME - 1
        np.testing.assert_allclose(meanspec.data, 1300)

    def test_unity_smooth(self, unity_acq):
        meanspec = am.acqplot7amoon(
            unity_acq,
            fstart=0,
            fstop=np.inf,
            smooth=8,
            tload=300,
            tcal=1000,
        )

        assert meanspec.nfreqs == FREQS.size // 8
        # Effective nsamples of the Gaussian-weighted mean, (sum k)^2 / sum(k^2),
        # is 12.0705 per spectrum for smooth=8 in the band interior.
        assert np.isclose(np.median(meanspec.nsamples), NTIME * 12.0705, rtol=1e-3)
        np.testing.assert_allclose(meanspec.data, 1300)

    def test_params_and_kwargs(self, unity_acq):
        """Regression test: kwargs used to replace params entirely."""
        params = am.ACQPlot7aMoonParams(smooth=0, tload=300, tcal=1000)
        meanspec = am.acqplot7amoon(unity_acq, params=params, fstart=0, fstop=np.inf)

        # smooth=0 from params is kept (the default would be smooth=8).
        assert meanspec.nfreqs == FREQS.size
        np.testing.assert_allclose(meanspec.data, 1300)


@pytest.mark.parametrize(
    ("smooth", "delaystart", "n"), [(8, 0, NTIME), (0, 1, NTIME - 1)]
)
def test_averaged_spectrum_file_holds_number_of_spectra(
    unity_acq, tmp_path: Path, smooth: int, delaystart: int, n: int
):
    """The header of the averaged-spectrum file holds the number of spectra.

    As in the C-code, it is the number of spectra that were averaged, not their
    effective number of samples (which frequency smoothing increases).
    """
    spectra = _average_spectra(
        {"ambient": [unity_acq]},
        out=tmp_path,
        redo_spectra=True,
        fstart=0,
        fstop=np.inf,
        telescope="edges-low",
        smooth=smooth,
        tload=300,
        tcal=1000,
        delaystart=delaystart,
    )

    with (tmp_path / "spambient.txt").open() as fl:
        assert int(fl.readline().split()[3]) == n
    assert np.max(spectra["ambient"].nsamples) == n


def test_write_spec_txt_gsd_explicit_n(tmp_path: Path):
    gsd = _make_spec_gsd()
    am.write_spec_txt_gsd(gsd, tmp_path / "sp.txt", n=7)
    new = am.read_spec_txt(tmp_path / "sp.txt")
    np.testing.assert_allclose(new.flagged_nsamples, 7.0 * (gsd.flagged_nsamples > 0))


class TestEdgesParams:
    """Regression tests: params given with kwargs were silently dropped."""

    class _Stop(Exception):  # ruff: ignore[error-suffix-on-exception-name]
        pass

    def _params_used(self, monkeypatch, **kwargs) -> am.EdgesScriptParams:
        def stop(spcold, sphot, spopen, spshort, params, *args):
            raise self._Stop(params)

        monkeypatch.setattr(am.alanmode, "_get_specs", stop)
        with pytest.raises(self._Stop) as exc:
            am.edges(
                spcold=None,
                sphot=None,
                spopen=None,
                spshort=None,
                s11freq=None,
                s11hot=None,
                s11cold=None,
                s11lna=None,
                s11open=None,
                s11short=None,
                tload=300 * un.K,
                tcal=1000 * un.K,
                **kwargs,
            )
        return exc.value.args[0]

    def test_params_with_kwargs(self, monkeypatch):
        params = am.EdgesScriptParams(cfit=4, wfit=3)
        used = self._params_used(monkeypatch, params=params, nfit3=12)
        assert used == attrs.evolve(params, nfit3=12)

    def test_kwargs_only(self, monkeypatch):
        used = self._params_used(monkeypatch, tcold=300.0)
        # tcab defaults to tcold, as when constructing the params directly.
        assert used == am.EdgesScriptParams(tcold=300.0)
        assert used.tcab == 300.0

    def test_default(self, monkeypatch):
        assert self._params_used(monkeypatch) == am.EdgesScriptParams()


class TestCorrcsv:
    def test_zero_cable_length(self):
        freqs = np.linspace(50, 100, 101) * un.MHz
        s11 = ReflectionCoefficient(
            reflection_coefficient=np.zeros(len(freqs), dtype=complex), freqs=freqs
        )

        corr = am.corrcsv(s11, cablen=0, cabdiel=0, cabloss=0)
        np.testing.assert_allclose(corr.reflection_coefficient, 0, atol=1e-15)


class TestGetLoadS11s:
    """Regression tests: polynomial load-S11 models used to crash."""

    def _inputs(self):
        s11freq = np.linspace(50, 190, 141) * un.MHz
        f = s11freq.to_value("MHz")
        s11 = (0.2 + 1e-3 * f) * np.exp(1j * 0.01 * f)
        mask = np.ones(f.size, dtype=bool)
        return s11, mask, s11freq

    @pytest.mark.parametrize(
        ("nfit2", "model_type"), [(8, mdl.Polynomial), (27, mdl.Fourier)]
    )
    def test_model_type(self, nfit2: int, model_type: type):
        s11, mask, s11freq = self._inputs()
        params = am.EdgesScriptParams(nfit2=nfit2)
        raw, model_params, models = am.alanmode._get_load_s11s(
            params, s11, s11, s11, s11, mask, s11freq, s11freq
        )
        assert isinstance(model_params.model, model_type)
        assert model_params.model.n_terms == nfit2
        for name, model in models.items():
            np.testing.assert_allclose(
                model.reflection_coefficient,
                raw[name].reflection_coefficient,
                atol=1e-4,
                rtol=0,
            )


def test_edges3_calobs_params_datadir_from_config(tmp_path: Path):
    with config.use(edges3_data=tmp_path):
        params = am.Edges3CalobsParams(specyear=2023, specday=70, s11date="")
    assert params.datadir == tmp_path

    params = am.Edges3CalobsParams(
        specyear=2023, specday=70, s11date="", datadir=str(tmp_path)
    )
    assert params.datadir == tmp_path


class TestReadSpeFile:
    """Regression tests: read_spe_file couldn't read the repo's spe0.txt."""

    @pytest.mark.parametrize("day", ["2022-316", "2023-210"])
    def test_read_edges3_spe0(self, alanmode_data_path: Path, day: str):
        fname = alanmode_data_path / f"edges3-{day}-alan" / "spe0.txt"
        raw = np.genfromtxt(fname, usecols=(1, 3, 6, 9))

        spec = am.read_spe_file(fname)
        np.testing.assert_allclose(spec.freqs.to_value("MHz"), raw[:, 0], rtol=1e-12)
        np.testing.assert_allclose(spec.data[0, 0, 0], raw[:, 1], rtol=1e-12)
        np.testing.assert_allclose(spec.nsamples[0, 0, 0], raw[:, 3], rtol=1e-12)
        assert isinstance(spec.flags["flags"], GSFlag)
        np.testing.assert_array_equal(spec.complete_flags[0, 0, 0], raw[:, 3] == 0)
        assert spec.residuals is None

    def test_read_with_resid_and_flags(self, tmp_path: Path):
        lines = [
            f"freq {f:10.6f} tantenna {t:11.6f} K skymodel {m:11.6f} K "
            f"resid {t - m:9.6f} K wt {w}\n"
            for f, t, m, w in [
                (50.0, 300.0, 299.5, 1),
                (50.5, 301.0, 300.0, 0),
                (51.0, 302.0, 302.5, 1),
            ]
        ]
        (tmp_path / "spe.txt").write_text("".join(lines))

        spec = am.read_spe_file(tmp_path / "spe.txt")
        np.testing.assert_allclose(spec.data[0, 0, 0], [300.0, 301.0, 302.0])
        np.testing.assert_allclose(spec.residuals[0, 0, 0], [0.5, 1.0, -0.5])
        np.testing.assert_array_equal(spec.complete_flags[0, 0, 0], [0, 1, 0])

    def test_default_time_is_not_import_time(self, alanmode_data_path: Path):
        fname = alanmode_data_path / "edges3-2022-316-alan" / "spe0.txt"
        before = Time.now()
        spec = am.read_spe_file(fname)
        assert spec.times[0, 0] >= before
