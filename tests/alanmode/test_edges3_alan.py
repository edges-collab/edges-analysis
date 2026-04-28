"""
Integration tests that ensure direct `edges()` calibration reproduces Alan's outputs.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
from astropy import units as un

from edges import alanmode as am
from edges.alanmode.alanmode import _average_spectra


@pytest.fixture(scope="module")
def alandata(alanmode_data_path: Path) -> Path:
    return alanmode_data_path / "edges3-2022-316-alan"


@pytest.fixture(scope="module")
def edges3_2022_316_edges(tmp_path_factory, alanmode_data_path):
    out = tmp_path_factory.mktemp("day316_edges")

    datadir = Path("/data5/edges/data/EDGES3_data/MRO")
    if not datadir.exists():
        datadir = alanmode_data_path / "edges3-data-for-alan-comparison"

    defparams = am.Edges3CalobsParams(
        specyear=2022,
        specday=316,
        s11date="2022_319_14",
        datadir=datadir,
        match_resistance=49.8,
        calkit_delays=33,
        lna_cable_length=4.26,
        lna_cable_loss=-91.5,
        lna_cable_dielectric=-1.24,
    )
    acqparams = am.ACQPlot7aMoonParams(
        fstart=48,
        fstop=198,
        smooth=8,
        tload=300,
        tcal=1000,
    )
    calparams = am.EdgesScriptParams(
        Lh=-1,
        wfstart=50.0,
        wfstop=190.0,
        tcold=306.5,
        thot=393.22,
        tcab=306.5,
        cfit=7,
        wfit=7,
        nfit3=10,
        nfit2=27,
    )

    if "/data5/" not in str(datadir):
        for fl in sorted(datadir.glob("sp*.txt")):
            shutil.copy(fl, out / fl.name)
    else:
        _average_spectra(
            specfiles=defparams.get_spectrum_files(),
            out=out,
            redo_spectra=False,
            fstart=acqparams.fstart,
            fstop=acqparams.fstop,
            smooth=acqparams.smooth,
            tload=acqparams.tload,
            tcal=acqparams.tcal,
            tstart=acqparams.tstart,
            tstop=acqparams.tstop,
            delaystart=acqparams.delaystart,
            telescope="edges3",
        )

    spcold = am.read_spec_txt(out / "spambient.txt", telescope="edges3", name="ambient")
    sphot = am.read_spec_txt(out / "sphot_load.txt", telescope="edges3", name="hot_load")
    spopen = am.read_spec_txt(out / "spopen.txt", telescope="edges3", name="open")
    spshort = am.read_spec_txt(out / "spshort.txt", telescope="edges3", name="short")

    s11freq, raws11s = defparams.get_raw_s11s()
    lna = raws11s.pop("receiver_s11")

    calobs, calibrator, _, _, hot_loss_model = am.edges(
        spcold=spcold,
        sphot=sphot,
        spopen=spopen,
        spshort=spshort,
        s11freq=s11freq,
        s11cold=raws11s["ambient"],
        s11hot=raws11s["hot_load"],
        s11open=raws11s["open"],
        s11short=raws11s["short"],
        s11lna=lna,
        s11rig=raws11s.get("s11rig"),
        s12rig=raws11s.get("s12rig"),
        s22rig=raws11s.get("s22rig"),
        params=calparams,
        tcal=acqparams.tcal * un.K,
        tload=acqparams.tload * un.K,
    )

    for name, load in calobs.loads.items():
        am.write_s11_csv(load._raw_s11.freqs, load._raw_s11.s11, out / f"s11{name}.csv")
    am.write_s11_csv(calobs._raw_receiver.freqs, calobs._raw_receiver.s11, out / "s11lna.csv")
    am.write_modelled_s11s(calobs, out / "s11_modelled.txt", hot_loss_model)
    am.write_specal(calibrator, out / "specal.txt", t_load=acqparams.tload, t_load_ns=acqparams.tcal)

    return out


@pytest.mark.parametrize("load", am.LOADMAP.keys())
def test_spectra(edges3_2022_316_edges: Path, load, alandata: Path):
    alan = am.read_spec_txt(alandata / f"sp{load}.txt")
    ours = am.read_spec_txt(edges3_2022_316_edges / f"sp{am.LOADMAP[load]}.txt")

    np.testing.assert_allclose(alan.freqs, ours.freqs)
    np.testing.assert_allclose(alan.data[20:-20], ours.data[20:-20], atol=1e-6)


@pytest.mark.parametrize("load", [*list(am.LOADMAP.keys()), "lna"])
def test_unmodelled_s11(edges3_2022_316_edges: Path, load, alandata: Path):
    s11freq, alans11 = am.read_s11_csv(alandata / f"s11{load}.csv")
    key = "lna" if load == "lna" else am.LOADMAP[load]
    ourfreq, ours11 = am.read_s11_csv(edges3_2022_316_edges / f"s11{key}.csv")

    mask = (s11freq >= ourfreq.min()) & (s11freq <= ourfreq.max())

    np.testing.assert_allclose(s11freq[mask], ourfreq)
    np.testing.assert_allclose(alans11[mask].real, ours11.real, atol=1e-10)
    np.testing.assert_allclose(alans11[mask].imag, ours11.imag, atol=1e-10)


def test_modelled_s11(edges3_2022_316_edges: Path, alandata: Path):
    alans11m = np.genfromtxt(alandata / "s11_modelled.txt", comments="#", names=True)
    ours11m = np.genfromtxt(
        edges3_2022_316_edges / "s11_modelled.txt", comments="#", names=True
    )

    for key in alans11m.dtype.names:
        np.testing.assert_allclose(alans11m[key], ours11m[key], atol=5e-9, rtol=0)


def test_specal(edges3_2022_316_edges: Path, alandata: Path):
    alan = am.read_specal(alandata / "specal.txt", t_load=300, t_load_ns=1000)
    ours = am.read_specal(edges3_2022_316_edges / "specal.txt", t_load=300, t_load_ns=1000)

    np.testing.assert_allclose(ours.Tsca, alan.Tsca, rtol=3e-6)
    np.testing.assert_allclose(ours.Toff, alan.Toff, atol=4e-5)
    np.testing.assert_allclose(ours.Tunc, alan.Tunc, atol=1.1e-6)
    np.testing.assert_allclose(ours.Tcos, alan.Tcos, atol=2e-5)
    np.testing.assert_allclose(ours.Tsin, alan.Tsin, atol=1.4e-5)
