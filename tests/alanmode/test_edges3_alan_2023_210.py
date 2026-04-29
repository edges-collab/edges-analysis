"""
Integration test for day 2023/210 comparing C-code outputs to direct `edges()`.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from astropy import units as un

from edges import alanmode as am
from edges.alanmode.alanmode import _average_spectra


CTERMS = 7
WTERMS = 7
YEAR = 2023
DAY = 210
S11DATE = "2023_210_03"

FSTART = 48.0
FSTOP = 198.0
WFSTART = 50.0
WFSTOP = 190.0

TCOLD = 300.87
THOT = 395.19
TCAB = 298.96

NFIT2 = 27
NFIT3 = 10

@pytest.fixture(scope="module")
def edges3_2023_210(tmp_path_factory):
    repo_root = Path(__file__).resolve().parents[4]
    c_pipeline_dir = repo_root / "packages" / "alans-pipeline"
    alan_data_root = Path(__file__).resolve().parents[1] / "data" / "alanmode"
    alan_out = alan_data_root / "edges3-2023-210-alan"
    # docal_edges3 currently points BINDIR at src/. Keep bin/ as fallback.
    c_bindir = c_pipeline_dir / "src"
    if not c_bindir.exists():
        c_bindir = c_pipeline_dir / "bin"
    c_reads1p1 = c_bindir / "reads1p1"
    c_corrcsv = c_bindir / "corrcsv"
    c_acqplot = c_bindir / "acqplot7amoon"
    c_edges3 = c_bindir / "edges3_fittpfix_pifix"

    datadir = Path("/data5/edges/data/EDGES3_data/MRO")
    if not datadir.exists():
        pytest.skip("Requires enterprise EDGES3 data at /data5/edges/data/EDGES3_data/MRO")
    if not c_pipeline_dir.exists():
        pytest.skip("Requires packages/alans-pipeline to be available")
    if not c_edges3.exists():
        pytest.skip("Required binary not found: edges3_fittpfix_pifix")
    if not all(p.exists() for p in (c_reads1p1, c_corrcsv, c_acqplot)):
        pytest.skip(
            "reads1p1/corrcsv/acqplot7amoon not available in packages/alans-pipeline"
        )

    defparams = am.Edges3CalobsParams(
        specyear=YEAR,
        specday=DAY,
        s11date=S11DATE,
        datadir=datadir,
        match_resistance=49.8,
        calkit_delays=33,
        lna_cable_length=4.26,
        lna_cable_loss=-91.5,
        lna_cable_dielectric=-1.24,
    )
    calparams = am.EdgesScriptParams(
        wfstart=WFSTART,
        wfstop=WFSTOP,
        Lh=-1,
        thot=THOT,
        tcold=TCOLD,
        tcab=TCAB,
        cfit=CTERMS,
        wfit=WTERMS,
        nfit2=NFIT2,
        nfit3=NFIT3,
    )
    acqparams = am.ACQPlot7aMoonParams(
        fstart=FSTART,
        fstop=FSTOP,
        smooth=8,
        tload=300,
        tcal=1000,
        tstart=0,
        tstop=23,
        delaystart=0,
    )

    # C path: run fully from scratch in a fresh workdir.
    workingdir = tmp_path_factory.mktemp("day210_c_workdir")
    d = datadir / S11DATE
    for s in ("ant", "amb", "hot", "open", "short"):
        subprocess.run(
            [
                str(c_reads1p1),
                "-res",
                "49.8",
                "-Tfopen",
                str(d.with_name(f"{d.name}_O.s1p")),
                "-Tfshort",
                str(d.with_name(f"{d.name}_S.s1p")),
                "-Tfload",
                str(d.with_name(f"{d.name}_L.s1p")),
                "-Tfant",
                str(d.with_name(f"{d.name}_{s}.s1p")),
                "-loadps",
                "33",
                "-openps",
                "33",
                "-shortps",
                "33",
            ],
            cwd=workingdir,
            check=True,
        )
        shutil.move(workingdir / "s11.csv", workingdir / f"s11{s}.csv")

    subprocess.run(
        [
            str(c_reads1p1),
            "-res",
            "49.8",
            "-Tfopen",
            str(d.with_name(f"{d.name}_lna_O.s1p")),
            "-Tfshort",
            str(d.with_name(f"{d.name}_lna_S.s1p")),
            "-Tfload",
            str(d.with_name(f"{d.name}_lna_L.s1p")),
            "-Tfant",
            str(d.with_name(f"{d.name}_lna.s1p")),
            "-loadps",
            "33",
            "-openps",
            "33",
            "-shortps",
            "33",
        ],
        cwd=workingdir,
        check=True,
    )
    subprocess.run(
        [
            str(c_corrcsv),
            "s11.csv",
            "-cablen",
            "4.26",
            "-cabdiel",
            "-1.24",
            "-cabloss",
            "-91.5",
        ],
        cwd=workingdir,
        check=True,
    )
    shutil.move(workingdir / "c_s11.csv", workingdir / "s11lna.csv")

    specfiles = defparams.get_spectrum_files()
    for c_load, py_load in [
        ("amb", "ambient"),
        ("hot", "hot_load"),
        ("open", "open"),
        ("short", "short"),
    ]:
        files = specfiles[py_load]
        temp_acq = workingdir / "temp.acq"
        if temp_acq.exists():
            temp_acq.unlink()
        with open(temp_acq, "wb") as outfl:
            for fl in files:
                with open(fl, "rb") as infl:
                    outfl.write(infl.read())

        subprocess.run(
            [
                str(c_acqplot),
                str(temp_acq),
                "-fstart",
                str(int(acqparams.fstart)),
                "-fstop",
                str(int(acqparams.fstop)),
                "-tload",
                str(int(acqparams.tload)),
                "-tcal",
                str(int(acqparams.tcal)),
                "-pfit",
                "27",
                "-smooth",
                str(int(acqparams.smooth)),
                "-rfi",
                "0",
                "-peakpwr",
                "10",
                "-minpwr",
                "1",
                "-pkpwrm",
                "40",
                "-maxrmsf",
                "400",
                "-maxfm",
                "200",
            ],
            cwd=workingdir,
            check=True,
        )
        shutil.move(workingdir / "spe.txt", workingdir / f"sp{c_load}.txt")
        if (workingdir / "flagfile.txt").exists():
            shutil.move(workingdir / "flagfile.txt", workingdir / f"flagfile-{c_load}.txt")

    subprocess.run(
        [
            str(c_edges3),
            "-fstart",
            str(int(FSTART)),
            "-fstop",
            str(int(FSTOP)),
            "-spant",
            "spopen.txt",
            "-spcold",
            "spamb.txt",
            "-sphot",
            "sphot.txt",
            "-spopen",
            "spopen.txt",
            "-spshort",
            "spshort.txt",
            "-s11ant",
            "s11open.csv",
            "-s11hot",
            "s11hot.csv",
            "-s11cold",
            "s11amb.csv",
            "-s11lna",
            "s11lna.csv",
            "-s11open",
            "s11open.csv",
            "-s11short",
            "s11short.csv",
            "-Lh",
            "-1",
            "-mfit",
            "-1",
            "-smooth",
            "-8",
            "-wfstart",
            str(int(WFSTART)),
            "-wfstop",
            str(int(WFSTOP)),
            "-eorcen",
            "0",
            "-eorwid",
            "0",
            "-tcold",
            str(TCOLD),
            "-thot",
            str(THOT),
            "-wtmode",
            "1",
            "-lmode",
            "-1",
            "-tant",
            str(TCOLD),
            "-tcab",
            str(TCAB),
            "-cfit",
            str(CTERMS),
            "-wfit",
            str(WTERMS),
            "-nfit3",
            str(NFIT3),
            "-ldb",
            "0.0",
            "-adb",
            "0.0",
            "-delaylna",
            "0e-12",
            "-nfit2",
            str(NFIT2),
        ],
        cwd=workingdir,
        check=True,
    )

    if alan_out.exists():
        shutil.rmtree(alan_out)
    alan_out.mkdir(parents=True, exist_ok=True)

    for fl in (
        list(workingdir.glob("sp*.txt"))
        + list(workingdir.glob("s11*.csv"))
        + [workingdir / "s11_modelled.txt", workingdir / "specal.txt"]
        + list(workingdir.glob("calibrated_*.txt"))
    ):
        if fl.exists():
            shutil.copy(fl, alan_out / fl.name)

    # Python path: independent from C outputs and run from scratch in fresh dir.
    py_out = tmp_path_factory.mktemp("day210_edges")
    _average_spectra(
        specfiles=defparams.get_spectrum_files(),
        out=py_out,
        redo_spectra=True,
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
    spcold = am.read_spec_txt(py_out / "spambient.txt", telescope="edges3", name="ambient")
    sphot = am.read_spec_txt(py_out / "sphot_load.txt", telescope="edges3", name="hot_load")
    spopen = am.read_spec_txt(py_out / "spopen.txt", telescope="edges3", name="open")
    spshort = am.read_spec_txt(py_out / "spshort.txt", telescope="edges3", name="short")

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
        am.write_s11_csv(load._raw_s11.freqs, load._raw_s11.s11, py_out / f"s11{name}.csv")
    am.write_s11_csv(calobs._raw_receiver.freqs, calobs._raw_receiver.s11, py_out / "s11lna.csv")
    am.write_modelled_s11s(calobs, py_out / "s11_modelled.txt", hot_loss_model)
    am.write_specal(
        calibrator,
        py_out / "specal.txt",
        t_load=acqparams.tload,
        t_load_ns=acqparams.tcal,
    )

    for ec_name, alan_name in {
        "ambient": "ambient",
        "hot_load": "hot",
        "open": "open",
        "short": "short",
    }.items():
        # Use the load object already restricted to the calibration band so
        # uncal/cal/freq arrays are guaranteed to align.
        load = calobs.loads[ec_name]
        freqs = load.freqs.to_value("MHz")
        uncal = load.averaged_q
        cal = calibrator.calibrate_load(load).value
        np.savetxt(
            py_out / f"calibrated_{alan_name}.txt",
            np.array([freqs, uncal, cal]).T,
            header="freq uncal cal",
        )

    return {"alan_out": alan_out, "py_out": py_out}


@pytest.mark.parametrize("load", am.LOADMAP.keys())
def test_spectra(edges3_2023_210, load):
    alan_out = edges3_2023_210["alan_out"]
    py_out = edges3_2023_210["py_out"]

    alan = am.read_spec_txt(alan_out / f"sp{load}.txt")
    ours = am.read_spec_txt(py_out / f"sp{am.LOADMAP[load]}.txt")

    np.testing.assert_allclose(alan.freqs, ours.freqs)
    np.testing.assert_allclose(alan.data[20:-20], ours.data[20:-20], atol=1e-6)


@pytest.mark.parametrize("load", [*list(am.LOADMAP.keys()), "lna"])
def test_unmodelled_s11(edges3_2023_210, load):
    alan_out = edges3_2023_210["alan_out"]
    py_out = edges3_2023_210["py_out"]

    s11freq, alans11 = am.read_s11_csv(alan_out / f"s11{load}.csv")
    key = "lna" if load == "lna" else am.LOADMAP[load]
    ourfreq, ours11 = am.read_s11_csv(py_out / f"s11{key}.csv")

    mask = (s11freq >= ourfreq.min()) & (s11freq <= ourfreq.max())
    np.testing.assert_allclose(s11freq[mask], ourfreq)
    np.testing.assert_allclose(alans11[mask].real, ours11.real, atol=1e-10)
    np.testing.assert_allclose(alans11[mask].imag, ours11.imag, atol=1e-10)


def test_modelled_s11(edges3_2023_210):
    alan_out = edges3_2023_210["alan_out"]
    py_out = edges3_2023_210["py_out"]

    alans11m = np.genfromtxt(alan_out / "s11_modelled.txt", comments="#", names=True)
    ours11m = np.genfromtxt(py_out / "s11_modelled.txt", comments="#", names=True)
    for key in alans11m.dtype.names:
        np.testing.assert_allclose(alans11m[key], ours11m[key], atol=1e-6, rtol=0)


def test_specal(edges3_2023_210):
    alan_out = edges3_2023_210["alan_out"]
    py_out = edges3_2023_210["py_out"]

    acqparams = am.ACQPlot7aMoonParams(fstart=FSTART, fstop=FSTOP, smooth=8, tload=300, tcal=1000)
    alan = am.read_specal(
        alan_out / "specal.txt",
        t_load=acqparams.tload,
        t_load_ns=acqparams.tcal,
    )
    ours = am.read_specal(
        py_out / "specal.txt",
        t_load=acqparams.tload,
        t_load_ns=acqparams.tcal,
    )

    np.testing.assert_allclose(ours.Tsca, alan.Tsca, rtol=3e-6)
    np.testing.assert_allclose(ours.Toff, alan.Toff, atol=4e-5)
    np.testing.assert_allclose(ours.Tunc, alan.Tunc, atol=1.1e-6)
    np.testing.assert_allclose(ours.Tcos, alan.Tcos, atol=2e-5)
    np.testing.assert_allclose(ours.Tsin, alan.Tsin, atol=1.4e-5)


@pytest.mark.parametrize("alan_name", ["ambient", "hot", "open", "short"])
def test_calibrated_temps(edges3_2023_210, alan_name):
    alan_out = edges3_2023_210["alan_out"]
    py_out = edges3_2023_210["py_out"]

    alan = am.read_alan_calibrated_temp(alan_out / f"calibrated_{alan_name}.txt")
    ours = am.read_alan_calibrated_temp(py_out / f"calibrated_{alan_name}.txt")

    np.testing.assert_allclose(ours["freq"], alan["freq"], atol=0, rtol=0)
    np.testing.assert_allclose(ours["uncal"], alan["uncal"], atol=1e-6, rtol=0)
    np.testing.assert_allclose(ours["cal"], alan["cal"], atol=5e-5, rtol=0)
