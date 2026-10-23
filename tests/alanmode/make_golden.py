"""Generate golden regression snapshots of the alanmode calibration pipeline.

The tests in ``test_b18_cal.py``, ``test_edges3_alan.py`` and
``test_edges3_alan_2022_316_cli.py`` compare alanmode to the outputs of Alan's C code,
with tolerances between 1e-10 and 2e-3, so they cannot detect a last-bit change of the
results. The snapshots written by this script hold the *current* outputs of the
pipeline exactly, and ``test_golden.py`` checks that the pipeline still reproduces them
bit for bit.

Each snapshot is one compressed ``.npz`` file per case (see :data:`CASES` and
:data:`SNAPSHOT_OF`; the two 2022_316 cases share one snapshot). It holds the arrays
returned by :func:`edges.alanmode.edges` / :func:`edges.alanmode.alancal` and the
calobs and calibrator they produce: the raw S11s as read from the input files, the
frequencies, the raw and modelled S11s of each load and the receiver (and the delays
and coefficients of the S11 models), the hot-load loss (and the modelled S-parameters
of the hot-load cable for EDGES-2), the averaged spectra Q, the load temperatures, the
noise-wave parameters (Tsca, Toff, Tunc, Tcos, Tsin) and the calibrated temperature of
each load. No array is subsampled (the total size is ~1.7 MB).

Only steps that run on the data in the repository are covered. The averaging of the
raw spectra (``acqplot7amoon``) needs data that is only on the cluster (``/data5``, and
the B18 ``.acq`` files in the repository are empty placeholders), so every case starts
from averaged-spectrum files that are in the repository.

The results are bit-reproducible on one machine, but their last bits depend on the
platform (see :func:`platform_fingerprint`), so the snapshots are not committed.
Instead, they are written from the code *before* a change and compared with the code
*after* it, on the same machine::

    git switch <base>  # e.g. main
    uv run python tests/alanmode/make_golden.py --outdir /tmp/alanmode-golden
    git switch <branch>
    export EDGES_ALANMODE_GOLDEN_DIR=/tmp/alanmode-golden
    uv run pytest tests/alanmode/test_golden.py

The ``alanmode-golden`` CI workflow does this for every pull request, between the
base and the head of the pull request. ``test_golden.py`` is skipped when
``EDGES_ALANMODE_GOLDEN_DIR`` is not set, and when the snapshots were written on a
platform that differs from the one running the tests (e.g. because a pull request
changes the numpy or scipy versions).
"""

import argparse
import json
import platform
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
from astropy import units as un
from numpy.typing import ArrayLike
from pygsdata import GSData

from edges import alanmode as am
from edges.cal.calibrator import Calibrator
from edges.cal.calobs import CalibrationObservation
from edges.cal.loss import LossFunctionGivenSparams
from edges.cal.sparams import ReflectionCoefficient, S11ModelParams
from edges.cal.sparams.core.s11model import get_s11_model

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "alanmode"

# The key under which each snapshot stores the platform_fingerprint() (as JSON).
FINGERPRINT_KEY = "__platform_fingerprint__"

B18_DATA = DATA_PATH / "b18_cal_test"

# The parameters of the B18 test (test_b18_cal.py), i.e. the H2 case of Bowman+2018.
B18_DEFPARAMS = am.Edges2CalobsParams(
    s11_path=B18_DATA
    / "s11_calibration_low_band_LNA25degC_2015-09-16-12-30-29_simulator2_long.txt",
    # The ACQ files are not used, since the averaged spectra are in the repository.
    ambient_acqs=[],
    hotload_acqs=[],
    open_acqs=[],
    short_acqs=[],
)
B18_ACQPARAMS = am.ACQPlot7aMoonParams(
    fstart=40.0,
    fstop=110.0,
    smooth=8,
    tload=300.0,
    tstart=0,
    tstop=23,
    delaystart=7200,
)
B18_CALPARAMS = am.EdgesScriptParams(
    wfstart=50.0,
    wfstop=100.0,
    Lh=-2,
    tcold=296,
    thot=399,
    cfit=6,
    wfit=5,
    nfit2=27,
    nfit3=11,
    lna_poly=0,
)

# The parameters of acqplot7amoon of the EDGES-3 tests (test_edges3_alan.py and
# test_edges3_alan_2022_316_cli.py). Only tload and tcal matter here, since the
# spectra are not averaged.
EDGES3_ACQPARAMS = am.ACQPlot7aMoonParams(
    fstart=48.0,
    fstop=198.0,
    smooth=8,
    tload=300,
    tcal=1000,
    tstart=0,
    tstop=23,
    delaystart=0,
    peakpwr=10,
    minpwr=1,
    pkpwrm=40,
    maxrmsf=400,
    maxfm=200,
)

EDGES3_DEFPARAMS = {
    name: am.Edges3CalobsParams(
        specyear=int(name.split("_")[0]),
        specday=int(name.split("_")[1]),
        s11date=s11date,
        datadir=DATA_PATH / f"edges3-{name.replace('_', '-')}-raw",
        match_resistance=49.8,
        calkit_delays=33,
        lna_cable_length=4.26,
        lna_cable_loss=-91.5,
        lna_cable_dielectric=-1.24,
    )
    for name, s11date in [("2022_316", "2022_319_14"), ("2023_210", "2023_210_03")]
}

EDGES3_CALPARAMS = {
    "2022_316": am.EdgesScriptParams(
        fstart=50.0,
        fstop=190.0,
        wfstart=50.0,
        wfstop=190.0,
        Lh=-1,
        tcold=306.5,
        thot=393.22,
        tcab=306.5,
        cfit=7,
        wfit=7,
        nfit2=27,
        nfit3=10,
    ),
    "2023_210": am.EdgesScriptParams(
        fstart=48.0,
        fstop=198.0,
        wfstart=50.0,
        wfstop=190.0,
        Lh=-1,
        thot=395.19,
        tcold=300.87,
        tcab=298.96,
        cfit=7,
        wfit=7,
        nfit2=27,
        nfit3=10,
    ),
}

EdgesResult = tuple[
    CalibrationObservation, Calibrator, S11ModelParams, S11ModelParams, Callable | None
]


def run_b18(workdir: Path) -> tuple[EdgesResult, tuple[un.Quantity, dict]]:
    """Run the B18 (EDGES-2) calibration with :func:`edges.alanmode.alancal`.

    This is the calibration of ``test_b18_cal.py``, starting from the averaged spectra
    in the repository (which are copied into ``workdir``).

    Parameters
    ----------
    workdir
        An empty directory into which the averaged spectra are copied.

    Returns
    -------
    result
        The output of :func:`edges.alanmode.alancal`.
    raw_s11s
        The frequencies and raw S11s read from the input files, as returned by
        ``get_raw_s11s`` of the data parameters.
    """
    for fl in sorted(B18_DATA.glob("sp*.txt")):
        shutil.copy(fl, workdir / fl.name)

    result = am.alancal(
        defparams=B18_DEFPARAMS,
        out=workdir,
        redo_spectra=False,
        redo_cal=True,
        acqparams=B18_ACQPARAMS,
        calparams=B18_CALPARAMS,
    )
    return result, B18_DEFPARAMS.get_raw_s11s()


def run_edges3_alancal(
    name: str, workdir: Path
) -> tuple[EdgesResult, tuple[un.Quantity, dict]]:
    """Run an EDGES-3 calibration with :func:`edges.alanmode.alancal`.

    This is the calibration that the ``cal-edges3`` CLI command runs in
    ``test_edges3_alan_2022_316_cli.py``, starting from the averaged spectra in the
    ``edges3-<year>-<day>-raw`` directory (which are copied into ``workdir``).

    Parameters
    ----------
    name
        The dataset, as ``<year>_<day>``. Only ``2022_316`` has averaged spectra in its
        ``-raw`` directory.
    workdir
        An empty directory into which the averaged spectra are copied.

    Returns
    -------
    result
        The output of :func:`edges.alanmode.alancal`.
    raw_s11s
        The frequencies and raw S11s read from the input files, as returned by
        ``get_raw_s11s`` of the data parameters.
    """
    defparams = EDGES3_DEFPARAMS[name]
    for fl in sorted(defparams.datadir.glob("sp*.txt")):
        shutil.copy(fl, workdir / fl.name)

    result = am.alancal(
        defparams=defparams,
        out=workdir,
        redo_spectra=False,
        redo_cal=True,
        acqparams=EDGES3_ACQPARAMS,
        calparams=EDGES3_CALPARAMS[name],
    )
    return result, defparams.get_raw_s11s()


def run_edges3_legacy_spectra(
    name: str, workdir: Path | None = None
) -> tuple[EdgesResult, tuple[un.Quantity, dict]]:
    """Run an EDGES-3 calibration with :func:`edges.alanmode.edges`.

    This is the calibration of ``test_edges3_alan.py``, using the averaged spectra
    written by the C code (in the ``edges3-<year>-<day>-alan`` directory) and the raw
    S11s computed from the files in the ``edges3-<year>-<day>-raw`` directory.

    Parameters
    ----------
    name
        The dataset, as ``<year>_<day>``.
    workdir
        Not used, but accepted for a uniform interface with the other runners.

    Returns
    -------
    result
        The output of :func:`edges.alanmode.edges`.
    raw_s11s
        The frequencies and raw S11s read from the input files, as returned by
        ``get_raw_s11s`` of the data parameters.
    """
    legacy_path = DATA_PATH / f"edges3-{name.replace('_', '-')}-alan"
    spectra: dict[str, GSData] = {
        load: am.read_spec_txt(
            legacy_path / f"sp{legacy_key}.txt", telescope="edges3", name=load
        )
        for legacy_key, load in am.LOADMAP.items()
    }
    s11freq, raws11s = EDGES3_DEFPARAMS[name].get_raw_s11s()

    result = am.edges(
        spcold=spectra["ambient"],
        sphot=spectra["hot_load"],
        spopen=spectra["open"],
        spshort=spectra["short"],
        s11freq=s11freq,
        s11cold=raws11s["ambient"],
        s11hot=raws11s["hot_load"],
        s11open=raws11s["open"],
        s11short=raws11s["short"],
        s11lna=raws11s["receiver_s11"],
        params=EDGES3_CALPARAMS[name],
        tcal=EDGES3_ACQPARAMS.tcal * un.K,
        tload=EDGES3_ACQPARAMS.tload * un.K,
    )
    return result, (s11freq, raws11s)


# The cases that are tested, mapping the name of each case to the function that runs
# the pipeline (which takes a scratch directory).
CASES: dict[str, Callable[[Path], tuple[EdgesResult, tuple[un.Quantity, dict]]]] = {
    "b18": run_b18,
    "edges3_2022_316": lambda workdir: run_edges3_legacy_spectra("2022_316", workdir),
    "edges3_2022_316_alancal": lambda workdir: run_edges3_alancal("2022_316", workdir),
    "edges3_2023_210": lambda workdir: run_edges3_legacy_spectra("2023_210", workdir),
}

# The snapshot that each case is compared to. The averaged spectra in the -raw and
# -alan directories of 2022_316 differ only in their formatting, so that both 2022_316
# cases give identical results and share one snapshot (which main() checks).
SNAPSHOT_OF: dict[str, str] = {
    "b18": "b18",
    "edges3_2022_316": "edges3_2022_316",
    "edges3_2022_316_alancal": "edges3_2022_316",
    "edges3_2023_210": "edges3_2023_210",
}


def _array(x: ArrayLike) -> np.ndarray:
    """Convert to an array without changing the values; complex must be complex128."""
    x = np.asarray(x)
    if np.iscomplexobj(x) and x.dtype != np.complex128:
        raise TypeError(f"Expected complex128, got {x.dtype}")
    return x


def _s11_model_arrays(
    prefix: str, params: S11ModelParams, raw: ReflectionCoefficient
) -> dict[str, np.ndarray]:
    """The delay and the coefficients of the S11 model fitted to a raw S11."""
    model = get_s11_model(params, raw)
    return {
        f"{prefix}__s11_model_delay": _array(model.delay.to_value("s")),
        f"{prefix}__s11_model_params": _array([
            model.cmodel.real.model.parameters,
            model.cmodel.imag.model.parameters,
        ]),
    }


def capture(
    result: EdgesResult, raw_s11s: tuple[un.Quantity, dict]
) -> dict[str, np.ndarray]:
    """Collect the intermediate and final arrays of a calibration.

    Parameters
    ----------
    result
        The output of :func:`edges.alanmode.edges` or :func:`edges.alanmode.alancal`.
    raw_s11s
        The frequencies and raw S11s read from the input files, as returned by
        ``get_raw_s11s`` of the data parameters.

    Returns
    -------
    arrays
        The arrays to snapshot, by name. Names of quantities belonging to a load (or
        the receiver) are prefixed by ``<load>__``.
    """
    calobs, calibrator, s11_model_params, receiver_model_params, hot_loss_model = result
    s11freq, raws11s = raw_s11s

    out = {"input_s11_freq": _array(s11freq.to_value("MHz"))}
    out |= {f"input_s11__{key}": _array(val) for key, val in sorted(raws11s.items())}

    out["freqs"] = _array(calobs.freqs.to_value("MHz"))

    for name, load in calobs.loads.items():
        out |= {
            f"{name}__q": _array(load.averaged_q),
            f"{name}__q_nsamples": _array(load.spectrum.q.nsamples),
            f"{name}__q_flags": _array(load.spectrum.q.complete_flags),
            f"{name}__q_freqs": _array(load.freqs.to_value("MHz")),
            f"{name}__spectrum_temp_ave": _array(load.spectrum.temp_ave.to_value("K")),
            f"{name}__temp_ave": _array(un.Quantity(load.temp_ave, "K").value),
            f"{name}__ambient_temperature": _array(
                load.ambient_temperature.to_value("K")
            ),
            f"{name}__loss": _array(load.loss),
            f"{name}__raw_s11_freqs": _array(load._raw_s11.freqs.to_value("MHz")),
            f"{name}__raw_s11": _array(load._raw_s11.s11),
            f"{name}__s11_freqs": _array(
                load.reflection_coefficient.freqs.to_value("MHz")
            ),
            f"{name}__s11": _array(load.reflection_coefficient.s11),
            f"{name}__calibrated_temp": _array(
                calibrator.calibrate_load(load).to_value("K")
            ),
        }
        out |= _s11_model_arrays(name, s11_model_params, load._raw_s11)

    out |= {
        "receiver__raw_s11_freqs": _array(calobs._raw_receiver.freqs.to_value("MHz")),
        "receiver__raw_s11": _array(calobs._raw_receiver.s11),
        "receiver__s11_freqs": _array(calobs.receiver.freqs.to_value("MHz")),
        "receiver__s11": _array(calobs.receiver.s11),
    }
    out |= _s11_model_arrays("receiver", receiver_model_params, calobs._raw_receiver)

    if isinstance(hot_loss_model, LossFunctionGivenSparams):
        sparams = hot_loss_model.sparams
        out |= {
            f"hot_load_cable__{key}": _array(getattr(sparams, key))
            for key in ("s11", "s12", "s21", "s22")
        }
        out["hot_load_cable__freqs"] = _array(sparams.freqs.to_value("MHz"))

    out |= {
        "calibrator__freqs": _array(calibrator.freqs.to_value("MHz")),
        "calibrator__receiver_s11": _array(calibrator.receiver_s11),
    }
    out |= {
        f"calibrator__{key}": _array(getattr(calibrator, key))
        for key in ("Tsca", "Toff", "Tunc", "Tcos", "Tsin")
    }

    return out


def compute_case(name: str, workdir: Path) -> dict[str, np.ndarray]:
    """Run the pipeline of one case and collect its arrays.

    Parameters
    ----------
    name
        The name of the case (a key of :data:`CASES`).
    workdir
        An empty scratch directory for the files written by the pipeline.

    Returns
    -------
    arrays
        The arrays to snapshot, by name (see :func:`capture`).
    """
    result, raw_s11s = CASES[name](Path(workdir))
    return capture(result, raw_s11s)


def golden_file(case: str, golden_path: Path) -> Path:
    """Return the path of the snapshot that a case is compared to.

    Parameters
    ----------
    case
        The name of the case (a key of :data:`CASES`).
    golden_path
        The directory holding the snapshots.

    Returns
    -------
    path
        The path of the ``.npz`` file.
    """
    return golden_path / f"{SNAPSHOT_OF[case]}.npz"


def platform_fingerprint() -> dict[str, str | list[str]]:
    """Describe what the last bits of the results depend on, apart from the code.

    The results are reproducible bit for bit on one machine (independent of the number
    of threads), but their last bits depend on the numpy and scipy versions, the SIMD
    kernels that numpy dispatches to on the CPU (e.g. AVX-512 versions of exp, sin and
    cos), and the BLAS/LAPACK kernels that OpenBLAS selects for the CPU. For example,
    forcing another OpenBLAS kernel (``OPENBLAS_CORETYPE``) or disabling numpy's
    AVX-512 kernels (``NPY_DISABLE_CPU_FEATURES``) changes the fitted S11 model
    coefficients by up to ~1e-5 and the calibrated temperatures by up to ~1e-6 K.

    Returns
    -------
    fingerprint
        The machine architecture, numpy and scipy versions, the CPU features that
        numpy uses, and the BLAS/LAPACK libraries that are loaded (with the kernel
        architecture they use).
    """
    import scipy
    import scipy.linalg  # ruff: ignore[unused-import] -- load scipy's BLAS, to list it below
    from threadpoolctl import threadpool_info

    try:
        from numpy._core._multiarray_umath import __cpu_features__

        cpu_features = sorted(key for key, val in __cpu_features__.items() if val)
    except ImportError:  # pragma: no cover
        cpu_features = ["unknown"]

    blas = sorted(
        f"{lib['internal_api']} {lib['version']} {lib.get('architecture', '')}"
        for lib in threadpool_info()
        if lib["user_api"] == "blas"
    )
    return {
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numpy_cpu_features": cpu_features,
        "blas": blas,
    }


def load_golden(
    case: str, golden_path: Path
) -> tuple[dict[str, np.ndarray], dict[str, str | list[str]]]:
    """Read the snapshot that a case is compared to.

    Parameters
    ----------
    case
        The name of the case (a key of :data:`CASES`).
    golden_path
        The directory holding the snapshots.

    Returns
    -------
    arrays
        The arrays of the snapshot, by name.
    fingerprint
        The :func:`platform_fingerprint` of the machine that wrote the snapshot.
    """
    with np.load(golden_file(case, golden_path), allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    fingerprint = json.loads(str(arrays.pop(FINGERPRINT_KEY)))
    return arrays, fingerprint


def _identical(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> bool:
    """Whether two sets of arrays are exactly the same."""
    return a.keys() == b.keys() and all(
        a[key].dtype == b[key].dtype
        and a[key].shape == b[key].shape
        and np.array_equal(a[key], b[key], equal_nan=True)
        for key in a
    )


def main(argv: list[str] | None = None) -> None:
    """Run all cases and write their snapshots.

    Parameters
    ----------
    argv
        The command-line arguments (by default, ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="directory to write the snapshots into",
    )
    args = parser.parse_args(argv)

    snapshots: dict[str, dict[str, np.ndarray]] = {}
    for case in CASES:
        with tempfile.TemporaryDirectory() as workdir:
            arrays = compute_case(case, Path(workdir))

        name = SNAPSHOT_OF[case]
        if name not in snapshots:
            snapshots[name] = arrays
        elif not _identical(arrays, snapshots[name]):
            raise RuntimeError(
                f"Case {case} no longer gives the same results as the other cases of "
                f"snapshot {name}. Give it its own snapshot in SNAPSHOT_OF."
            )

    fingerprint = np.array(json.dumps(platform_fingerprint(), sort_keys=True))
    args.outdir.mkdir(parents=True, exist_ok=True)
    for name, arrays in snapshots.items():
        path = args.outdir / f"{name}.npz"
        np.savez_compressed(path, **arrays, **{FINGERPRINT_KEY: fingerprint})
        print(f"Wrote {len(arrays)} arrays to {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
