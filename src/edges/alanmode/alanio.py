"""Functions for writing/reading to/from file formats used in the C pipeline."""
import numpy as np
from pygsdata import GSData, Stamp, History
from ..cal import Calibrator, CalibrationObservation
from astropy import units as un
from edges.cal.loss import HotLoadCorrection
from .. import types as tp
from astropy.time import Time
from ..const import KNOWN_TELESCOPES

def read_raul_s11_format(fname: tp.PathLike) -> dict[str, np.ndarray]:
    """
    Read files containing S11's for all loads, LNA, and rigid cable, for EDGES-2.

    These files are outputs from Raul's pipeline, and were used as inputs for the C-code
    in EDGES-2 in some cases.
    """
    s11 = np.genfromtxt(
        fname,
        names=[
            "freq",
            "lna_rl",
            "lna_im",
            "amb_rl",
            "amb_im",
            "hot_rl",
            "hot_im",
            "open_rl",
            "open_im",
            "short_rl",
            "short_im",
            "s11rig_rl",
            "s11rig_im",
            "s12rig_rl",
            "s12rig_im",
            "s22rig_rl",
            "s22rig_im",
        ],
    )

    out = {
        "freq": s11["freq"]*un.MHz,
    }
    for name in s11.dtype.names:
        if "_" not in name:
            continue

        if "_rl" in name:
            out[name.split("_")[0]] = s11[name] + 0j
        else:
            out[name.split("_")[0]] += 1j * s11[name]
    return out


def read_s11_csv(fname) -> tuple[np.ndarray, np.ndarray]:
    """Read a CSV file containing S11 data in Alan's output format."""
    with open(fname) as fl:
        data = np.genfromtxt(fl, delimiter=",", skip_header=1, skip_footer=1)
        freq = data[:, 0]
        s11 = data[:, 1] + data[:, 2] * 1j
    return freq, s11


def read_spec_txt(
    fname: tp.PathLike, 
    time: Time = Time.now(),
    telescope: str = 'edges-low',
    name: str = ""
) -> GSData:
    """Read an averaged-spectrum file, like the ones output by acqplot7amoon."""
    out = np.genfromtxt(
        fname,
        names=["freq", "spectra", "weight"],
        comments="/",
        usecols=[0, 1, 2],
    )
    with open(fname) as fl:
        n = int(fl.readline()[31:].split(" ")[0])

    return GSData(
        telescope=KNOWN_TELESCOPES.get(telescope),
        data = out['spectra'][None, None, None, :],
        freqs = out['freq']*un.MHz,
        times = [[time]],
        effective_integration_time = 13.0 * un.second,
        nsamples = out['weight'][None, None, None, :] * n,
        flags = {'flags': ~(out['weight'].astype(bool))},
        history = History([Stamp(f'read from {fname}')]),
        data_unit='uncalibrated_temp',
        name=name
    )


def write_spec_txt(freq: np.ndarray | tp.FreqType, n: int | float, spec: np.ndarray, fname: tp.PathLike):
    """Write an averaged-spectrum file, like spe_<load>r.txt files from edges2k.c."""
    if hasattr(freq, "unit"):
        freq = freq.to_value("MHz")

    with open(fname, "w") as fl:
        for i, (f, sp) in enumerate(zip(freq, spec, strict=False)):
            if i == 0:
                fl.write(f"{f:12.6f} {sp:12.6f} {1:4.0f} {n} // temp.acq\n")
            else:
                fl.write(f"{f:12.6f} {sp:12.6f} {1:4.0f}\n")

def write_spec_txt_gsd(gsd: GSData, fname: tp.PathLike):
    write_spec_txt(
        freq=gsd.freqs,
        n = np.mean(gsd.nsamples),
        spec=gsd.data[0,0,0],
        fname=fname
    )

def read_specal(fname: tp.PathLike, t_load: float, t_load_ns: float) -> Calibrator:
    """Read a specal file, like the ones output by edges3(k)."""
    data = np.genfromtxt(
        fname,
        names=[
            "freq",
            "s11lna_real",
            "s11lna_imag",
            "C1",
            "C2",
            "Tunc",
            "Tcos",
            "Tsin",
            "weight",
        ],
        usecols=(1, 3, 4, 6, 8, 10, 12, 14, 16),
    )
    mask = data['weight']>0
    data = data[mask]
    
    return Calibrator(
        freq = data['freq'] * un.MHz,
        C1 = data['C1'],
        C2 = data['C2'],
        Tunc = data['Tunc'],
        Tcos = data['Tcos'],
        Tsin = data['Tsin'],
        receiver_s11 = data['s11lna_real'] + 1j * data['s11lna_imag'],
        t_load=t_load,
        t_load_ns = t_load_ns
    )

def read_specal_iter(fname: tp.PathLike) -> np.ndarray:
    """
    Read a specal file, like the ones output by edges3(k).

    The outputs are from an intermediate iteration of NW modeling step.
    """
    return np.genfromtxt(
        fname,
        names=[
            "iter",
            "freq",
            "C1",
            "C2",
            "Tunc",
            "Tcos",
            "Tsin",
        ],
        usecols=(1, 3, 5, 7, 9, 11, 13),
    )


def read_alan_calibrated_temp(fname: tp.PathLike) -> np.ndarray:
    """
    Read calibrated_{load}.txt from edges3.c.

    This gives uncalibrated and calibrated temperatures.
    """
    return np.genfromtxt(
        fname,
        names=[
            "freq",
            "uncal",
            "cal",
        ],
        delimiter=" ",
        skip_header=1,
    )


def write_specal(calibrator: Calibrator, outfile: tp.PathLike, precision="10.6f"):
    """Write a specal file in the same format as those output by the C-code edges3.c."""
    sca = calibrator.C1
    ofs = calibrator.C2
    tlnau = calibrator.Tunc
    tlnac = calibrator.Tcos
    tlnas = calibrator.Tsin
    lna = calibrator.receiver_s11
    
    with open(outfile, "w") as fl:
        for i in range(calibrator.freq.size):        
            fl.write(
                f"freq {calibrator.freq[i].to_value('MHz'):{precision}} "
                f"s11lna {lna[i].real:{precision}} {lna[i].imag:{precision}} "
                f"sca {sca[i]:{precision}} ofs {ofs[i]:{precision}} "
                f"tlnau {tlnau[i]:{precision}} tlnac {tlnac[i]:{precision}} "
                f"tlnas {tlnas[i]:{precision}} wtcal 1 cal_data\n"
            )


def write_modelled_s11s(calobs: CalibrationObservation, fname: tp.PathLike):
    """Write all modelled S11's in a calobs object to file, in the same format as C.

    If a HotLoadCorrection exists, also write the rigid cable S-parameters, as
    edges2k.c does, otherwise assume the edges3.c format.
    """
    s11m = {name: load.s11_model(calobs.freq) for name, load in calobs.loads.items()}
    lna = calobs.receiver_s11
    if isinstance(calobs.hot_load._loss_model, HotLoadCorrection):
        f = calobs.freq.to_value("MHz")
        s11m |= {
            "rig_s11": calobs.hot_load._loss_model.s11_model(f),
            "rig_s12": calobs.hot_load._loss_model.s12s21_model(f),
            "rig_s22": calobs.hot_load._loss_model.s22_model(f),
        }

    with open(fname, "w") as fl:
        if "rig_s11" in s11m:
            fl.write(
                "# freq, amb_real amb_imag hot_real hot_imag open_real open_imag "
                "short_real short_imag lna_real lna_imag rig_s11_real rig_s11_imag "
                "rig_s12_real rig_s12_imag rig_s22_real rig_s22_imag\n"
            )
            for i, (f, amb, hot, op, sh, rigs11, rigs12, rigs22) in enumerate(
                zip(
                    calobs.freq.to_value("MHz"),
                    s11m["ambient"],
                    s11m["hot_load"],
                    s11m["open"],
                    s11m["short"],
                    s11m["rig_s11"],
                    s11m["rig_s12"],
                    s11m["rig_s22"],
                    strict=False,
                )
            ):
                fl.write(
                    f"{f} {amb.real} {amb.imag} "
                    f"{hot.real} {hot.imag} "
                    f"{op.real} {op.imag} "
                    f"{sh.real} {sh.imag} "
                    f"{lna[i].real} {lna[i].imag} "
                    f"{rigs11.real} {rigs11.imag} "
                    f"{rigs12.real} {rigs12.imag} "
                    f"{rigs22.real} {rigs22.imag}\n"
                )

        else:
            fl.write(
                "# freq, amb_real amb_imag hot_real hot_imag open_real open_imag "
                "short_real short_imag lna_real lna_imag\n"
            )
            for i, (f, amb, hot, op, sh) in enumerate(
                zip(
                    calobs.freq.to_value("MHz"),
                    s11m["ambient"],
                    s11m["hot_load"],
                    s11m["open"],
                    s11m["short"],
                    strict=False,
                )
            ):
                fl.write(
                    f"{f} {amb.real} {amb.imag} "
                    f"{hot.real} {hot.imag} "
                    f"{op.real} {op.imag} "
                    f"{sh.real} {sh.imag} "
                    f"{lna[i].real} {lna[i].imag}\n"
                )


def read_spe_file(
    filename: tp.PathLike,
    time: Time = Time.now(),
    telescope: str = 'edges-low',
    name: str = ""
):
    """Read Alan's spectrum files with formats like those of spe0.txt."""
    out = np.genfromtxt(
        filename,
        usecols=(1, 3, 6, 9, 12),
        names=("freq", "tant", "model", "resid", "weight"),
    )
    
    return GSData(
        telescope=KNOWN_TELESCOPES.get(telescope),
        data = out['tant'][None, None, None, :],
        freqs = out['freq']*un.MHz,
        times = [[time]],
        effective_integration_time = 13.0 * un.second,
        nsamples = out['weight'][None, None, None, :],
        residuals=out['resid'][None, None, None],
        flags = {'flags': ~(out['weight'].astype(bool))},
        history = History([Stamp(f'read from {filename}')]),
        data_unit='temperature',
        name=name
    )