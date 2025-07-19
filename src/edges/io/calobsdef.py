"""A module defining the overall file structure and internal contents of cal obs.

This module defines the overall file structure and internal contents of the
calibration observations. It does *not* implement any algorithms/methods on that data,
making it easier to separate the algorithms from the data checking/reading.
"""

from __future__ import annotations

import tomllib as toml
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import attrs
import yaml
from astropy import units
from bidict import bidict

from .. import DATA_PATH
from .. import types as tp

with (DATA_PATH / "calibration_loads.toml").open("rb") as fl:
    data = toml.load(fl)
    LOAD_ALIASES = bidict({v["alias"]: k for k, v in data.items()})
    LOAD_MAPPINGS = {
        v: k
        for k, val in data.items()
        for v in [*val.get("misspells", []), val["alias"]]
    }

with (DATA_PATH / "antenna_simulators.toml").open("rb") as fl:
    ANTENNA_SIMULATORS = toml.load(fl)

# Dictionary of misspelled:true mappings.
ANTSIM_REVERSE = {
    v: k for k, val in ANTENNA_SIMULATORS.items() for v in val.get("misspells", [])
}


def _list_of_path(x: Sequence[tp.PathLike]) -> list[Path]:
    return [Path(xx) for xx in x]


def _vld_path_exists(inst, att, val):
    if not val.exists():
        raise ValueError(f"{att.name} path does not exist! ({val})")


@attrs.define(frozen=True, kw_only=True)
class Calkit:
    match: Path = attrs.field(converter=Path, validator=_vld_path_exists)
    open: Path = attrs.field(converter=Path, validator=_vld_path_exists)
    short: Path = attrs.field(converter=Path, validator=_vld_path_exists)


class CalkitEdges2(Calkit):
    @classmethod
    def from_standard_layout(
        cls,
        direc: tp.PathLike,
        repeat_num: int = 1,
        allow_other: bool = True,
        prefix: str = "",
    ) -> Self:
        direc = Path(direc)
        _open = direc / f"{prefix}Open{repeat_num:02}.s1p"

        if not _open.exists():
            if allow_other:
                _open = next(direc.glob("{prefix}Open*.s1p"))

                warnings.warn(
                    f"Could not find {prefix}Open{repeat_num:02} in {direc}, using {_open.name}"
                )
            else:
                raise OSError(f"Could not find {prefix}Open{repeat_num:02} in {direc}")

        _rep_num = _open.stem[-2:]

        return cls(
            open=direc / f"{prefix}Open{_rep_num}.s1p",
            short=direc / f"{prefix}Short{_rep_num}.s1p",
            match=direc / f"{prefix}Match{_rep_num}.s1p",
        )


@attrs.define()
class LoadS11:
    calkit: Calkit = attrs.field(validator=attrs.validators.instance_of(Calkit))
    external: Path = attrs.field(converter=Path, validator=_vld_path_exists)


@attrs.define(frozen=True, kw_only=True)
class LoadDefEDGES2:
    name: str = attrs.field()

    thermistor: Path = attrs.field(converter=Path, validator=_vld_path_exists)
    s11: LoadS11 = attrs.field()
    spectra: list[Path] = attrs.field(converter=_list_of_path)
    sparams_file: Path | None = attrs.field(default=None)

    @classmethod
    def from_standard_layout(
        cls,
        root: tp.PathLike,
        loadname: str,
        run_num: int = 1,
        rep_num: int = 1,
        sparams_file: Path | None = None,
    ) -> Self:
        root = Path(root)
        assert root.exists()

        # Check the basic validity of this observation directory
        assert (root / "Resistance").exists()
        assert (root / "S11").exists()
        assert (root / "Spectra").exists()

        # Get Resistance
        res = next((root / "Resistance").glob(f"{loadname}_{run_num:02}_*.csv"))
        spec = list((root / "Spectra").glob(f"{loadname}_{run_num:02}_*.acq"))

        s11dir = root / "S11" / f"{loadname}{run_num:02}"
        clk = CalkitEdges2.from_standard_layout(s11dir, rep_num)
        _repnum = clk.open.stem[-2:]
        s11 = LoadS11(calkit=clk, external=s11dir / f"External{_repnum}.s1p")

        # By default, the hot load uses a semi-rigid cable S-parameter file.
        if loadname == "hot_load" and sparams_file is None:
            sparams_file = DATA_PATH / "semi_rigid_s_parameters_WITH_HEADER.txt"

        return cls(
            thermistor=res,
            s11=s11,
            spectra=spec,
            name=loadname,
            sparams_file=sparams_file,
        )


@attrs.define(frozen=True, kw_only=True)
class SwitchingState:
    internal: Calkit = attrs.field(validator=attrs.validators.instance_of(Calkit))
    external: Calkit = attrs.field(validator=attrs.validators.instance_of(Calkit))


@attrs.define(frozen=True, kw_only=True)
class ReceiverS11:
    calkit: Calkit = attrs.field(validator=attrs.validators.instance_of(Calkit))
    device: Path = attrs.field(converter=Path, validator=_vld_path_exists)


@attrs.define(frozen=True, kw_only=True)
class CalObsDefEDGES2:
    open: LoadDefEDGES2 = attrs.field()
    short: LoadDefEDGES2 = attrs.field()
    ambient: LoadDefEDGES2 = attrs.field()
    hot_load: LoadDefEDGES2 = attrs.field()

    switching_state: SwitchingState = attrs.field()
    receiver_s11: ReceiverS11 = attrs.field()

    run_num: int = attrs.field()
    receiver_female_resistance: units.Quantity[units.ohm] = attrs.field(
        default=50 * units.ohm
    )
    male_resistance: units.Quantity[units.ohm] = attrs.field(default=50 * units.ohm)

    @property
    def loads(self) -> dict[str, LoadDefEDGES2]:
        return {
            "open": self.open,
            "short": self.short,
            "ambient": self.ambient,
            "hot_load": self.hot_load,
        }

    @classmethod
    def from_standard_layout(
        cls,
        rootdir: tp.PathLike,
        run_num: int = 1,
        repeat_num: int = 1,
        receiver_female_resistance: units.Quantity[units.ohm] | None = None,
        male_resistance: units.Quantity[units.ohm] | None = None,
    ) -> Self:
        rootdir = Path(rootdir)
        if not rootdir.exists():
            raise FileNotFoundError(f"rootdir {rootdir} does not exist")

        # Check the basic validity of this observation directory
        assert (rootdir / "Resistance").exists()
        assert (rootdir / "S11").exists()
        assert (rootdir / "Spectra").exists()

        # Get the ReceiverS11
        rcvdir = rootdir / "S11" / f"ReceiverReading{run_num:02}"
        if not rcvdir.exists():
            # Try any run num:
            rcvdir = next((rootdir / "S11").glob("ReceiverReading*"))
            warnings.warn(
                f"Could not find ReceiverReading{run_num:02}, using {rcvdir.name}"
            )

        run_num = int(rcvdir.stem[-2:])
        rcv_calkit = CalkitEdges2.from_standard_layout(rcvdir, repeat_num)
        rcv = ReceiverS11(
            calkit=rcv_calkit,
            device=rcvdir / f"ReceiverReading{rcv_calkit.open.stem[-2:]}.s1p",
        )

        # Get the Switching State
        swstate = rootdir / "S11" / f"SwitchingState{run_num:02}"
        if not swstate.exists():
            # Try any run num:
            swstate = next((rootdir / "S11").glob("SwitchingState*"))
            warnings.warn(
                f"Could not find SwitchingState{run_num:02}, using {swstate.name}"
            )

        sw_calkit = CalkitEdges2.from_standard_layout(swstate, repeat_num)
        repnum = int(sw_calkit.open.stem[-2:])
        swsate = SwitchingState(
            internal=sw_calkit,
            external=CalkitEdges2.from_standard_layout(
                swstate, repeat_num=repnum, allow_other=False, prefix="External"
            ),
        )

        # Now, get the Loads
        _open = LoadDefEDGES2.from_standard_layout(
            rootdir, "LongCableOpen", run_num=run_num, rep_num=repeat_num
        )
        short = LoadDefEDGES2.from_standard_layout(
            rootdir, "LongCableShorted", run_num=run_num, rep_num=repeat_num
        )
        hotload = LoadDefEDGES2.from_standard_layout(
            rootdir, "HotLoad", run_num=run_num, rep_num=repeat_num
        )
        ambient = LoadDefEDGES2.from_standard_layout(
            rootdir, "Ambient", run_num=run_num, rep_num=repeat_num
        )

        # Try getting the female receiver resistance from a definition.yaml
        if receiver_female_resistance is None or male_resistance is None:
            if (defn := rootdir / "definition.yaml").exists():
                with defn.open("r") as fl:
                    dd = yaml.safe_load(fl)

                    if receiver_female_resistance is None:
                        receiver_female_resistance = (
                            dd.get("measurements", {})
                            .get("resistance_f", {})
                            .get(run_num, 50 * units.ohm)
                        )

                    if male_resistance is None:
                        male_resistance = (
                            dd.get("measurements", {})
                            .get("resistance_m", {})
                            .get(run_num, 50 * units.ohm)
                        )

        if male_resistance is None:
            male_resistance = 50 * units.ohm
        if receiver_female_resistance is None:
            receiver_female_resistance = 50 * units.ohm

        return cls(
            open=_open,
            short=short,
            hot_load=hotload,
            ambient=ambient,
            switching_state=swsate,
            receiver_s11=rcv,
            run_num=run_num,
            receiver_female_resistance=receiver_female_resistance,
            male_resistance=male_resistance,
        )
