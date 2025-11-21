"""A module defining the overall file structure and internal contents of cal obs.

This module defines the overall file structure and internal contents of the
calibration observations. It does *not* implement any algorithms/methods on that data,
making it easier to separate the algorithms from the data checking/reading.
"""

import tomllib as toml
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import attrs
import yaml
from astropy import units as un
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
class CalkitFileSpec:
    """File-specification for calkit S11 measurements.

    Parameters
    ----------
    match
        The S11 measurements of the match standard.
    open
        The S11 measurements of the open standard.
    short
        The S11 measurements of the short standard.
    """

    match: Path = attrs.field(converter=Path, validator=_vld_path_exists)
    open: Path = attrs.field(converter=Path, validator=_vld_path_exists)
    short: Path = attrs.field(converter=Path, validator=_vld_path_exists)

    @classmethod
    def from_edges2_layout(
        cls,
        direc: tp.PathLike,
        repeat_num: int = 1,
        allow_other: bool = True,
        prefix: str = "",
    ) -> Self:
        """
        Create a CalkitEdges2 object from a standard directory layout.

        Parameters
        ----------
        direc
            The directory to search.
        repeat_num
            The repeat num to use.
        allow_other
            Whether to allow other repeat numbers if the one specified is not found.
        prefix
            A prefix for the files. Sometimes this is necessary to find, e.g.
            External<load>.s1p
        """
        direc = Path(direc)
        open_ = direc / f"{prefix}Open{repeat_num:02}.s1p"

        if not open_.exists():
            if allow_other:
                open_ = next(direc.glob(f"{prefix}Open*.s1p"))

                warnings.warn(
                    f"Could not find {prefix}Open{repeat_num:02} in {direc}, using"
                    f" {open_.name}",
                    stacklevel=2,
                )
            else:
                raise OSError(f"Could not find {prefix}Open{repeat_num:02} in {direc}")

        rep_num = open_.stem[-2:]

        return cls(
            open=direc / f"{prefix}Open{rep_num}.s1p",
            short=direc / f"{prefix}Short{rep_num}.s1p",
            match=direc / f"{prefix}Match{rep_num}.s1p",
        )


@attrs.define()
class LoadS11:
    """File-specification for the S11 measurements of a calibration load.

    Parameters
    ----------
    calkit
        The calkit measurements of the load.
    external
        The external S11 measurement of the load.
    """

    calkit: CalkitFileSpec = attrs.field(
        validator=attrs.validators.instance_of(CalkitFileSpec)
    )
    external: Path = attrs.field(converter=Path, validator=_vld_path_exists)


@attrs.define(frozen=True, kw_only=True)
class LoadDefEDGES2:
    """File-specification for an EDGES2 calibration load.

    Parameters
    ----------
    name
        The name of the load.
    thermistor
        The thermistor measurements of the load.
    s11
        The S11 measurements of the load.
    spectra
        The spectra measurements of the load.
    sparams_file
        This file, if given, contains the S-parameters of the load device (e.g. the
        semi-rigid cable for a hot load).
    """

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
        """Createa a LoadDefEDGES2 object from a standard directory layout.

        Parameters
        ----------
        root
            The root directory of the observation.
        loadname
            The name of the load to search for.
        run_num
            The run number to search for.
        rep_num
            The repeat number to search for.
        sparams_file
            An optional file containing S-parameters of the load device (e.g. the
            semi-rigid cable for a hot load).
        """
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
        clk = CalkitFileSpec.from_edges2_layout(s11dir, rep_num)
        repnum = clk.open.stem[-2:]
        s11 = LoadS11(calkit=clk, external=s11dir / f"External{repnum}.s1p")

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


def _yamlfile_to_dict(val: Path | str | dict) -> dict:
    """Convert a Path/str/dict to a dict."""
    if isinstance(val, dict):
        return val
    if isinstance(val, (Path, str)):
        with Path(val).open("r") as fl:
            return yaml.safe_load(fl)
    else:
        raise TypeError("Value must be a dict, Path, or str!")


@attrs.define(frozen=True, kw_only=True)
class InternalSwitch:
    """File-specification for VNA measurements to compute Sparams of internal switch.

    Parameters
    ----------
    internal
        Measurements of the internal calkit loads.
    external
        Measurements of the external calkit loads (plugged in at the source
        input).
    """

    internal: CalkitFileSpec = attrs.field(
        validator=attrs.validators.instance_of(CalkitFileSpec)
    )
    external: CalkitFileSpec = attrs.field(
        validator=attrs.validators.instance_of(CalkitFileSpec)
    )
    _metadata: dict = attrs.field(converter=_yamlfile_to_dict)

    @_metadata.default
    def _meta_yaml_default(self) -> Path:
        return self.internal.open.parent / "metadata.yaml"

    @property
    def calkit_match_resistance(self) -> tp.OhmType:
        """The measured resistance of the match standard in the calkit."""
        return self._metadata["calkit_match_resistance"] * un.ohm

    @property
    def temperature(self) -> tp.TemperatureType | None:
        """The temperature at which the internal switch measurements were made."""
        try:
            return self._metadata["temperature"] * un.Celsius
        except KeyError:
            return None

    @property
    def external_calkit(self):
        """The name of the calkit used for the calkit measurements."""
        return self._metadata["external_calkit"]


@attrs.define(frozen=True, kw_only=True)
class ReceiverS11:
    """File-specification for the receiver S11 measurement.

    Parameters
    ----------
    calkit
        The calkit measurements of the receiver S11.
    device
        The external measurements of the receiver S11.
    """

    calkit: CalkitFileSpec = attrs.field(
        validator=attrs.validators.instance_of(CalkitFileSpec)
    )
    device: Path = attrs.field(converter=Path, validator=_vld_path_exists)
    _metadata: dict = attrs.field(converter=_yamlfile_to_dict)

    @property
    def external(self) -> Path:
        """Alias for the 'device' measurement."""
        return self.device

    @_metadata.default
    def _meta_yaml_default(self) -> Path:
        return self.device.parent / "metadata.yaml"

    @property
    def calkit_match_resistance(self):
        """The measured resistance of the match standard in the calkit."""
        return self._metadata["calkit_match_resistance"]

    @property
    def calkit_name(self):
        """The name of the calkit used for the calkit measurements."""
        return self._metadata["calkit"]


@attrs.define(frozen=True, kw_only=True)
class HotLoadSemiRigidCable:
    """File specification for the hot-load semi-rigid cable S-parameters."""

    osl: CalkitFileSpec = attrs.field(
        validator=attrs.validators.instance_of(CalkitFileSpec)
    )
    _metadata: dict = attrs.field(converter=_yamlfile_to_dict)

    @_metadata.default
    def _meta_yaml_default(self):
        return self.osl.open.parent / "metadata.yaml"

    @property
    def calkit(self) -> str:
        """The model name of the calkit used for the calkit measurements."""
        return self._metadata["calkit"]

    @property
    def calkit_match_resistance(self) -> tp.ImpedanceType:
        """The measured resistance of the match standard in the calkit."""
        return self._metadata["calkit_match_resistance"] * un.ohm


@attrs.define(frozen=True, kw_only=True)
class CalObsDefEDGES2:
    """File-specification for a full calibration observation with EDGES-2.

    Parameters
    ----------
    open
        The open load definition.
    short
        The short load definition.
    ambient
        The ambient load definition.
    hot_load
        The hot load definition.
    internal_switch
        The internal switch definition.
    receiver_s11
        The receiver S11 definition.
    run_num
        The run number used throughout the files.
    receiver_female_resistance
        The female resistance of the receiver, used in calibrating the calkit
        standards measurements.
    male_resistance
        The male resistance.
    """

    open: LoadDefEDGES2 = attrs.field()
    short: LoadDefEDGES2 = attrs.field()
    ambient: LoadDefEDGES2 = attrs.field()
    hot_load: LoadDefEDGES2 = attrs.field()

    internal_switch: InternalSwitch | list[InternalSwitch] = attrs.field()
    receiver_s11: ReceiverS11 | list[ReceiverS11] = attrs.field()

    run_num: int = attrs.field(default=None)

    @property
    def loads(self) -> dict[str, LoadDefEDGES2]:
        """A dictionary of the loads."""
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
    ) -> Self:
        """
        Create a CalObsDefEDGES2 object from a standard directory layout.

        Parameters
        ----------
        rootdir
            The root directory of the observation.
        run_num
            The run number to search for (often there is only one run).
        repeat_num
            The repeat number to search for (generally, repeats are taken closer
            together than "runs").
        """
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
            rcvdir = sorted((rootdir / "S11").glob("ReceiverReading*"))[0]
            warnings.warn(
                f"Could not find ReceiverReading{run_num:02}, using {rcvdir.name}",
                stacklevel=2,
            )

        run_num = int(rcvdir.stem[-2:])
        rcv_calkit = CalkitFileSpec.from_edges2_layout(rcvdir, repeat_num)
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
                f"Could not find SwitchingState{run_num:02}, using {swstate.name}",
                stacklevel=2,
            )

        sw_calkit = CalkitFileSpec.from_edges2_layout(swstate, repeat_num)
        repnum = int(sw_calkit.open.stem[-2:])
        swsate = InternalSwitch(
            internal=sw_calkit,
            external=CalkitFileSpec.from_edges2_layout(
                swstate, repeat_num=repnum, allow_other=False, prefix="External"
            ),
        )

        # Now, get the Loads
        open_ = LoadDefEDGES2.from_standard_layout(
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

        return cls(
            open=open_,
            short=short,
            hot_load=hotload,
            ambient=ambient,
            internal_switch=swsate,
            receiver_s11=rcv,
            run_num=run_num,
        )
