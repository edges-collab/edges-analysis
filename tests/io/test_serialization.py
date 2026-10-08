from datetime import datetime

import attrs
import h5py
import hickle
import numpy as np
import pytest
from astropy.coordinates import EarthLocation
from astropy.table import QTable
from astropy.time import Time
from astropy.units import Quantity, Unit
from astropy.units.core import UnitBase

from edges.io.serialization import converter, hickleable


@hickleable
@attrs.define
class _HickleBase:
    x: int


@hickleable
@attrs.define
class _HickleChild(_HickleBase):
    y: float = 1.0


def test_numpy_ndarray_hook_structures_list_to_ndarray():
    arr = converter.structure([1, 2, 3], np.ndarray)
    assert isinstance(arr, np.ndarray)
    assert np.all(arr == np.array([1, 2, 3]))


def test_numpy_ndarray_generic_type_hook_structures_list_to_ndarray():
    # This hits the hook registered with register_structure_hook_func that matches
    # annotations like ``np.ndarray[...`` (origin is ``np.ndarray``).
    typ = np.ndarray[tuple[int], np.dtype[np.float64]]
    arr = converter.structure([1, 2, 3], typ)
    assert isinstance(arr, np.ndarray)
    assert np.all(arr == np.array([1, 2, 3]))


def test_astropy_unit_hook_structures_str_and_passes_unitbase_through():
    u = converter.structure("mK", Unit)
    assert isinstance(u, Unit)
    assert str(u) == "mK"

    u2 = converter.structure(Unit("K"), Unit)
    assert isinstance(u2, UnitBase)
    assert str(u2) == "K"

    with pytest.raises(TypeError):
        converter.structure(123, Unit)


def test_astropy_quantity_hook_roundtrip():
    q = 3.5 * Unit("m")
    data = converter.unstructure(q)
    assert set(data.keys()) == {"value", "unit", "dtype"}

    q2 = converter.structure(data, Quantity)
    assert isinstance(q2, Quantity)
    assert q2.unit == q.unit
    assert np.all(q2.value == q.value)


def test_astropy_time_hook_roundtrip():
    t = Time(2451545.0, format="jd")
    raw = converter.unstructure(t)
    assert raw["jd1"] + raw["jd2"] == 2451545.0
    assert raw["scale"] == "utc"

    t2 = converter.structure(raw, Time)
    assert isinstance(t2, Time)
    assert t2.jd == t.jd
    assert t2.scale == "utc"
    assert t2.location is None


MRO = EarthLocation(lat=-26.714778 * Unit("deg"), lon=116.605528 * Unit("deg"))


def _assert_time_identical(t2: Time, t: Time):
    assert t2.scale == t.scale
    assert t2.format == t.format
    assert t2.shape == t.shape
    np.testing.assert_array_equal(t2.jd1, t.jd1)
    np.testing.assert_array_equal(t2.jd2, t.jd2)
    np.testing.assert_allclose((t2 - t).to_value("s"), 0, rtol=0, atol=1e-9)


@pytest.mark.parametrize(
    "t",
    [
        Time("2020-01-01T12:34:56.123456", scale="tt", location=MRO),
        Time(
            ["2020-01-01T12:34:56.000001", "2021-06-30T23:59:59.999999"],
            scale="tai",
            format="isot",
        ),
        Time(2459856.123456789, 1.234e-11, format="jd", scale="utc", location=MRO),
    ],
)
def test_astropy_time_roundtrip_is_lossless(t: Time):
    t2 = converter.structure(converter.unstructure(t), Time)
    _assert_time_identical(t2, t)
    if t.location is None:
        assert t2.location is None
    else:
        for a, b in zip(
            t2.location.to_geocentric(), t.location.to_geocentric(), strict=True
        ):
            np.testing.assert_allclose(
                a.to_value("m"), b.to_value("m"), rtol=0, atol=1e-6
            )
        # The location is needed for sidereal times.
        np.testing.assert_allclose(
            t2.sidereal_time("apparent").hour,
            t.sidereal_time("apparent").hour,
            rtol=0,
            atol=1e-12,
        )


def test_astropy_time_reads_old_jd_format():
    # Files written before the lossless format stored Time as bare UTC JDs.
    t2 = converter.structure(np.array([2451545.0, 2451546.5]), Time)
    assert t2.scale == "utc"
    np.testing.assert_array_equal(t2.jd, [2451545.0, 2451546.5])

    t3 = converter.structure(2451545.0, Time)
    assert t3.jd == 2451545.0


@hickleable
@attrs.define
class _HasTime:
    t: Time


def test_hickleable_time_roundtrip_file(tmp_path):
    t = Time("2020-01-01T12:34:56.123456", scale="tt", location=MRO)
    path = tmp_path / "time.h5"
    _HasTime(t=t).write(path)
    new = _HasTime.from_file(path)
    _assert_time_identical(new.t, t)
    assert new.t.location is not None


def test_hickleable_time_reads_old_file(tmp_path):
    path = tmp_path / "old.h5"
    hickle.dump(
        {"t": np.array([2451545.0, 2451546.0]), "__classname__": "_HasTime"}, path
    )
    new = _HasTime.from_file(path)
    assert new.t.scale == "utc"
    np.testing.assert_array_equal(new.t.jd, [2451545.0, 2451546.0])


def test_datetime_hook_roundtrip():
    dt = datetime(2020, 1, 2, 3, 4, 5)
    raw = converter.unstructure(dt)
    assert raw == "2020-01-02T03:04:05"

    dt2 = converter.structure(raw, datetime)
    assert dt2 == dt


def test_location_hook_roundtrip():
    loc = EarthLocation(
        lat=1.0 * Unit("deg"), lon=2.0 * Unit("deg"), height=3.0 * Unit("m")
    )
    raw = converter.unstructure(loc)
    assert isinstance(raw, list)
    assert len(raw) == 3

    loc2 = converter.structure(raw, EarthLocation)
    assert isinstance(loc2, EarthLocation)
    assert loc2.lat.to_value(Unit("deg")) == pytest.approx(1.0)
    assert loc2.lon.to_value(Unit("deg")) == pytest.approx(2.0)
    assert loc2.height.to_value(Unit("m")) == pytest.approx(3.0)


def test_qtable_hook_roundtrip():
    qt = QTable(
        data={
            "a": np.array([1, 2, 3]),
            "b": np.array([4.0, 5.0, 6.0]) * Unit("m"),
        }
    )
    raw = converter.unstructure(qt)
    assert isinstance(raw, dict)
    assert set(raw.keys()) == {"a", "b"}

    qt2 = converter.structure(raw, QTable)
    assert isinstance(qt2, QTable)
    assert np.all(qt2["a"] == qt["a"])
    assert np.all(qt2["b"] == qt["b"])


def test_hickleable_roundtrip_and_subclass_structuring(tmp_path):
    obj: _HickleBase = _HickleChild(x=3, y=2.5)
    path = tmp_path / "obj.h5"

    obj.write(path)
    new = _HickleBase.from_file(path)

    assert isinstance(new, _HickleChild)
    assert new == obj


def test_write_and_load_hdf5_accepts_h5py_group(tmp_path):
    @hickleable
    @attrs.define
    class Thing:
        x: int

    obj = Thing(x=7)
    path = tmp_path / "thing.h5"

    with h5py.File(path, "w") as f:
        obj.write(f)

    with h5py.File(path, "r") as f:
        new = Thing.from_file(f)

    assert new == obj


def test_hickleable_rejects_non_attrs_class():
    class NotAttrs:
        pass

    with pytest.raises(TypeError, match="needs to be attrs"):
        hickleable(NotAttrs)


def test_hickleable_rejects_untyped_fields():
    @attrs.define
    class HasUntyped:
        x = attrs.field()

    with pytest.raises(TypeError, match="untyped fields"):
        hickleable(HasUntyped)
