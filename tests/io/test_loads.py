"""Tests of the names of the calibration loads."""

import pytest

from edges.io import loads


@pytest.mark.parametrize("name", ["ambient", "hot_load", "open", "short"])
def test_canonical_names_map_to_themselves(name: str):
    assert loads.canonical_load_name(name) == name


@pytest.mark.parametrize(
    ("name", "canonical"),
    [
        ("Ambient", "ambient"),
        ("HotLoad", "hot_load"),
        ("LongCableOpen", "open"),
        ("LongCableShorted", "short"),
    ],
)
def test_edges2_layout_names(name: str, canonical: str):
    assert loads.canonical_load_name(name) == canonical
    assert loads.EDGES2_LAYOUT_NAMES[canonical] == name


@pytest.mark.parametrize(
    ("name", "canonical"),
    [("amb", "ambient"), ("hot", "hot_load"), ("open", "open"), ("short", "short")],
)
def test_edges3_layout_names(name: str, canonical: str):
    assert loads.canonical_load_name(name) == canonical
    assert loads.EDGES3_LAYOUT_NAMES[canonical] == name


@pytest.mark.parametrize(
    ("name", "canonical"),
    [
        ("Hot_Load", "hot_load"),
        ("ambient_load", "ambient"),
        ("cable_open", "open"),
        ("LongCableShort", "short"),
    ],
)
def test_aliases(name: str, canonical: str):
    assert loads.canonical_load_name(name) == canonical


@pytest.mark.parametrize("name", ["not-a-load", "HOTLOAD", "box", "lna", "ant", ""])
def test_unknown_name_raises(name: str):
    with pytest.raises(ValueError, match="Unknown calibration load"):
        loads.canonical_load_name(name)


def test_canonical_load_keys():
    out = loads.canonical_load_keys(
        {"amb": 1, "HotLoad": 2, "open": 3, "default": 4}, other_keys=("default",)
    )
    assert out == {"ambient": 1, "hot_load": 2, "open": 3, "default": 4}


def test_canonical_load_keys_unknown_key():
    with pytest.raises(ValueError, match="Unknown calibration load 'default'"):
        loads.canonical_load_keys({"default": 1})


def test_canonical_load_keys_duplicate_load():
    with pytest.raises(ValueError, match="More than one key given for load 'hot_load'"):
        loads.canonical_load_keys({"hot": 1, "hot_load": 2})
