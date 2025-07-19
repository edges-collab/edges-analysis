"""Test config class."""

from typing import ClassVar

import pytest

from edges.config import Config


@pytest.fixture(scope="module")
def cfg():
    c = Config()
    c._add_to_schema({"key": "value", "dict": {"key": "value"}})
    return c


def test_use(cfg):
    assert cfg["key"] == "value"

    with cfg.use(key="new"):
        assert cfg["key"] == "new"

    assert cfg["key"] == "value"

    with cfg.use(dict={"key": "new"}):
        assert cfg["dict"]["key"] == "new"

    assert cfg["dict"]["key"] == "value"


def test_write_and_load(cfg, tmpdir):
    cfg.write(tmpdir / "config.yaml")

    assert cfg.path == tmpdir / "config.yaml"

    cfg2 = Config.load(cfg.path)
    for k, v in cfg.items():
        if not isinstance(v, dict):
            assert v == cfg2[k]
        else:
            for kk, vv in v.items():
                assert cfg2[k][kk] == vv


def test_aliases(tmpdir):
    class NewConfig(Config):
        _aliases: ClassVar = {"key": ("old_key",)}
        _defaults: ClassVar = {"key": "value"}

    with pytest.warns(
        UserWarning, match="Your configuration spec has old key 'old_key'"
    ):
        c = NewConfig(path=tmpdir / "cfg-example.yml", old_key="old_value")

    assert c["key"] == "old_value"

    c1 = NewConfig.load(tmpdir / "cfg-example.yml")
    assert c1._loaded_from_file
    assert c1["key"] == "old_value"


def test_extra_keys():
    class NewConfig(Config):
        _aliases: ClassVar = {"key": ("old_key",)}
        _defaults: ClassVar = {"key": "value"}

    c = NewConfig(new_key="new_value")
    assert c["new_key"] == "new_value"
    assert "new_key" not in c._defaults


def test_bad_load():
    with pytest.raises(ValueError, match="cannot have been loaded"):
        Config(_loaded_from_file=True)


def test_cant_use_nonexistent(cfg):
    with pytest.raises(KeyError, match="Cannot use bad in config"):  # noqa: SIM117
        with cfg.use(bad="bad"):
            pass
