"""Test config class."""

import copy
from pathlib import Path

import pytest

from edges.config import Config, config


@pytest.fixture(scope="module")
def cfg():
    return Config()  # not the global one.


def test_use(cfg):
    # Compare against a fresh default Config rather than the global config, which
    # is loaded from the user's config file (if any) and so is machine-dependent.
    assert cfg == Config()

    with cfg.use(beams=Path("/a/path")):
        assert cfg.beams == Path("/a/path")

    assert cfg == Config()  # returned to normal


def test_use_global_config_restores():
    before = copy.deepcopy(config)
    with config.use(beams=Path("/a/path"), raw_lab_data=Path("/b")):
        assert config.beams == Path("/a/path")
        assert config.raw_lab_data == Path("/b")
    assert config == before


def test_write_and_load(cfg, tmpdir):
    cfg.write(tmpdir / "config.yaml")

    cfg2 = Config.load(tmpdir / "config.yaml")
    print(cfg.antenna)
    print(cfg2.antenna)
    assert cfg == cfg2


def test_cant_use_nonexistent(cfg):
    with pytest.raises(KeyError, match="Cannot use bad in config"):  # ruff: ignore[multiple-with-statements]
        with cfg.use(bad="bad"):
            pass
