"""Test config class."""

from pathlib import Path

import pytest

from edges.config import Config, config


@pytest.fixture(scope="module")
def cfg():
    return Config()  # not the global one.


def test_use(cfg):
    assert cfg == config

    with cfg.use(beams=Path("/a/path")):
        assert cfg.beams == Path("/a/path")

    assert cfg == config  # returned to normal


def test_write_and_load(cfg, tmpdir):
    cfg.write(tmpdir / "config.yaml")

    cfg2 = Config.load(tmpdir / "config.yaml")
    print(cfg.antenna)
    print(cfg2.antenna)
    assert cfg == cfg2


def test_cant_use_nonexistent(cfg):
    with pytest.raises(KeyError, match="Cannot use bad in config"):  # noqa: SIM117
        with cfg.use(bad="bad"):
            pass
