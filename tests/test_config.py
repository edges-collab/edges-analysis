"""Test config class."""

import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from edges import config as edges_config
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


def test_write_default_path(cfg, tmp_path, monkeypatch):
    fname = tmp_path / "sub" / "config.yaml"
    monkeypatch.setattr(edges_config, "_config_filename", fname)
    cfg.write()
    assert Config.load(fname) == cfg


def test_written_file_is_plain_yaml(cfg, tmp_path):
    cfg.write(tmp_path / "config.yaml")
    raw = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert raw["beams"] == str(cfg.beams)


def test_load_partial_and_empty(tmp_path):
    fl = tmp_path / "config.yaml"
    fl.write_text("beams: /some/where\n")
    assert Config.load(fl) == Config(beams=Path("/some/where"))

    fl.write_text("")
    assert Config.load(fl) == Config()


def test_load_rejects_unsafe_yaml(tmp_path):
    fl = tmp_path / "config.yaml"
    fl.write_text("beams: !!python/object/apply:os.getcwd []\n")
    with pytest.raises(yaml.YAMLError):
        Config.load(fl)


def test_load_rejects_non_mapping(tmp_path):
    fl = tmp_path / "config.yaml"
    fl.write_text("- a\n- b\n")
    with pytest.raises(TypeError, match="must contain a mapping"):
        Config.load(fl)


def test_user_config_missing(tmp_path, recwarn):
    assert edges_config._load_user_config(tmp_path / "nope.yaml") == Config()
    assert not recwarn.list


@pytest.mark.parametrize(
    "content",
    [
        "beams: [unclosed\n",  # invalid YAML
        "beams: {a: 1}\n",  # wrong type
        "- a\n- b\n",  # not a mapping
        "beams: !!python/object/apply:os.getcwd []\n",  # unsafe tag
    ],
)
def test_user_config_malformed_falls_back(tmp_path, content):
    fl = tmp_path / "config.yaml"
    fl.write_text(content)
    with pytest.warns(UserWarning, match="Could not read the edges config file"):
        out = edges_config._load_user_config(fl)
    assert out == Config()


def test_user_config_valid(tmp_path):
    fl = tmp_path / "config.yaml"
    fl.write_text("raw_lab_data: /lab\n")
    assert edges_config._load_user_config(fl).raw_lab_data == Path("/lab")


@pytest.mark.skipif(sys.platform != "linux", reason="uses XDG_CONFIG_HOME")
def test_import_with_malformed_user_config(tmp_path):
    cfgdir = tmp_path / "edges"
    cfgdir.mkdir()
    (cfgdir / "config.yaml").write_text("beams: [unclosed\n")
    env = os.environ | {"XDG_CONFIG_HOME": str(tmp_path)}
    code = "import edges.config as c; print(c.config == c.Config())"
    proc = subprocess.run(
        [sys.executable, "-W", "always", "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "True"
    assert "Could not read the edges config file" in proc.stderr
