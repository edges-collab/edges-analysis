from edges_analysis.config import Config


def test_load_equality(tmp_path_factory):
    d = tmp_path_factory.mktemp("configs")

    cfg1 = Config()
    cfg1.write(d / "config.yml")

    cfg2 = Config.load(d / "config.yml")

    assert cfg1 == cfg2


def test_simple_inclusion():
    cfg1 = Config()
    assert "paths" in cfg1
