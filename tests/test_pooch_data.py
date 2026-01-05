from edges.data.cli import app


def test_fetch_b18_testing(capsys):
    app("fetch_b18", dataset="testing")
    assert "/S11" in capsys.readouterr().out
    assert "/Resistance" in capsys.readouterr().out
    assert "Fetched B18 calibration testing data files:" in capsys.readouterr().out


def test_fetch_b18_none(capsys):
    app("fetch_b18", dataset="none")
    assert "No files fetched." in capsys.readouterr().out
