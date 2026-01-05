from edges.data.cli import app


def test_fetch_b18_testing(capsys):
    app("data fetch_b18 --dataset testing", result_action="return_value")
    out = capsys.readouterr().out
    assert "/S11" in out
    assert "/Resistance" in out
    assert "Fetched B18 calibration testing data files:" in out


def test_fetch_b18_none(capsys):
    app("data fetch_b18 --dataset none", result_action="return_value")
    out = capsys.readouterr().out
    assert "No files fetched." in out
