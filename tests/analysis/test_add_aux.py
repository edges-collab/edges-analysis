"""Tests of adding aux data from weather/thermlog files to GSData objects."""

import pytest
from astropy import units as un
from astropy.time import Time

from edges.analysis.aux_data import WeatherError, add_thermlog_data, add_weather_data
from edges.io import TEST_DATA_PATH
from edges.testing import create_mock_edges_data


def test_add_thermlog_data():
    t0 = Time("2020:104:14:00")

    gsd = create_mock_edges_data(
        flow=50 * un.MHz,
        fhigh=100 * un.MHz,
        ntime=100,
        time0=t0.jd,
    )

    out = add_thermlog_data(
        gsd, thermlog_file=TEST_DATA_PATH / "thermlog_test_data.txt"
    )

    assert "receiver_temp" in out.auxiliary_measurements.columns


def test_add_thermlog_data_out_of_bounds():
    t0 = Time("2020:115:14:00")

    gsd = create_mock_edges_data(
        flow=50 * un.MHz,
        fhigh=100 * un.MHz,
        ntime=100,
        time0=t0.jd,
    )

    with pytest.raises(WeatherError):
        add_thermlog_data(gsd, thermlog_file=TEST_DATA_PATH / "thermlog_test_data.txt")


def test_add_weather_data():
    t0 = Time("2020:107:14:00")

    gsd = create_mock_edges_data(
        flow=50 * un.MHz,
        fhigh=100 * un.MHz,
        ntime=100,
        time0=t0.jd,
    )

    out = add_weather_data(gsd, weather_file=TEST_DATA_PATH / "weather_test_data.txt")

    assert "ambient_temp" in out.auxiliary_measurements.columns


def test_add_weather_data_out_of_bounds():
    t0 = Time("2020:115:14:00")

    gsd = create_mock_edges_data(
        flow=50 * un.MHz,
        fhigh=100 * un.MHz,
        ntime=100,
        time0=t0.jd,
    )

    with pytest.raises(WeatherError):
        add_weather_data(gsd, weather_file=TEST_DATA_PATH / "weather_test_data.txt")
