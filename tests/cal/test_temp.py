import pytest
from pytest_cases import fixture_ref as fxref
from pytest_cases import parametrize


@pytest.fixture(scope="module")
def case_a():
    """Default frequencies."""
    return 3


@pytest.fixture(scope="module")
def case_b():
    """Default frequencies."""
    return 4


@parametrize("fxt", [fxref(case_a), fxref(case_b)])
def test_temp(fxt):
    """Test that uses the fixture."""
    assert fxt in [3, 4], f"Expected 3 or 4, got {fxt}"
