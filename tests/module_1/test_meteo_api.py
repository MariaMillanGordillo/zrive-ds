import pytest
from src.module_1.module_1_meteo_api import (
    call_api,
    get_data_meteo_api,
    validate_response,
    process_data,
)

# Pre-fetch response for tests
@pytest.fixture
def madrid_response():
    return get_data_meteo_api("Madrid")


def test_call_api():
    response = call_api("Madrid")
    assert response is not None, "API call failed, no response returned."


def test_get_data_meteo_api(response=madrid_response):
    assert hasattr(response, "Daily"), "Response does not contain 'Daily' data."

# Fixture  with fixed response for validation and processing tests
@pytest.fixture
def fixed_response():
    return {
        "daily": {
            "time": ["2025-09-29T00:00", "2025-09-30T00:00"],
            "temperature_2m_mean": [20, 21],
            "precipitation_sum": [0, 0],
            "wind_speed_10m_max": [5, 6]
        }
    }


def test_validate_response_valid(fixed_response):
    result = validate_response(fixed_response)
    assert result is None, "Valid response failed validation."


# Invalid response missing required fields
@pytest.fixture
def invalid_response():
    return {"daily": {"time": [], "temperature_2m_mean": []}}

def test_validate_response_invalid(invalid_response):
    with pytest.raises(ValueError):
        validate_response(invalid_response) # Expecting ValueError for invalid response


def test_process_data(response=madrid_response):
    dataframe = process_data(response, "Madrid")
    assert not dataframe.empty, "Processed DataFrame is empty."
    expected_columns = [
        "date",
        "temperature_2m_mean",
        "precipitation_sum",
        "wind_speed_10m_max",
        "city",
    ]
    for col in expected_columns:
        assert col in dataframe.columns, f"Missing expected column: {col}"
