from unittest.mock import Mock, patch

import pytest

from src.module_1.module_1_meteo_api import (
    VARIABLES,
    call_api,
    get_data_meteo_api,
    process_data,
    validate_response,
)


# Fixed valid response for testing
@pytest.fixture
def valid_mock_response():
    """Simulate a WeatherApiResponse object."""
    mock_resp = Mock()

    # Mock the Daily() method
    daily_mock = Mock()
    daily_mock.Time.return_value = 0
    daily_mock.TimeEnd.return_value = 86400
    daily_mock.Interval.return_value = 86400

    # Mock Variable for temperature, precipitation, wind
    daily_mock.VariablesCount.return_value = len(VARIABLES)
    daily_mock.Variables.side_effect = [
        Mock(ValuesAsNumpy=Mock(return_value=[20])) for _ in VARIABLES
    ]

    mock_resp.Daily.return_value = daily_mock
    mock_resp.error = False  # no API error
    return mock_resp


# Invalid response missing required fields
@pytest.fixture
def invalid_mock_response():
    """Simulate an invalid WeatherApiResponse object."""
    mock_resp = Mock()

    # Mock the Daily() method with missing fields
    daily_mock = Mock()
    daily_mock.Time.return_value = 0
    daily_mock.TimeEnd.return_value = 86400
    daily_mock.Interval.return_value = 86400

    # Missing Variables method to simulate invalid response
    daily_mock.VariablesCount.return_value = len(VARIABLES) - 1
    daily_mock.Variables.side_effect = [
        Mock(ValuesAsNumpy=Mock(return_value=[20])) for _ in VARIABLES[:-1]
    ]

    mock_resp.Daily.return_value = daily_mock
    mock_resp.error = False  # no API error
    return mock_resp


# Use mocking for API call tests
def test_call_api_mock(valid_mock_response):
    """Test call_api using a mock to avoid calling the real API."""
    with patch(
        "src.module_1.module_1_meteo_api.openmeteo.weather_api",
        return_value=valid_mock_response,
    ) as mock_api:
        result = call_api("Madrid")

        # The function should return the mock object
        assert result == valid_mock_response
        # The API function should be called once
        mock_api.assert_called_once()


def test_get_data_meteo_mock(valid_mock_response):
    """Test get_data_meteo_api using mocks for API call and validate_response."""
    # Patch the API call inside get_data_meteo_api
    with patch(
        "src.module_1.module_1_meteo_api.call_api", return_value=[valid_mock_response]
    ) as mock_call:
        with patch(
            "src.module_1.module_1_meteo_api.validate_response"
        ) as mock_validate:
            result = get_data_meteo_api("Madrid")

            # Check the returned value is the mocked API response
            assert result == valid_mock_response

            # Ensure call_api was called correctly
            mock_call.assert_called_once_with("Madrid")

            # Ensure validate_response was called with the API response
            mock_validate.assert_called_once_with(valid_mock_response)


def test_validate_response_valid(valid_mock_response):
    result = validate_response(valid_mock_response)
    assert result is None


def test_validate_response_invalid(invalid_mock_response):
    with pytest.raises(ValueError):
        validate_response(invalid_mock_response)  # Expecting ValueError


def test_process_data(valid_mock_response):
    """process_data should return a valid DataFrame with expected columns."""
    df = process_data(valid_mock_response, "Madrid")
    assert not df.empty

    expected_columns = [
        "date",
        "temperature_2m_mean",
        "precipitation_sum",
        "wind_speed_10m_max",
        "city",
    ]
    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"

    # Check that all columns have the same length
    lengths = [len(df[col]) for col in expected_columns]
    assert len(set(lengths)) == 1
