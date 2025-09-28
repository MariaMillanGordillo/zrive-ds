from src.module_1.module_1_meteo_api import (
    call_api,
    get_data_meteo_api,
    validate_response,
    process_data,
)


def test_call_api():
    response = call_api("Madrid")
    assert response is not None, "API call failed, no response returned."
    print("test_call_api passed.")


# Pre-fetch response for other tests
response = get_data_meteo_api("Madrid")


def test_get_data_meteo_api(response=response):
    assert hasattr(response, "Daily"), "Response does not contain 'Daily' data."
    print("test_get_data_meteo_api passed.")


def test_validate_response(response=response):
    try:
        validate_response(response)
        print("test_validate_response passed.")
    except ValueError as e:
        assert False, f"Response validation failed: {e}"


def test_process_data(response=response):
    validate_response(response)
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
    print("test_process_data passed.")


def test_all(response=response):
    test_call_api()
    test_get_data_meteo_api()
    test_validate_response()
    test_process_data()
    print("All tests passed.")


if __name__ == "__main__":
    test_all()
