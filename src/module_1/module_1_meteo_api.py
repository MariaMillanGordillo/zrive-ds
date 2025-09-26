import openmeteo_requests
import pandas as pd
import matplotlib.pyplot as plt

# AÑADIR PABLO EN EL REPO

# Setup the Open-Meteo API client
openmeteo = openmeteo_requests.Client()

# Variables and functions for the Meteo API module
API_URL = "https://archive-api.open-meteo.com/v1/archive"
COORDINATES = { 
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
VARIABLES = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]
START_DATE = "2010-01-01"
END_DATE = "2020-12-31"

## API functions

def call_api(city, url=API_URL):
    params = {
        "latitude": COORDINATES[city]["latitude"],
        "longitude": COORDINATES[city]["longitude"],
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ",".join(VARIABLES),
        "timezone": "auto",
    }
    responses = openmeteo.weather_api(url, params=params)
    return responses

def validate_response(response):
    # Check if the response has an 'error' attribute and if it's True
    if hasattr(response, "error") and response.error:
        raise ValueError(f"API Error: {getattr(response, 'reason', 'Unknown error')}")
    
    # Check for daily data
    if not response.Daily():
        raise ValueError("No daily data in response")
    
    # Check for expected number of variables
    if getattr(response.Daily(), "VariablesCount", None) is not None:
        if response.Daily().VariablesCount() != len(VARIABLES):
            raise ValueError("Missing expected variables in response")
    else:
        for i in range(len(VARIABLES)):
            try:
                response.Daily().Variables(i)
            except Exception:
                raise ValueError(f"Missing variable at index {i} in response")

def get_data_meteo_api(city):
    # Call the API and validate the response
    responses = call_api(city)
    if not responses:
        print(f"No data returned for {city}")
        return

    response = responses[0]
    validate_response(response)
    return response

## Data processing functions

def process_data(response, city):
    # Process daily data
    daily = response.Daily()
    daily_temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()
    daily_precipitation_sum = daily.Variables(1).ValuesAsNumpy()
    daily_wind_speed_10m_max = daily.Variables(2).ValuesAsNumpy()

    daily_data = {"date": pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left"
    )}

    daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
    daily_data["precipitation_sum"] = daily_precipitation_sum
    daily_data["wind_speed_10m_max"] = daily_wind_speed_10m_max
    daily_data["city"] = city
    daily_dataframe = pd.DataFrame(data=daily_data)

    return daily_dataframe

## Data plotting functions

def plot_data(dataframe):
    pass

## TEST MAIN

def main():
    all_dataframes = []

    # Get and process data for each city
    for city in COORDINATES.keys():
        print(f"\nGetting data for {city}")
        response = get_data_meteo_api(city)
        if response:
            dataframe = process_data(response, city)
            all_dataframes.append(dataframe)

    # Combine all dataframes into one
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print("\nCombined DataFrame:\n")
        print(combined_df.sample(5))

        # Plot the combined data
        plot_data(combined_df)


if __name__ == "__main__":
    main()