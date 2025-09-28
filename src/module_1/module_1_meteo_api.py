import openmeteo_requests
import pandas as pd
import matplotlib.pyplot as plt

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

# API functions


def call_api(city, url=API_URL, params=None, retries=3, backoff=3):
    """ Call the Open-Meteo API for a given city and return the response.
    Inputs:
        city (str): Name of the city to get data for.
        url (str): API endpoint URL.
        params (dict): Additional parameters for the API call.
        retries (int): Number of retries for the API call in case of failure.
        backoff (int): Backoff time in seconds between retries.
    Returns:
        response (object): API response object.
    """
    if not params:
        params = {
            "latitude": COORDINATES[city]["latitude"],
            "longitude": COORDINATES[city]["longitude"],
            "start_date": START_DATE,
            "end_date": END_DATE,
            "daily": ",".join(VARIABLES),
            "timezone": "auto",
        }

    for attempt in range(retries):
        try:
            responses = openmeteo.weather_api(url, params=params)
            return responses
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {backoff} seconds...")
                import time

                time.sleep(backoff)
            else:
                print("All attempts failed.")
                raise e

    return responses


def validate_response(response):
    """ Validate the API response for errors and expected data.
    Inputs:
        response (object): API response object.
    Raises:
        ValueError: If the response contains an error or is missing expected data.
    """
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
    """ Get and validate data from the Open-Meteo API for a given city.
    Inputs:
        city (str): Name of the city to get data for.
    Returns:
        response (object): Validated API response object.
    """
    # Call the API and validate the response
    responses = call_api(city)
    if not responses:
        print(f"No data returned for {city}")
        return

    response = responses[0]
    validate_response(response)
    return response


# Data processing functions


def process_data(response, city):
    """ Process the API response and convert it into a pandas DataFrame.
    Inputs:
        response (object): Validated API response object.
        city (str): Name of the city the data corresponds to.
    Returns:
        daily_dataframe (pd.DataFrame): DataFrame containing the processed daily data.
    """
    # Process daily data
    daily = response.Daily()
    daily_temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()
    daily_precipitation_sum = daily.Variables(1).ValuesAsNumpy()
    daily_wind_speed_10m_max = daily.Variables(2).ValuesAsNumpy()

    daily_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left",
        )
    }

    daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
    daily_data["precipitation_sum"] = daily_precipitation_sum
    daily_data["wind_speed_10m_max"] = daily_wind_speed_10m_max
    daily_data["city"] = city
    daily_dataframe = pd.DataFrame(data=daily_data)

    return daily_dataframe


# Data plotting functions


def plot_temperature(dataframe):
    """ Plot the average temperature data for each city per month.
    Inputs:
        dataframe (pd.DataFrame): DataFrame containing the daily data for all cities.
    Returns:
        None
    """
    dataframe["month"] = dataframe["date"].dt.tz_localize(None).dt.to_period("M")
    monthly_avg = (
        dataframe.groupby(["city", "month"])["temperature_2m_mean"].mean().reset_index()
    )
    monthly_avg["month"] = monthly_avg["month"].dt.to_timestamp()

    plt.figure(figsize=(12, 6))
    for city in monthly_avg["city"].unique():
        city_data = monthly_avg[monthly_avg["city"] == city]
        plt.plot(city_data["month"], city_data["temperature_2m_mean"], label=city)

    plt.title("Average Monthly Temperature by City")
    plt.xlabel("Month")
    plt.ylabel("Average Temperature (°C)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_precipitation(dataframe):
    """ Plot the total precipitation data for each city per month.
    Inputs:
        dataframe (pd.DataFrame): DataFrame containing the daily data for all cities.
    Returns:
        None
    """
    dataframe["month"] = dataframe["date"].dt.tz_localize(None).dt.to_period("M")
    monthly_total = (
        dataframe.groupby(["city", "month"])["precipitation_sum"].sum().reset_index()
    )
    monthly_total["month"] = monthly_total["month"].dt.to_timestamp()

    plt.figure(figsize=(12, 6))
    for city in monthly_total["city"].unique():
        city_data = monthly_total[monthly_total["city"] == city]
        plt.plot(city_data["month"], city_data["precipitation_sum"], label=city)

    plt.title("Total Monthly Precipitation by City")
    plt.xlabel("Month")
    plt.ylabel("Total Precipitation (mm)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_wind(dataframe):
    """ Plot the maximum wind speed data for each city per month.
    Inputs:
        dataframe (pd.DataFrame): DataFrame containing the daily data for all cities.
    Returns:
        None
    """
    dataframe["month"] = dataframe["date"].dt.tz_localize(None).dt.to_period("M")
    monthly_max = (
        dataframe.groupby(["city", "month"])["wind_speed_10m_max"].max().reset_index()
    )
    monthly_max["month"] = monthly_max["month"].dt.to_timestamp()

    plt.figure(figsize=(12, 6))
    for city in monthly_max["city"].unique():
        city_data = monthly_max[monthly_max["city"] == city]
        plt.plot(city_data["month"], city_data["wind_speed_10m_max"], label=city)

    plt.title("Maximum Monthly Wind Speed by City")
    plt.xlabel("Month")
    plt.ylabel("Maximum Wind Speed (m/s)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_per_city(dataframe):
    """ Plot all weather variables for each city per month.
    Inputs:
        dataframe (pd.DataFrame): DataFrame containing the daily data for all cities.
    Returns:
        None
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 7))
    cities = dataframe["city"].unique()

    for city in cities:
        city_data = dataframe[dataframe["city"] == city].copy()
        city_data["month"] = city_data["date"].dt.tz_localize(None).dt.to_period("M")

        # Temperature
        monthly_avg_temp = (
            city_data.groupby("month")["temperature_2m_mean"].mean().reset_index()
        )
        monthly_avg_temp["month"] = monthly_avg_temp["month"].dt.to_timestamp()
        axes[0].plot(
            monthly_avg_temp["month"],
            monthly_avg_temp["temperature_2m_mean"],
            label=city,
        )

        # Precipitation
        monthly_total_precip = (
            city_data.groupby("month")["precipitation_sum"].sum().reset_index()
        )
        monthly_total_precip["month"] = monthly_total_precip["month"].dt.to_timestamp()
        axes[1].plot(
            monthly_total_precip["month"],
            monthly_total_precip["precipitation_sum"],
            label=city,
        )

        # Wind Speed
        monthly_max_wind = (
            city_data.groupby("month")["wind_speed_10m_max"].max().reset_index()
        )
        monthly_max_wind["month"] = monthly_max_wind["month"].dt.to_timestamp()
        axes[2].plot(
            monthly_max_wind["month"],
            monthly_max_wind["wind_speed_10m_max"],
            label=city,
        )

    # Plot axes titles and labels
    axes[0].set_title("Average Monthly Temperature by City")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Average Temperature (°C)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_title("Total Monthly Precipitation by City")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Total Precipitation (mm)")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].set_title("Maximum Monthly Wind Speed by City")
    axes[2].set_xlabel("Month")
    axes[2].set_ylabel("Maximum Wind Speed (m/s)")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


def plot_all(dataframe):
    """ Plot all weather variables for all cities.
    Inputs:
        dataframe (pd.DataFrame): DataFrame containing the daily data for all cities.
    Returns:
        None
    """
    plot_temperature(dataframe)
    plot_precipitation(dataframe)
    plot_wind(dataframe)
    plot_per_city(dataframe)


# MAIN


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
        plot_all(combined_df)


if __name__ == "__main__":
    main()
