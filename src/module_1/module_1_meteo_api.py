import time
import logging
import pandas as pd
import openmeteo_requests
from typing import Optional
import matplotlib.pyplot as plt

# Logging configuration
logging.basicConfig(
    level=logging.INFO, # Info level for general information
    format="%(asctime)s - %(levelname)s - %(message)s"
)

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


def call_api(
        city: str, 
        url: str = API_URL, 
        params: Optional[dict[str, str]] = None, 
        retries: int = 3, 
        backoff: int = 3
) -> Optional[dict]:
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
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                logging.info(f"Retrying in {backoff} seconds...")
                time.sleep(backoff)
            else:
                logging.critical("All attempts failed.")
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
        logging.error(f"No data returned for {city}")
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
    """ Plot the average monthly temperature for each city with subplots,
    comparing months across years.
    Inputs:
        dataframe (pd.DataFrame): DataFrame containing the daily data for all cities.
    Returns:
        None
    """
    df = dataframe.copy()
    df["date"] = df["date"].dt.tz_localize(None)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    monthly_avg = (
        df.groupby(["city", "year", "month"])["temperature_2m_mean"]
        .mean()
        .reset_index()
    )

    cities = monthly_avg["city"].unique()
    fig, axes = plt.subplots(len(cities), 1, figsize=(11, 8), sharex=True)

    for ax, city in zip(axes, cities):
        city_data = monthly_avg[monthly_avg["city"] == city]

        for year in city_data["year"].unique():
            year_data = city_data[city_data["year"] == year]
            ax.plot(
                year_data["month"],
                year_data["temperature_2m_mean"],
                marker="o",
                label=str(year)
            )

        ax.set_title(f"Average Monthly Temperature in {city}")
        ax.set_ylabel("°C")
        ax.legend(title="Year", loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True)

    axes[-1].set_xlabel("Month")
    axes[-1].set_xticks(range(1, 13))
    axes[-1].set_xticklabels(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )
    fig.tight_layout(rect=[0, 0, 0.97, 1])
    plt.show()


def plot_precipitation(dataframe):
    """ Plot the total monthly precipitation for each city with subplots,
    comparing months across years.
    Inputs:
        dataframe (pd.DataFrame): DataFrame containing the daily data for all cities.
    Returns:
        None
    """
    df = dataframe.copy()
    df["date"] = df["date"].dt.tz_localize(None)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    monthly_total = (
        df.groupby(["city", "year", "month"])["precipitation_sum"]
        .sum()
        .reset_index()
    )

    cities = monthly_total["city"].unique()
    fig, axes = plt.subplots(len(cities), 1, figsize=(11, 8), sharex=True)

    for ax, city in zip(axes, cities):
        city_data = monthly_total[monthly_total["city"] == city]

        for year in city_data["year"].unique():
            year_data = city_data[city_data["year"] == year]
            ax.plot(
                year_data["month"],
                year_data["precipitation_sum"],
                marker="o",
                label=str(year)
            )

        ax.set_title(f"Monthly Precipitation in {city}")
        ax.set_ylabel("mm")
        ax.legend(title="Year", loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True)

    axes[-1].set_xlabel("Month")
    axes[-1].set_xticks(range(1, 13))
    axes[-1].set_xticklabels(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )
    fig.tight_layout(rect=[0, 0, 0.97, 1])
    plt.show()


def plot_wind(dataframe):
    """ Plot the maximum monthly wind speed for each city with subplots,
    comparing months across years.
    Inputs:
        dataframe (pd.DataFrame): DataFrame containing the daily data for all cities.
    Returns:
        None
    """
    df = dataframe.copy()
    df["date"] = df["date"].dt.tz_localize(None)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    monthly_max = (
        df.groupby(["city", "year", "month"])["wind_speed_10m_max"]
        .max()
        .reset_index()
    )

    cities = monthly_max["city"].unique()
    fig, axes = plt.subplots(len(cities), 1, figsize=(11, 8), sharex=True)

    for ax, city in zip(axes, cities):
        city_data = monthly_max[monthly_max["city"] == city]

        for year in city_data["year"].unique():
            year_data = city_data[city_data["year"] == year]
            ax.plot(
                year_data["month"],
                year_data["wind_speed_10m_max"],
                marker="o",
                label=str(year)
            )

        ax.set_title(f"Maximum Monthly Wind Speed in {city}")
        ax.set_ylabel("m/s")
        ax.legend(title="Year", loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True)

    axes[-1].set_xlabel("Month")
    axes[-1].set_xticks(range(1, 13))
    axes[-1].set_xticklabels(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )
    fig.tight_layout(rect=[0, 0, 0.97, 1])
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
        logging.info(f"Getting data for {city}")
        response = get_data_meteo_api(city)
        if response:
            dataframe = process_data(response, city)
            all_dataframes.append(dataframe)
        else:
            logging.warning(f"Skipping {city} due to no data.")

    # Combine all dataframes into one
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        logging.debug("DataFrames successfully combined.")  # Debug = más técnico
        logging.info("Showing a sample of the combined DataFrame:")
        logging.info("\n%s", combined_df.sample(5).to_string())

        # Plot the combined data
        plot_all(combined_df)
    else:
        logging.error("No dataframes were created.")


if __name__ == "__main__":
    main()
