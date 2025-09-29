import time
import logging
import requests
import pandas as pd
import requests_cache
import openmeteo_requests
from typing import Optional
import matplotlib.pyplot as plt
from retry_requests import retry

# Logging configuration
logging.basicConfig(
    level=logging.INFO, # Info level for general information
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

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
        params: Optional[dict[str, str]] = None
) -> Optional[dict]:
    """
    Call the weather API for a specific city.
    Args:
        city (str): Name of the city to request data for. 
            Example: "Madrid", "London".
        url (str, optional): API endpoint URL. Defaults to the global constant API_URL. 
            Normally you should not override this unless you are testing a different API base.
        params (dict[str, str], optional): Additional query parameters to include in the request. 
            Common keys include:
                - "latitude": Latitude of the city (float).
                - "longitude": Longitude of the city (float).
                - "start_date": Start date for the data in "YYYY-MM-DD" format (str).
                - "end_date": End date for the data in "YYYY-MM-DD" format (str).
                - "daily": Comma-separated list of weather variables to retrieve (str).
                - "timezone": Timezone for the data (str).
            Defaults to None, meaning only the required parameters will be sent.
    Returns:
        dict | None: Parsed JSON response from the API as a dictionary, 
        or None if the request failed after all retries.
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

    try:
        responses = openmeteo.weather_api(url, params=params)
        if not responses:
            logging.warning(f"No data returned for {city}.")
            return None
        return responses

    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error for {city}: {e}")
        return None
    except requests.exceptions.ConnectionError as e:
        logging.error(f"Connection error for {city}: {e}")
        return None
    except requests.exceptions.Timeout as e:
        logging.error(f"Timeout for {city}: {e}")
        return None
    except ValueError as e:
        logging.error(f"JSON decode error for {city}: {e}")
        return None
    except Exception as e:
        logging.exception(f"Unexpected error for {city}: {e}")
        return None



def validate_response(response):
    """
    Validate the API response to ensure it contains the expected data.
    Raises an error if the response is invalid or missing expected data.
    Args:
        response (object): The API response object to validate.
    Raises:
        ValueError: If the response contains an error or is missing expected data.
    Returns:
        None
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
    """
    Get and validate weather data for a specific city from the Meteo API.
    Args:
        city (str): Name of the city to request data for.
        Example: "Madrid", "London".
    Returns:
        object | None: Validated API response object, or None if no data was returned.
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


def process_data(response, city, variables=VARIABLES):
    """
    Process the API response and convert it into a pandas DataFrame.
    Args:
        response (object): Validated API response object.
        city (str): Name of the city the data corresponds to.
        Example: "Madrid", "London".
        variables (list, optional): List of variable names to extract from the response. Defaults to the global constant VARIABLES.
    Returns:
        pd.DataFrame: DataFrame containing the processed daily weather data.
    """
    # Process daily data
    daily = response.Daily()
    daily_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left",
        )
    }
    for i, var_name in enumerate(variables):
        daily_data[var_name] = daily.Variables(i).ValuesAsNumpy()

    daily_data["city"] = city
    daily_dataframe = pd.DataFrame(data=daily_data)

    return daily_dataframe


# Data plotting functions


def plot_temperature(df):
    """
    Plot the average monthly temperature for each city with subplots,
    comparing months across years.
    Args:
        df (pd.DataFrame): DataFrame containing the daily data for all cities.
    Returns:
        None
    """
    data = df.copy()
    data["date"] = data["date"].dt.tz_localize(None)
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month

    monthly_avg = (
        data.groupby(["city", "year", "month"])["temperature_2m_mean"]
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


def plot_precipitation(df):
    """
    Plot the total monthly precipitation for each city with subplots,
    comparing months across years.
    Args:
        df (pd.DataFrame): DataFrame containing the daily data for all cities.
    Returns:
        None
    """
    data = df.copy()
    data["date"] = data["date"].dt.tz_localize(None)
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month

    monthly_total = (
        data.groupby(["city", "year", "month"])["precipitation_sum"]
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


def plot_wind(df):
    """
    Plot the maximum monthly wind speed for each city with subplots,
    comparing months across years.
    Args:
        df (pd.DataFrame): DataFrame containing the daily data for all cities.
    Returns:
        None
    """
    data = df.copy()
    data["date"] = data["date"].dt.tz_localize(None)
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month

    monthly_max = (
        data.groupby(["city", "year", "month"])["wind_speed_10m_max"]
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


def plot_per_city(df):
    """
    Plot all weather variables for each city in a single figure with subplots.
    Args:
        df (pd.DataFrame): DataFrame containing the daily data for all cities.
    Returns:
        None
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 7))
    cities = df["city"].unique()

    for city in cities:
        city_data = df[df["city"] == city].copy()
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


def plot_all(df):
    """
    Plot all weather variables using the defined plotting functions.
    Args:
        df (pd.DataFrame): DataFrame containing the daily data for all cities.
    Returns:
        None
    """
    plot_temperature(df)
    plot_precipitation(df)
    plot_wind(df)
    plot_per_city(df)


def main():
    """
    Main function to get, process, and plot weather data for multiple cities.
    """
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
