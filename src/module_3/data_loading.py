import os
import boto3
import logging
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()
AWS_ACCESS_KEY = os.getenv("AWW_API_KEY")
AWS_SECRET_KEY = os.getenv("AWW_SECRET")


def download_s3_folder(bucket_name: str, prefix: str, local_dir: Path) -> None:
    """
    Download all files from an S3 bucket folder to a local directory.

    Args:
        bucket_name (str): Name of the S3 bucket.
        prefix (str): Folder path in the S3 bucket.
        local_dir (Path): Local directory to store downloaded files.
    """
    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )

    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if "Contents" not in response:
            logging.warning(f"No files found in {bucket_name}/{prefix}")
            return

        for obj in response["Contents"]:
            file_key = obj["Key"]
            if file_key.endswith("/"):
                continue  # skip folders

            relative_path = Path(*file_key.split("/")[1:])
            local_path = local_dir / relative_path
            local_path.parent.mkdir(parents=True, exist_ok=True)

            if local_path.exists():
                logging.info(f"Already exists: {local_path}, skipping download.")
            else:
                s3.download_file(bucket_name, file_key, str(local_path))
                logging.info(f"Downloaded {local_path}")

    except Exception as e:
        logging.error(f"Error downloading from S3: {e}")
        raise


def load_data(local_file: Path, download_if_missing: bool = True) -> pd.DataFrame:
    """
    Load data from a local CSV file, optionally downloading it from S3 if missing.

    Args:
        local_file (Path): Path to the local CSV file.
        download_if_missing (bool): Whether to download from S3 if the file is missing.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not local_file.exists() and download_if_missing:
        logging.info(f"{local_file} not found locally. Downloading from S3...")
        bucket_name = "zrive-ds-data"
        prefix = "groceries/box_builder_dataset/"
        download_s3_folder(bucket_name, prefix, local_file.parent.parent)  # download into 'data/'

    try:
        df = pd.read_csv(local_file)
        logging.info(f"Loaded data with shape {df.shape}")
        return df
    except FileNotFoundError:
        logging.error(f"File {local_file} not found.")
        raise
    except pd.errors.ParserError as e:
        logging.error(f"Error parsing CSV: {e}")
        raise


def filter_orders(df: pd.DataFrame, min_items: int = 5) -> pd.DataFrame:
    """
    Filter DataFrame to include only orders with outcome=1 and at least `min_items` unique variants.

    Args:
        df (pd.DataFrame): Input DataFrame.
        min_items (int): Minimum number of unique items per order.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    qualifying_orders = (
        df[df["outcome"] == 1]
        .groupby("order_id")["variant_id"]
        .nunique()
        .loc[lambda x: x >= min_items]
        .index
    )
    filtered_df = df[df["order_id"].isin(qualifying_orders)]
    logging.info(f"Filtered data to {filtered_df.shape[0]} rows across {filtered_df['order_id'].nunique()} orders")
    return filtered_df


if __name__ == "__main__":
    PROJECT_ROOT = Path().resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    DATA_DIR.mkdir(exist_ok=True)
    FILE_PATH = DATA_DIR / "box_builder_dataset" / "feature_frame.csv"

    df = load_data(FILE_PATH)
    df_filtered = filter_orders(df)
