import logging
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


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


def prepare_features(df: pd.DataFrame):
    """
    Prepare features and target, encode categorical variables, split the dataset, and scale features.

    Args:
        df (pd.DataFrame): Filtered DataFrame.

    Returns:
        tuple: Scaled train, validation, test sets and targets.
    """
    # Select features and target
    X = df[[
        "variant_id",
        "product_type",
        "ordered_before",
        "abandoned_before",
        "active_snoozed",
        "set_as_regular",
        "global_popularity"
    ]]
    y = df["outcome"]

    # Frequency encode 'product_type'
    X_encoded = X.copy()
    freq_map = X_encoded['product_type'].value_counts(normalize=True)
    X_encoded['product_type'] = X_encoded['product_type'].map(freq_map)

    # Split into train (70%), validation (20%), test (10%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_encoded, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val, y_val, test_size=0.3333, random_state=42
    )

    logging.info(f"Training set size: {X_train.shape[0]} rows")
    logging.info(f"Validation set size: {X_val.shape[0]} rows")
    logging.info(f"Test set size: {X_test.shape[0]} rows")

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    FILE_PATH = DATA_DIR / "box_builder_dataset" / "feature_frame.csv"
    PREPROCESSING_DIR = DATA_DIR / "preprocessing"

    logging.info(f"Creating preprocessing directory at {PREPROCESSING_DIR}")
    PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)

    # Load the CSV
    logging.info(f"Loading data from {FILE_PATH}")
    df = pd.read_csv(FILE_PATH)

    # Filter the dataset
    df_filtered = filter_orders(df)

    # Prepare features and get scaled datasets
    X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test = prepare_features(df_filtered)

    # Save datasets in a loop
    datasets = {
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test
    }

    for name, data in datasets.items():
        save_path = PREPROCESSING_DIR / f"{name}.pkl"
        pd.to_pickle(data, save_path)
        logging.info(f"Saved {name} to {save_path}")

    logging.info("Preprocessed and scaled datasets successfully saved.")

