import logging
import pandas as pd
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

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

def temporal_split_by_order(
    df, 
    date_col, 
    order_col="order_id", 
    feature_cols=None,
    target_col="outcome", 
    train_size=0.7, 
    val_size=0.2, 
    test_size=0.1
):
    """
    Splits a DataFrame into train, validation, and test sets based on order date,
    ensuring no items from the same order appear in multiple splits.
    """
    if not abs(train_size + val_size + test_size - 1.0) < 1e-6:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    # Order unique orders by date
    orders_sorted = (
        df[[order_col, date_col]]
        .drop_duplicates()
        .sort_values(date_col)
    )

    n_orders = len(orders_sorted)
    train_end = int(train_size * n_orders)
    val_end = int((train_size + val_size) * n_orders)

    # Assign orders to splits
    train_orders = orders_sorted.iloc[:train_end][order_col]
    val_orders = orders_sorted.iloc[train_end:val_end][order_col]
    test_orders = orders_sorted.iloc[val_end:][order_col]

    # Filter DataFrame based on order assignments
    train_df = df[df[order_col].isin(train_orders)]
    val_df = df[df[order_col].isin(val_orders)]
    test_df = df[df[order_col].isin(test_orders)]

    # If feature_cols is specified, select only those columns along with target_col
    if feature_cols is not None:
        train_df = train_df[feature_cols + [target_col]]
        val_df = val_df[feature_cols + [target_col]]
        test_df = test_df[feature_cols + [target_col]]

    # Divide into X and y
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    logging.info(f"Total unique orders: {n_orders}")
    logging.info(f"Train orders: {len(train_orders)} ({train_size*100:.1f}%)")
    logging.info(f"Val orders: {len(val_orders)} ({val_size*100:.1f}%)")
    logging.info(f"Test orders: {len(test_orders)} ({test_size*100:.1f}%)")
    logging.info(f"Train rows: {train_df.shape[0]}, Val rows: {val_df.shape[0]}, Test rows: {test_df.shape[0]}")

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    FILE_PATH = DATA_DIR / "box_builder_dataset" / "feature_frame.csv"
    PREPROCESSING_DIR = DATA_DIR / "preprocessing"

    logging.info(f"Creating preprocessing directory at {PREPROCESSING_DIR}")
    PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading data from {FILE_PATH}")
    df = pd.read_csv(FILE_PATH)

    df_filtered = filter_orders(df, min_items=5)

    feature_cols = ["product_type", "ordered_before", "abandoned_before", "active_snoozed", "set_as_regular", "global_popularity"]
    X_train, X_val, X_test, y_train, y_val, y_test = temporal_split_by_order(
        df_filtered,
        date_col="order_date",
        order_col="order_id",
        feature_cols=feature_cols,
        target_col="outcome"
    )

    pipeline = make_pipeline(
        FunctionTransformer(
            lambda X: X.assign(
                product_type=X['product_type'].map(X['product_type'].value_counts(normalize=True))
            )
        ),
        StandardScaler()
    )

    X_train_scaled = pd.DataFrame(
        pipeline.fit_transform(X_train),
        columns=feature_cols,
        index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        pipeline.transform(X_val),
        columns=feature_cols,
        index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        pipeline.transform(X_test),
        columns=feature_cols,
        index=X_test.index
    )

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
