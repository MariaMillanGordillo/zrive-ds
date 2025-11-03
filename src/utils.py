import pandas as pd
import logging
from typing import Tuple, List, Optional

logging.basicConfig(
    level=logging.INFO,  # Info level for general information
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def temporal_split_by_order(
    df: pd.DataFrame,
    date_col: str,
    order_col: str = "order_id",
    feature_cols: Optional[List[str]] = None,
    target_col: str = "outcome",
    train_size: float = 0.7,
    val_size: float = 0.2,
    test_size: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
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
