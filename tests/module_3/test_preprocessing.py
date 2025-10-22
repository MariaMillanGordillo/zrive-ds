import pandas as pd
from src.module_3.preprocessing import filter_orders, temporal_split_by_order


def test_filter_orders_filters_correctly():
    df = pd.DataFrame({
        "order_id": [1, 1, 2, 2, 2, 3],
        "variant_id": [10, 11, 20, 21, 22, 30],
        "outcome": [1, 1, 1, 1, 1, 0]
    })
    result = filter_orders(df, min_items=2)
    assert set(result["order_id"].unique()) == {1, 2}
    assert result["outcome"].eq(1).all()


def test_temporal_split_by_order_splits_cleanly():
    df = pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5],
        "order_date": pd.date_range("2024-01-01", periods=5),
        "feature": [10, 20, 30, 40, 50],
        "outcome": [0, 1, 0, 1, 0]
    })
    X_train, X_val, X_test, y_train, y_val, y_test = temporal_split_by_order(
        df,
        date_col="order_date",
        order_col="order_id",
        feature_cols=["feature"],
        target_col="outcome",
        train_size=0.6,
        val_size=0.2,
        test_size=0.2
    )

    # No order_id should overlap between splits
    all_orders = [
        set(X_train.index),
        set(X_val.index),
        set(X_test.index)
    ]
    assert len(set.union(*all_orders)) == len(df)
    assert set.intersection(*all_orders) == set()
