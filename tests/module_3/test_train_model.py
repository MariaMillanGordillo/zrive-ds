import pandas as pd
import pytest

from src.module_3.train_model import train_logistic_regression


@pytest.fixture
def dummy_data():
    X_train = pd.DataFrame({"x1": [0, 1, 2, 3]})
    y_train = pd.Series([0, 0, 1, 1])
    X_val = pd.DataFrame({"x1": [1, 2, 3, 4]})
    y_val = pd.Series([0, 1, 1, 1])
    return X_train, y_train, X_val, y_val


def test_train_logistic_regression_returns_best_model(dummy_data):
    X_train, y_train, X_val, y_val = dummy_data
    model, y_pred, results = train_logistic_regression(X_train, y_train, X_val, y_val)
    assert "C" in results.columns
    assert hasattr(model, "predict_proba")
    assert len(y_pred) == len(y_val)
