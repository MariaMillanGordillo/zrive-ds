import pytest
import pandas as pd
from unittest.mock import patch
from src.module_3.train_model import (
    train_logistic_regression,
    plot_confusion_matrix,
    plot_roc_pr,
)


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


@patch("src.module_3.train_model.plt.show")
def test_plot_confusion_matrix_creates_figure(mock_show):
    import numpy as np
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0.1, 0.8, 0.6, 0.2])
    fig, ax = plot_confusion_matrix(y_true, y_pred, threshold=0.5)
    assert fig is not None
    assert ax is not None
    mock_show.assert_called_once()


@patch("src.module_3.train_model.plt.show")
def test_plot_roc_pr_creates_figure(mock_show):
    import numpy as np
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0.1, 0.9, 0.6, 0.2])
    fig, ax = plot_roc_pr(y_true, y_pred)
    assert fig is not None
    assert len(ax) == 2
    mock_show.assert_called_once()
