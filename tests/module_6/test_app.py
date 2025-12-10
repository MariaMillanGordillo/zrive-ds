import pandas as pd
import pytest

from fastapi.testclient import TestClient
from src.module_6.app import app

from src.module_6.basket_model.feature_store import FeatureStore
from src.module_6.basket_model.basket_model import BasketModel
from src.module_6.exceptions import PredictionException


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_get_features():
    def _mock_get_features(self, _):
        return pd.Series([1, 2, 3, 4], dtype=float)

    return _mock_get_features


@pytest.fixture
def mock_predict_success():
    def _mock_predict(self, _):
        return [42.0]

    return _mock_predict


@pytest.fixture
def mock_predict_failure():
    def _mock_predict(self, _):
        raise PredictionException("fail")

    return _mock_predict


def test_status(client):
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_user_not_found(client):
    response = client.post("/predict", json={"user_id": "non_existent_user"})
    assert response.status_code == 404


def test_predict_valid_user(
    client, monkeypatch, mock_get_features, mock_predict_success
):
    monkeypatch.setattr(FeatureStore, "get_features", mock_get_features)
    monkeypatch.setattr(BasketModel, "predict", mock_predict_success)

    response = client.post("/predict", json={"user_id": "123"})
    assert response.status_code == 200
    assert response.json() == {"predicted_price": 42.0}


def test_predict_invalid_user_id(client):
    response = client.post("/predict", json={"user_id": ""})
    assert response.status_code == 422


def test_predict_model_failure(
    client, monkeypatch, mock_get_features, mock_predict_failure
):
    monkeypatch.setattr(FeatureStore, "get_features", mock_get_features)
    monkeypatch.setattr(BasketModel, "predict", mock_predict_failure)

    response = client.post("/predict", json={"user_id": "123"})
    assert response.status_code == 500
