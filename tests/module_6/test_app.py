import pandas as pd

from fastapi.testclient import TestClient
from module_6.app import app

from src.module_6.basket_model.feature_store import FeatureStore
from src.module_6.basket_model.basket_model import BasketModel
from src.module_6.exceptions import PredictionException

client = TestClient(app)


def test_status():
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_user_not_found():
    response = client.post("/predict", json={"user_id": "non_existent_user"})
    assert response.status_code == 404


def test_predict_valid_user(monkeypatch):

    # Mock Feature Store
    def mock_get_features(self, _):
        return pd.Series([1, 2, 3, 4], dtype=float)

    # Mock Prediction
    def mock_predict(self, _):
        return [42.0]

    monkeypatch.setattr(FeatureStore, "get_features", mock_get_features)
    monkeypatch.setattr(BasketModel, "predict", mock_predict)

    response = client.post("/predict", json={"user_id": "123"})
    assert response.status_code == 200
    assert response.json() == {"predicted_price": 42.0}


def test_predict_invalid_user_id():
    response = client.post("/predict", json={"user_id": ""})
    assert response.status_code == 422


def test_predict_model_failure(monkeypatch):
    def mock_get_features(self, _):
        return pd.Series([1, 2, 3, 4], dtype=float)

    def mock_predict(self, _):
        raise PredictionException("fail")

    monkeypatch.setattr(FeatureStore, "get_features", mock_get_features)
    monkeypatch.setattr(BasketModel, "predict", mock_predict)

    response = client.post("/predict", json={"user_id": "123"})
    assert response.status_code == 500
