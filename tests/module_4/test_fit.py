import json
import pytest
from unittest import mock
from src.module_4.fit import train_model, handler_fit


@pytest.fixture
def test_event():
    return {"model_parametrisation": {"n_estimators": 10, "max_depth": 2}}


def test_train_model_with_mock_joblib_dump(test_event):
    with mock.patch("joblib.dump") as mock_dump:
        result = train_model(test_event)
        mock_dump.assert_called_once()
        assert "model_name" in result
        assert "model_path" in result
        assert "validation_report" in result


def test_handler_fit_success(test_event):
    with mock.patch("joblib.dump") as mock_dump:
        response = handler_fit(test_event, None)
        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert "model_name" in body
        assert "model_path" in body
        mock_dump.assert_called_once()


def test_handler_fit_failure(monkeypatch, test_event):
    def raise_exception(event):
        raise ValueError("Simulated error")

    monkeypatch.setattr("src.module_4.fit.train_model", raise_exception)
    response = handler_fit(test_event, None)
    assert response["statusCode"] == 500
    body = json.loads(response["body"])
    assert "error" in body
    assert "Simulated error" in body["error"]
