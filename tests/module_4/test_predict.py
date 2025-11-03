import json
from unittest import mock
from src.module_4.predict import handler_predict

# Test event
test_event = {
    "users": json.dumps({
        "user_1": {
            "product_type": 0.123,
            "ordered_before": 1,
            "abandoned_before": 0,
            "active_snoozed": 0,
            "set_as_regular": 1,
            "global_popularity": 0.5,
        },
        "user_2": {
            "product_type": 0.231,
            "ordered_before": 0,
            "abandoned_before": 1,
            "active_snoozed": 1,
            "set_as_regular": 0,
            "global_popularity": 0.3,
        },
    })
}


def test_handler_predict_success():
    # Mock pipeline and model
    mock_pipeline = mock.MagicMock()
    mock_pipeline.transform.return_value = "transformed_data"
    mock_model = mock.MagicMock()
    mock_model.predict.return_value = [1, 0]

    # Mock load_pipeline_and_model to return the mocks
    with mock.patch("src.module_4.predict.load_pipeline_and_model",
                    return_value=(mock_pipeline, mock_model)):
        response = handler_predict(test_event, None)
        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert "prediction" in body
        assert body["prediction"] == {"user_1": 1, "user_2": 0}


def test_handler_predict_failure():
    # Force an exception in load_pipeline_and_model
    with mock.patch("src.module_4.predict.load_pipeline_and_model",
                    side_effect=Exception("Simulated error")):
        response = handler_predict(test_event, None)
        assert response["statusCode"] == 500
        body = json.loads(response["body"])
        assert "error" in body
        assert "Simulated error" in body["error"]



if __name__ == "__main__":
    test_event = {
        "users": json.dumps(
            {
                "user_1": {
                    "product_type": 0.123,
                    "ordered_before": 1,
                    "abandoned_before": 0,
                    "active_snoozed": 0,
                    "set_as_regular": 1,
                    "global_popularity": 0.5,
                },
                "user_2": {
                    "product_type": 0.231,
                    "ordered_before": 0,
                    "abandoned_before": 1,
                    "active_snoozed": 1,
                    "set_as_regular": 0,
                    "global_popularity": 0.3,
                },
            }
        )
    }
    response = handler_predict(test_event, None)
    print(response)
