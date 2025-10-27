import json
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from src.module_4.fit import product_type_transform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_pipeline_and_model(model_path=None):
    """
    Loads saved pipeline and model dictionary.
    If no path is provided, loads most recent from 'models/' directory.
    """
    if model_path is not None:
        saved = joblib.load(model_path)
        return saved["pipeline"], saved["model"]
    
    # Find latest model file
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    model_dir = PROJECT_ROOT / "models"
    model_files = sorted(model_dir.glob("push_*.pkl"), reverse=True)
    if not model_files:
        raise FileNotFoundError("No model files found in 'models/' directory.")
    latest_model_path = model_files[0]
    logging.info(f"Loading pipeline and model from {latest_model_path}")
    saved = joblib.load(latest_model_path)
    return saved["pipeline"], saved["model"]

def predict_with_pipeline(pipeline, model, data_df):
    """
    Receives pipeline, trained model, and user data. Returns prediction dict.
    """
    data_transformed = pd.DataFrame(
        pipeline.transform(data_df), 
        columns=data_df.columns, 
        index=data_df.index
    )
    preds = model.predict(data_transformed)
    return dict(zip(data_df.index, preds))

def handler_predict(event, _):
    """
    Handler for prediction requests. Expects 'users' key with JSON string of user features.
    Returns predictions in JSON format.
    """
    try:
        users_dict = json.loads(event["users"])
        data_to_predict = pd.DataFrame.from_dict(users_dict, orient='index')

        # Cargar pipeline + modelo
        pipeline, model = load_pipeline_and_model()
        predictions = predict_with_pipeline(pipeline, model, data_to_predict)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "prediction": predictions
            })
        }
    except Exception as e:
        logging.error(f"Error en predict: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }

if __name__ == "__main__":
    test_event = {
        "users": json.dumps({
            "user_1": {
                "product_type": 0.123,
                "ordered_before": 1,
                "abandoned_before": 0,
                "active_snoozed": 0,
                "set_as_regular": 1,
                "global_popularity": 0.5
            },
            "user_2": {
                "product_type": 0.231,
                "ordered_before": 0,
                "abandoned_before": 1,
                "active_snoozed": 1,
                "set_as_regular": 0,
                "global_popularity": 0.3
            }
        })
    }
    response = handler_predict(test_event, None)
    print(response)
