import os
import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from src.module_4.fit import product_type_transform
from typing import Optional, Tuple, Any, Union

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_pipeline_and_model(model_path: Optional[Union[str, Path]] = None) -> Tuple[Any, Any]:
    """
    Loads saved pipeline and model dictionary.
    If no path is provided, loads most recent from 'models/' directory.

    Args:
        model_path: Optional path to a specific .pkl file (str or Path).

    Returns:
        Tuple containing (pipeline, model).
    """
    if model_path is not None:
        model_dir = Path("models/")
        pkls = list(model_dir.glob("*.pkl"))
        if not pkls:
            raise FileNotFoundError("No .pkl files found in models/ directory.")
        model_path = max(pkls, key=os.path.getmtime)
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

def predict_with_pipeline(pipeline: Any, model: Any, data_df: pd.DataFrame) -> dict:
    """
    Receives pipeline, trained model, and user data. Returns prediction dict.
    """
    data_transformed = pd.DataFrame(
        pipeline.transform(data_df), columns=data_df.columns, index=data_df.index
    )
    preds = model.predict(data_transformed)
    return dict(zip(data_df.index, preds))


def handler_predict(event: dict, _: Any) -> dict:
    """
    Handler for prediction requests.
    Expects 'users' key with JSON string of user features.
    Returns predictions in JSON format.
    """
    try:
        users_dict = json.loads(event["users"])
        data_to_predict = pd.DataFrame.from_dict(users_dict, orient="index")

        pipeline, model = load_pipeline_and_model()
        predictions = predict_with_pipeline(pipeline, model, data_to_predict)

        return {"statusCode": 200, "body": json.dumps({"prediction": predictions})}
    except Exception as e:
        logging.error(f"Error en predict: {e}")
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}

