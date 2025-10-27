import json
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_model(model_path=None):
    """
    Loads a trained model from the specified path.
    If no path is provided, loads the most recent model from 'models/' directory.
    """
    if model_path is not None:
        return joblib.load(model_path)
    
    # Find the most recent model in 'models/' directory
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    model_dir = PROJECT_ROOT / "models"
    model_files = sorted(model_dir.glob("push_*.pkl"), reverse=True)
    if not model_files:
        raise FileNotFoundError("No model files found in 'models/' directory.")
    latest_model_path = model_files[0]
    logging.info(f"Loading model from {latest_model_path}")
    return joblib.load(latest_model_path)

def predict_with_model(model, data_df):
    """
    Receives a trained model and a DataFrame of user features,
    returns predictions.
    """
    preds = model.predict(data_df)
    return preds

def handler_predict(event, _):
    """
    Handler for prediction requests.
    Expects 'users' key in event with JSON string of user features.
    Returns predictions in JSON format.
    """
    try:
        users_dict = json.loads(event["users"])
        data_to_predict = pd.DataFrame.from_dict(users_dict, orient='index')

        model = load_model()

        # Preprocesamiento necesario (igual que para train). Aquí debe aplicarse misma lógica de pipeline que en train (ejemplo simplificado).
        # Nota: Si tienes pipeline guardado, cargar y aplicar aquí.
        # En este ejemplo transformamos product_type a proporción y escalamos.
        if 'product_type' in data_to_predict.columns:
            value_counts = data_to_predict['product_type'].value_counts(normalize=True)
            data_to_predict['product_type'] = data_to_predict['product_type'].map(value_counts)

        # Escalado simple ejemplo (en producción usar pipeline guardado)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data_scaled = pd.DataFrame(scaler.fit_transform(data_to_predict), columns=data_to_predict.columns, index=data_to_predict.index)

        # Predecir
        predictions = predict_with_model(model, data_scaled)

        # Formatear output
        prediction_dict = dict(zip(data_to_predict.index, predictions))

        return {
            "statusCode": 200,
            "body": json.dumps({
                "prediction": prediction_dict
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
                "product_type": "A",
                "ordered_before": 1,
                "abandoned_before": 0,
                "active_snoozed": 0,
                "set_as_regular": 1,
                "global_popularity": 0.5
            },
            "user_2": {
                "product_type": "B",
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
