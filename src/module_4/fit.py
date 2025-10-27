import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.module_3.data_loading import load_data
from src.module_3.preprocessing import filter_orders, temporal_split_by_order

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def product_type_transform(X):
    return X.assign(
        product_type=(
            X["product_type"].map(X["product_type"].value_counts(normalize=True))
        )
    )


def train_model(event):
    """
    Train Gradient Boosting model using SMOTE and save pipeline + model to disk.
    """
    model_params = event.get("model_parametrisation", {})
    logging.info(f"Training with parameters: {model_params}")

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    DATA_PATH = PROJECT_ROOT / "data" / "box_builder_dataset" / "feature_frame.csv"
    df = load_data(DATA_PATH)

    df_filtered = filter_orders(df, min_items=5)
    feature_cols = [
        "product_type",
        "ordered_before",
        "abandoned_before",
        "active_snoozed",
        "set_as_regular",
        "global_popularity",
    ]

    X_train, X_val, X_test, y_train, y_val, y_test = temporal_split_by_order(
        df_filtered,
        date_col="order_date",
        order_col="order_id",
        feature_cols=feature_cols,
        target_col="outcome",
    )

    pipeline = make_pipeline(
        FunctionTransformer(product_type_transform), StandardScaler()
    )

    X_train_scaled = pd.DataFrame(
        pipeline.fit_transform(X_train), columns=feature_cols, index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        pipeline.transform(X_val), columns=feature_cols, index=X_val.index
    )

    # Resample training data using SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    logging.info(f"After SMOTE: {X_train_res.shape[0]} samples")

    model = GradientBoostingClassifier(**model_params)
    model.fit(X_train_res, y_train_res)
    logging.info("Model training completed.")

    y_pred_val = model.predict(X_val_scaled)
    report = classification_report(y_val, y_pred_val)
    logging.info("Validation Report:\n" + report)

    date_str = datetime.now().strftime("%Y_%m_%d")
    model_name = f"push_{date_str}.pkl"
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / model_name

    # Save as dict pipeline + model
    to_save = {"pipeline": pipeline, "model": model}
    joblib.dump(to_save, model_path)
    logging.info(f"Pipeline and model saved in {model_path}")

    return {
        "model_name": model_name,
        "model_path": str(model_path),
        "validation_report": report,
    }


def handler_fit(event, _):
    """
    API-compatible handler: trains the model and returns metadata.
    """
    try:
        result = train_model(event)
        return {
            "statusCode": 200,
            "body": json.dumps(
                {"model_name": result["model_name"], "model_path": result["model_path"]}
            ),
        }
    except Exception as e:
        logging.error(f"Training failed: {e}")
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}


if __name__ == "__main__":
    test_event = {"model_parametrisation": {"n_estimators": 200, "max_depth": 3}}
    response = handler_fit(test_event, None)
    print(response)
