import time
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from basket_model.basket_model import BasketModel
from basket_model.feature_store import FeatureStore
from exceptions import (
    UserNotFoundException,
    PredictionException
)

router = APIRouter(prefix='/predict')
model = BasketModel()
feature_store = FeatureStore()


class PredictRequest(BaseModel):  # Validate non-empty user_id
    user_id: str = Field(...)

    @field_validator('user_id')
    @classmethod
    def user_id_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('user_id must not be empty')
        return v


@router.post("/")
async def predict(req: PredictRequest):
    user_id = req.user_id
    logging.info(f"Received prediction request for user_id: {user_id}")
    try:
        features = feature_store.get_features(user_id).values.reshape(1, -1)
    except UserNotFoundException:
        logging.error(f"User {user_id} not found in feature store")
        raise HTTPException(status_code=404, detail="User not found")
    try:
        model_start = time.time()
        prediction = model.predict(features)[0]
        model_latency = time.time() - model_start

        logging.info(
            f"[MODEL] user_id={user_id} "
            f"num_features={features.shape[1]} "
            f"features_sample={features[0][:5].tolist()} "
            f"prediction={prediction} "
            f"model_latency={model_latency:.4f}s"
        )
    except PredictionException:
        logging.error(f"Prediction failed for user {user_id}")
        raise HTTPException(status_code=500, detail="Prediction failed")
    return {"predicted_price": float(prediction.mean())}