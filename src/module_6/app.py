import os
import time
import uvicorn
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator, Field

from module_6.basket_model.basket_model import BasketModel
from module_6.basket_model.feature_store import FeatureStore
from module_6.exceptions import (
    UserNotFoundException,
    PredictionException
)

os.makedirs("logs", exist_ok=True)
log_path = os.path.join("logs", "service_metrics.txt")

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Servicio iniciado: log de prueba")
app = FastAPI()
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


@app.middleware("http")
async def log_metrics(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        latency = time.time() - start_time
        logging.info(f"""Request {request.method} {request.url.path}
                     completed in {latency:.3f}s with status {response.status_code}""")
        return response
    except Exception as ex:
        latency = time.time() - start_time
        logging.error(f"""Request {request.method} {request.url.path}
                      failed in {latency:.3f}s: {ex}""")
        raise ex


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"""Unhandled error for request
                  {request.method} {request.url.path}: {exc}""")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


@app.get("/status")
async def status():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {"message": "API is running"}


@app.post("/predict")
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
    return {"predicted_price": float(prediction)}


# Execute with: poetry run python src/module_6/app.py
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
