import time
import uvicorn
import logging
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from starlette.responses import JSONResponse

from module_6.basket_model.basket_model import BasketModel
from module_6. basket_model.feature_store import FeatureStore
from module_6.basket_model.utils.exceptions import UserNotFoundException, PredictionException

logging.basicConfig(
    filename="service_metrics.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI()

model = BasketModel()
feature_store = FeatureStore()

class PredictRequest(BaseModel):
    user_id: str

@app.middleware("http")
async def log_metrics(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        latency = time.time() - start_time
        logging.info(f"Request {request.method} {request.url.path} completed in {latency:.3f}s with status {response.status_code}")
        return response
    except Exception as ex:
        latency = time.time() - start_time
        logging.error(f"Request {request.method} {request.url.path} failed in {latency:.3f}s: {ex}")
        raise ex

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
        prediction = model.predict(features)[0]
        logging.info(f"Prediction for user {user_id}: {prediction}")
    except PredictionException:
        logging.error(f"Prediction failed for user {user_id}")
        raise HTTPException(status_code=500, detail="Prediction failed")
    return {"predicted_price": float(prediction)}

# Execute with: poetry run python src/module_6/app.py
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
