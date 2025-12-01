from fastapi import FastAPI

from module_6.handlers.predict import predict
from module_6.handlers.status import status
from module_6.handlers.metrics import metrics

def create_app() -> FastAPI:
    app = FastAPI()
    app.include_router(status.router)
    app.include_router(metrics.router)
    app.include_router(predict.router)
    return app