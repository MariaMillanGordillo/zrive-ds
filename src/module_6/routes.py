from fastapi import FastAPI

from src.module_6.handlers.predict import router as predict_router
from src.module_6.handlers.status import router as status_router
from src.module_6.handlers.metrics import router as metrics_router


def create_app() -> FastAPI:
    app = FastAPI()
    app.include_router(status_router)
    app.include_router(metrics_router)
    app.include_router(predict_router)
    return app
