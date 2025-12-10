from fastapi import APIRouter, Response
from prometheus_client import core, exposition

router = APIRouter(prefix="/metrics")


@router.get("/")
async def metrics():
    """Expose Prometheus metrics."""
    registry = core.REGISTRY
    metrics_data = exposition.generate_latest(registry)
    return Response(
        content=metrics_data,
        headers={"Content-Type": "text/plain; version=0.0.4; charset=utf-8"},
    )
