from fastapi import APIRoute

router = APIRoute(prefix="/status")

@router.get("/")
async def status():
    return {"status": "ok"}