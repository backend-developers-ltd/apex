from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from fastapi import status
from app.client.client_selector import get_ready_client
import os

router = APIRouter()

@router.get("/health")
async def health():
    return {"status": "ok"}

@router.get("/health/model")
async def health_model(request: Request):
    model_id = request.headers.get("X-Model-ID")
    if not model_id:
        # Default to the default model from environment
        model_id = os.environ["DEFAULT_MODEL"]

    client = await get_ready_client(request, model_id)
    if client:
        return {"status": "ok", "model_id": model_id}
    return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"status": "not ready"})
