from fastapi import APIRouter, Request, Response, status
from app.client.client_selector import get_or_create_client
from app.client.base_client import BaseVLLMClient
from app.client.compute_horde_client import ComputeHordeVLLMClient
from app.verification.verification import verify_with_compute_horde_client
from starlette.responses import JSONResponse
from loguru import logger
import httpx
import os
import asyncio
import uuid
import json
import pathlib
import time

router = APIRouter()


@router.api_route("/{path:path}", methods=["GET", "POST"])
async def proxy_endpoint(request: Request, path: str):
    """
    Generic proxy endpoint that forwards any request to the appropriate client.

    This endpoint captures all requests that don't match other defined endpoints
    and forwards them to the appropriate client based on the model ID in the headers.

    Note: Health endpoints (/health and /health/model) are handled by the health router,
    which is included before this router in app.py.
    """
    # Skip health endpoints as they are handled separately
    if path.startswith("health"):
        return Response(status_code=404)

    # Get the model ID from the headers
    model_id = request.headers.get("X-Model-ID")
    if not model_id:
        # Default to the default model from environment
        model_id = os.environ["DEFAULT_MODEL"]

    # Get or create a client for the model
    client = await get_or_create_client(request, model_id, local=True)

    # Get the request method and body
    method = request.method
    body = await request.json() if method in ["POST"] and await request.body() else None

    # Check if parallel verification is enabled
    parallel_verification = True

    # Record ID for tracking the same request across different clients

    # Use the provided record_id or generate a new one
    record_id = str(uuid.uuid4())

    # If parallel verification is enabled and client is a local client,
    # also send the request to a compute horde client for verification
    if parallel_verification and isinstance(client, BaseVLLMClient) and not isinstance(client, ComputeHordeVLLMClient):
        # Send request to compute horde client in parallel
        asyncio.create_task(
            verify_with_compute_horde_client(request, path, method, body, model_id, record_id)
        )

    # Forward the request to the client
    response = await client.proxy_request(f"/{path}", method, body)

    # Save the request and response as JSON if client is a local client
    if isinstance(client, BaseVLLMClient) and not isinstance(client, ComputeHordeVLLMClient):
        # Create directories if they don't exist
        log_dir = pathlib.Path("request_logs/local")
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create a JSON file with the record data
        record_data = {
            "id": record_id,
            "timestamp": time.time(),
            "model_id": model_id,
            "endpoint": path,
            "method": method,
            "request_body": body,
            "response": response,
            "client_type": "local"
        }

        # Save the JSON file
        file_path = log_dir / f"{record_id}.json"
        with open(file_path, 'w') as f:
            json.dump(record_data, f, indent=2)

        logger.debug(f"BaseVLLMClient response {file_path}")

    # Record the request and response if client is ComputeHordeVLLMClient
    if isinstance(client, ComputeHordeVLLMClient):
        # Get miner ID if available
        miner_id = getattr(client, "miner_id", None)

        # Record the request and response
        await request.app.state.request_log.add_record(
            model_id=model_id,
            endpoint=path,
            method=method,
            request_body=body,
            response=response,
            client_type="compute_horde",
            miner_id=miner_id,
            record_id=record_id
        )

    # Check if the response contains an error
    if "error" in response:
        error_message = response["error"]

        # Handle connection errors that require client reconnection
        if client.needs_reconnect:
            logger.warning(f"Client for {model_id} needs reconnection. Removing from registry.")

            # Remove the client from the registry
            registry = request.app.state.model_registry
            clients = registry.get_clients(model_id)
            if client in clients:
                clients.remove(client)

            # Try to get or create a new client
            new_client = await get_or_create_client(request, model_id)

            # Try the request again with the new client
            if new_client and new_client != client:
                logger.info(f"Retrying request with new client for {model_id}")
                response = await new_client.proxy_request(f"/{path}", method, body)

                # If still error, return it
                if "error" in response:
                    return JSONResponse(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        content={"error": f"Failed to process request after reconnection: {response['error']}"}
                    )
                return response
            else:
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content={"error": f"Failed to create new client for {model_id}"}
                )

        # Handle specific error types
        if "Service unavailable" in error_message:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"error": error_message}
            )
        elif "not found" in error_message:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"error": error_message}
            )
        elif "Server error" in error_message or "Connection error" in error_message or "Request error" in error_message:
            return JSONResponse(
                status_code=status.HTTP_502_BAD_GATEWAY,
                content={"error": error_message}
            )
        else:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": error_message}
            )

    return response
