import httpx
import os
import time
from typing import Dict, Any, Optional
from fastapi import status
from loguru import logger

class BaseVLLMClient:
    """
    Client for interacting with a VLLM instance.

    This client connects to a VLLM instance (local or remote).
    """

    def __init__(self, model_id: str):
        """
        Initialize the base client.

        Args:
            model_id: The ID of the model to use.
        """
        self.model_id = model_id
        self._ready = False
        self.last_check = 0
        self.endpoint_url = os.environ["LOCAL_DOCKER_URL"]
        self.ssl_context = None
        self.needs_reconnect = False
        self.local = True  # Default to True for BaseVLLMClient

    async def check_health(self) -> bool:
        """
        Check if the VLLM instance is healthy.

        Returns:
            True if the instance is healthy, False otherwise.
        """
        if time.time() - self.last_check < 10 and not self.needs_reconnect:
            return self._ready
        try:
            async with httpx.AsyncClient(verify=self.ssl_context) as client:
                resp = await client.get(f"{self.endpoint_url}/health/model", timeout=5)
                self._ready = resp.status_code == 200
                if self._ready:
                    self.needs_reconnect = False
        except Exception as e:
            # logger.error(f"Health check failed for {self.model_id}: {str(e)}")
            self._ready = False
            if not self.local:
                self.needs_reconnect = True
        self.last_check = time.time()
        return self._ready

    async def proxy_request(self, endpoint: str, method: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Send a request to the specified endpoint.

        Args:
            endpoint: The endpoint to send the request to.
            method: The HTTP method to use (GET, POST, etc.).
            body: The request body (for POST requests).

        Returns:
            The response from the endpoint.
        """
        try:
            async with httpx.AsyncClient(verify=self.ssl_context) as client:
                if method.upper() == "GET":
                    resp = await client.get(f"{self.endpoint_url}{endpoint}", timeout=30)
                elif method.upper() == "POST":
                    resp = await client.post(f"{self.endpoint_url}{endpoint}", json=body, timeout=30)
                else:
                    return {"error": f"Unsupported HTTP method: {method}"}

                # Handle different response status codes
                if resp.status_code == status.HTTP_404_NOT_FOUND:
                    return {"error": f"Endpoint {endpoint} not found"}
                elif resp.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
                    # Model is still loading, this is expected sometimes
                    return {"error": "Service unavailable, model is still loading"}
                elif resp.status_code >= 500:
                    # Server error, mark client for reconnection if not local
                    logger.error(f"Server error for {self.model_id} at {endpoint}: {resp.status_code}")
                    if not self.local:
                        self.needs_reconnect = True
                    return {"error": f"Server error: {resp.status_code}"}

                # Try to parse JSON response
                try:
                    return resp.json()
                except Exception as e:
                    logger.error(f"Failed to parse JSON response: {str(e)}")
                    return {"error": f"Failed to parse response: {str(e)}"}

        except httpx.ConnectError as e:
            # Connection error, mark client for reconnection if not local
            logger.error(f"Connection error for {self.model_id} at {endpoint}: {str(e)}")
            if not self.local:
                self.needs_reconnect = True
            return {"error": f"Connection error: {str(e)}"}
        except httpx.RequestError as e:
            # Request error, mark client for reconnection if not local
            logger.error(f"Request error for {self.model_id} at {endpoint}: {str(e)}")
            if not self.local:
                self.needs_reconnect = True
            return {"error": f"Request error: {str(e)}"}
        except Exception as e:
            # Unexpected error
            logger.error(f"Unexpected error for {self.model_id} at {endpoint}: {str(e)}")
            return {"error": f"Unexpected error: {str(e)}"}
