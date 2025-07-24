from typing import Dict, List, Optional
from app.client.base_client import BaseVLLMClient
import asyncio

from app.client.compute_horde_client import ComputeHordeVLLMClient


class ModelRegistry:
    def __init__(self):
        self.clients: Dict[str, List[BaseVLLMClient]] = {}
    def get_clients(self, model_id: str) -> List[BaseVLLMClient]:
        """Get all clients for a model_id."""
        return self.clients.get(model_id, [])

    def add_client(self, model_id: str, client: BaseVLLMClient):
        """Add a client for a model_id."""
        self.clients.setdefault(model_id, []).append(client)

    async def initialize_local_client(self, model_id: str) -> BaseVLLMClient:
        """Initialize a local client for a model_id if it doesn't exist."""
        clients = self.get_clients(model_id)
        for client in clients:
            if isinstance(client, BaseVLLMClient) and not hasattr(client, 'compute_horde_job'):
                return client

        local_client = BaseVLLMClient(model_id)
        self.add_client(model_id, local_client)
        while not await local_client.check_health():
            await asyncio.sleep(1)
        return local_client

    async def initialize_compute_horde_client(self, model_id: str) -> Optional[BaseVLLMClient]:
        """
        Initialize a compute horde client for a model_id if it doesn't exist.

        Returns:
            The compute horde client if successful, None if initialization fails.
        """
        from loguru import logger

        # Check if we already have a compute horde client for this model
        clients = self.get_clients(model_id)
        for client in clients:
            if isinstance(client, ComputeHordeVLLMClient):
                logger.info(f"Found existing Compute Horde client for {model_id}")
                return client

        # Try to create a new compute horde client
        try:
            logger.info(f"Creating new Compute Horde client for {model_id}")
            compute_horde_client = await ComputeHordeVLLMClient.create(model_id)
            self.add_client(model_id, compute_horde_client)
            logger.info(f"Successfully created and registered Compute Horde client for {model_id}")
            return compute_horde_client
        except Exception as e:
            logger.error(f"Failed to initialize Compute Horde client for {model_id}: {e}")
            return None

    def get_all_model_ids(self) -> List[str]:
        """Get all model_ids that have clients."""
        return list(self.clients.keys())
