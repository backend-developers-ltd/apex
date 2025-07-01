from typing import Dict, List
from app.client.base_client import BaseVLLMClient
import asyncio

from compute_horde_proxy.app.client.compute_horde_client import ComputeHordeVLLMClient


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
        return local_client

    async def initialize_compute_horde_client(self, model_id: str) -> BaseVLLMClient:
        """Initialize a compute horde client for a model_id if it doesn't exist."""
        clients = self.get_clients(model_id)
        for client in clients:
            if isinstance(client, ComputeHordeVLLMClient):
                return client

        compute_horde_client = await ComputeHordeVLLMClient.create(model_id)
        self.add_client(model_id, compute_horde_client)
        return compute_horde_client

    def get_all_model_ids(self) -> List[str]:
        """Get all model_ids that have clients."""
        return list(self.clients.keys())
