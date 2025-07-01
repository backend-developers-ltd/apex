from fastapi import Request
from app.client.base_client import BaseVLLMClient
from app.client.compute_horde_client import ComputeHordeVLLMClient
import asyncio
from typing import Optional
import os

client_locks = {}

async def get_ready_client(request: Request, model_id: str) -> Optional[BaseVLLMClient]:
    """
    Get a healthy client for the specified model_id without creating a new one if none is available.
    """
    registry = request.app.state.model_registry
    clients = registry.get_clients(model_id)

    for client in clients:
        if await client.check_health():
            return client

    return None

async def get_or_create_client(request: Request, model_id: str, local: bool = False) -> BaseVLLMClient:
    """
    Get a healthy client for the specified model_id or create a new one if none is available.

    Args:
        request: The FastAPI request object
        model_id: The ID of the model to get or create a client for
        local: If True, only look for or create a local BaseVLLMClient
               If False, only look for or create a ComputeHordeVLLMClient
    """
    registry = request.app.state.model_registry
    clients = registry.get_clients(model_id)

    if local:
        # Only look for a healthy local client
        for client in clients:
            if isinstance(client, BaseVLLMClient) and not isinstance(client, ComputeHordeVLLMClient) and await client.check_health():
                return client
    else:
        # Only look for a healthy compute horde client
        # Also remove any unhealthy ComputeHordeVLLMClient from the registry
        clients_to_remove = []
        for client in clients:
            if isinstance(client, ComputeHordeVLLMClient):
                if await client.check_health():
                    return client
                else:
                    # Mark unhealthy client for removal
                    clients_to_remove.append(client)

        # Remove unhealthy clients from the registry
        for client in clients_to_remove:
            if client in registry.get_clients(model_id):
                registry.get_clients(model_id).remove(client)

    # Ensure only one job creation per model_id
    if model_id not in client_locks:
        client_locks[model_id] = asyncio.Lock()

    async with client_locks[model_id]:
        # Recheck if another coroutine created it
        clients = registry.get_clients(model_id)

        if local:
            # Recheck for local client
            for client in clients:
                if isinstance(client, BaseVLLMClient) and not isinstance(client, ComputeHordeVLLMClient) and await client.check_health():
                    return client

            # Create and register a new local client
            local_client = BaseVLLMClient(model_id)
            registry.add_client(model_id, local_client)
            return local_client
        else:
            # Recheck for compute horde client
            # Also remove any unhealthy ComputeHordeVLLMClient from the registry
            clients_to_remove = []
            for client in clients:
                if isinstance(client, ComputeHordeVLLMClient):
                    if await client.check_health():
                        return client
                    else:
                        # Mark unhealthy client for removal
                        clients_to_remove.append(client)

            # Remove unhealthy clients from the registry
            for client in clients_to_remove:
                if client in registry.get_clients(model_id):
                    registry.get_clients(model_id).remove(client)

            # Create and register a new compute horde client
            new_client = await ComputeHordeVLLMClient.create(model_id)
            registry.add_client(model_id, new_client)
            return new_client
