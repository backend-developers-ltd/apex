from fastapi import FastAPI
from app.routers import health, proxy
from app.model_registry import ModelRegistry
from app.client.base_client import BaseVLLMClient
from app.client.compute_horde_client import ComputeHordeVLLMClient
from app.verification.request_log import RequestLog
from app.verification.verification import run_trusted_verification
import os
import asyncio
import sys
from loguru import logger

# Required environment variables
REQUIRED_ENV_VARS = [
    "DEFAULT_MODEL",
    "LOCAL_DOCKER_URL",
    "COMPUTE_HORDE_FACILITATOR_URL",
    "DEFAULT_DOCKER_IMAGE",
    "COMPUTE_HORDE_DOWNLOAD_TIME_LIMIT_SEC",
    "COMPUTE_HORDE_EXECUTION_TIME_LIMIT_SEC",
    "COMPUTE_HORDE_STREAMING_START_TIME_LIMIT_SEC",
    "COMPUTE_HORDE_UPLOAD_TIME_LIMIT_SEC",
    # Verification-related variables (with defaults in code) # TODO
    # "REQUEST_LOG_MAX_SIZE",
    # "REQUEST_LOG_SAMPLE_RATE",
    # "ENABLE_VERIFICATION",
    # "ENABLE_PARALLEL_VERIFICATION",
    # "TRUSTED_VERIFICATION_INTERVAL",
]

# Validate required environment variables
missing_vars = [var for var in REQUIRED_ENV_VARS if not os.environ.get(var)]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    sys.exit(1)

app = FastAPI()
app.include_router(health.router)  # Include health router first for health checks
app.include_router(proxy.router)  # Include proxy router to handle all other requests

@app.on_event("startup")
async def startup_event():
    try:
        # Initialize model registry
        app.state.model_registry = ModelRegistry()

        # Initialize request log
        max_size = int(os.environ.get("REQUEST_LOG_MAX_SIZE", "100"))
        sample_rate = float(os.environ.get("REQUEST_LOG_SAMPLE_RATE", "0.1"))
        app.state.request_log = RequestLog(max_size=max_size, sample_rate=sample_rate)

        # Initialize local client for default model from .env
        default_model = os.environ["DEFAULT_MODEL"]

        logger.info(f"Initializing local client for default model: {default_model}")
        local_client = await app.state.model_registry.initialize_local_client(default_model)

        # Check if local client is healthy
        is_healthy = await local_client.check_health()
        logger.info(f"Local client for {default_model} is {'healthy' if is_healthy else 'unhealthy'}")

        # Initialize Compute Horde client for default model
        try:
            logger.info(f"Initializing Compute Horde client for default model: {default_model}")
            compute_horde_client = await app.state.model_registry.initialize_compute_horde_client(default_model)
            if compute_horde_client is not None:
                is_healthy = await compute_horde_client.check_health()
                logger.info(f"Compute Horde client for {default_model} is {'healthy' if is_healthy else 'unhealthy'}")
            else:
                logger.warning(f"Failed to initialize Compute Horde client for {default_model}")
                logger.warning("Continuing without Compute Horde client")
        except Exception as e:
            logger.error(f"Unexpected error during Compute Horde client initialization for {default_model}: {e}")
            logger.warning("Continuing without Compute Horde client")

        # Start background task to periodically check health of all clients
        asyncio.create_task(check_clients_health())

        # Start background task to periodically run trusted verification
        # TODO enable for trusted verification
        # asyncio.create_task(run_trusted_verification_task())
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        # Don't exit the application, let it continue running with whatever was initialized successfully

async def check_clients_health():
    """Background task to periodically check health of all clients."""
    while True:
        try:
            model_ids = app.state.model_registry.get_all_model_ids()
            for model_id in model_ids:
                clients = app.state.model_registry.get_clients(model_id)
                for client in clients:
                    is_healthy = await client.check_health()

                    client_type = "BaseVLLMClient" if not isinstance(client, ComputeHordeVLLMClient) else "ComputeHordeVLLMClient"
                    logger.debug(f"{client_type} for {model_id} is {'healthy' if is_healthy else 'unhealthy'}")
        except Exception as e:
            logger.error(f"Error checking clients health: {e}")

        await asyncio.sleep(60)

async def run_trusted_verification_task():
    """Background task to periodically run trusted verification."""
    while True:
        try:
            await run_trusted_verification(app)
        except Exception as e:
            logger.error(f"Error running trusted verification: {e}")

        # Sleep for a while before checking again
        # This is separate from the verification interval check in RequestLog
        await asyncio.sleep(300)  # Check every 5 minutes
