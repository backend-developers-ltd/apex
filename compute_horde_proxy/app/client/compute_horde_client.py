import asyncio
import httpx
import os
import ssl
import tempfile
import time
import pathlib
import uuid
from typing import Dict, Any, Optional
from fastapi import status

from compute_horde_sdk.v1 import ComputeHordeClient, ComputeHordeJobSpec, ExecutorClass
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509 import Certificate
import bittensor
from loguru import logger

from app.client.base_client import BaseVLLMClient

class ComputeHordeVLLMClient(BaseVLLMClient):
    """
    Client for interacting with a VLLM instance through Compute Horde.

    This client creates and manages a Compute Horde job running VLLM.
    """

    def __init__(self, model_id: str):
        """
        Initialize the Compute Horde client.

        Args:
            model_id: The ID of the model to use.
            endpoint_url: The URL of the Compute Horde job.
            ssl_context: The SSL context for secure communication.
        """
        super().__init__(model_id)
        self.endpoint_url = None
        self.ssl_context = None
        self.miner_id = None  # UUID of the compute horde miner
        self.local = False  # ComputeHordeVLLMClient is not local
        self.trusted = False

    @classmethod
    async def create(cls, model_id: str, trusted: bool = False):
        """
        Create a new Compute Horde VLLM client.

        Args:
            model_id: The ID of the model to use.
            trusted: Whether to create a trusted client for verification.

        Returns:
            A new ComputeHordeVLLMClient instance.
        """
        try:
            logger.info(f"Creating Compute Horde client for model: {model_id}, trusted: {trusted}")

            # Initialize wallet
            try:
                wallet_name = os.environ["BITTENSOR_WALLET_NAME"]
                wallet_hotkey = os.environ["BITTENSOR_WALLET_HOTKEY"]
                wallet_path = (pathlib.Path(__file__).parent.parent / "wallets").as_posix()
                logger.debug(f"Initializing wallet with name: {wallet_name}, hotkey: {wallet_hotkey}, path: {wallet_path}")
                wallet = bittensor.wallet(name=wallet_name, hotkey=wallet_hotkey, path=wallet_path)
                logger.debug(f"Wallet initialized successfully: {wallet.hotkey.ss58_address}")
            except Exception as e:
                logger.error(f"Failed to initialize wallet: {e}")
                raise

            # Initialize ComputeHordeClient
            try:
                facilitator_url = os.environ["COMPUTE_HORDE_FACILITATOR_URL"]
                validator_hotkey = os.environ["COMPUTE_HORDE_VALIDATOR_HOTKEY"]
                logger.debug(f"Initializing ComputeHordeClient with facilitator_url: {facilitator_url}, validator_hotkey: {validator_hotkey}")
                client = ComputeHordeClient(
                    hotkey=wallet.hotkey,
                    compute_horde_validator_hotkey=validator_hotkey,
                    facilitator_url=facilitator_url
                )
                logger.debug("ComputeHordeClient initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize ComputeHordeClient: {e}")
                raise

            # Get docker image from environment variable
            try:
                # Replace slashes with double dashes for Huggingface models
                docker_model_id = model_id.replace('/', '--')

                # Convert to environment variable format (uppercase with underscores)
                env_model_id = docker_model_id.replace('-', '_').upper()

                # Try to get model-specific docker image, fall back to default format
                model_specific_env_var = f"DOCKER_IMAGE_{env_model_id}"
                if model_specific_env_var in os.environ:
                    docker_image = os.environ[model_specific_env_var]
                    logger.debug(f"Using model-specific docker image: {docker_image}")
                else:
                    docker_image = os.environ["DEFAULT_DOCKER_IMAGE"]
                    logger.debug(f"Using default docker image: {docker_image}")
            except Exception as e:
                logger.error(f"Failed to determine docker image: {e}")
                raise

            # Get job spec parameters from environment variables
            try:
                # These are required in app.py validation
                download_time_limit_sec = int(os.environ["COMPUTE_HORDE_DOWNLOAD_TIME_LIMIT_SEC"])
                execution_time_limit_sec = int(os.environ["COMPUTE_HORDE_EXECUTION_TIME_LIMIT_SEC"])
                streaming_start_time_limit_sec = int(os.environ["COMPUTE_HORDE_STREAMING_START_TIME_LIMIT_SEC"])
                upload_time_limit_sec = int(os.environ["COMPUTE_HORDE_UPLOAD_TIME_LIMIT_SEC"])
                logger.debug(f"Job spec parameters: download_time_limit_sec={download_time_limit_sec}, execution_time_limit_sec={execution_time_limit_sec}, streaming_start_time_limit_sec={streaming_start_time_limit_sec}, upload_time_limit_sec={upload_time_limit_sec}")
            except Exception as e:
                logger.error(f"Failed to get job spec parameters: {e}")
                raise

            # Create job spec
            try:
                job_spec = ComputeHordeJobSpec(
                    executor_class=ExecutorClass.always_on__gpu_a100_80gb,
                    job_namespace="SN1.VLLM",
                    docker_image=docker_image,
                    args=[],
                    artifacts_dir="/artifacts",
                    streaming=True,
                    download_time_limit_sec=download_time_limit_sec,
                    execution_time_limit_sec=execution_time_limit_sec,
                    streaming_start_time_limit_sec=streaming_start_time_limit_sec,
                    upload_time_limit_sec=upload_time_limit_sec,
                )
                logger.debug("Job spec created successfully")
            except Exception as e:
                logger.error(f"Failed to create job spec: {e}")
                raise

            # Create job
            try:
                logger.info("Creating Compute Horde job...")
                job = await client.create_job(job_spec, on_trusted_miner=trusted)
                logger.info(f"Job created successfully with UUID: {job.uuid}")
            except Exception as e:
                logger.error(f"Failed to create job: {e}")
                raise

            # Wait for streaming
            try:
                logger.info(f"Waiting for streaming to start (timeout: {streaming_start_time_limit_sec}s)...")
                await job.wait_for_streaming(timeout=streaming_start_time_limit_sec)
                logger.info("Streaming started successfully")
            except Exception as e:
                logger.error(f"Failed to wait for streaming: {e}")
                raise

            # Create SSL context
            try:
                logger.debug("Creating SSL context...")
                ssl_context = cls._create_ssl_context(
                    job.streaming_public_cert,
                    job.streaming_private_key,
                    job.streaming_server_cert,
                )
                logger.debug("SSL context created successfully")
            except Exception as e:
                logger.error(f"Failed to create SSL context: {e}")
                raise

            # Set the miner ID from the job
            miner_id = job.uuid  # TODO actual method to get
            if not miner_id:
                # If miner_id is not available directly, try to extract it from other attributes
                miner_id = str(uuid.uuid4())  # Fallback to a random UUID
                logger.warning(f"Could not get miner ID from job, using fallback: {miner_id}")
            else:
                logger.debug(f"Using miner ID from job: {miner_id}")

            # Create instance
            try:
                miner_url = f"https://{job.streaming_server_address}:{job.streaming_server_port}"
                logger.debug(f"Miner URL: {miner_url}")
                instance = cls(model_id)
                instance.endpoint_url = miner_url
                instance.ssl_context = ssl_context
                instance.miner_id = miner_id
                instance.trusted = trusted
                logger.debug("Instance created successfully")
            except Exception as e:
                logger.error(f"Failed to create instance: {e}")
                raise

            # Check if the remote model is ready
            try:
                logger.info("Checking if remote model is ready...")
                instance._ready = await cls.check_endpoint_health(miner_url, ssl_context)
                logger.info(f"Remote model is {'ready' if instance._ready else 'not ready'}")
            except Exception as e:
                logger.error(f"Failed to check if remote model is ready: {e}")
                instance._ready = False

            return instance
        except Exception as e:
            logger.error(f"Failed to create Compute Horde client: {e}")
            raise

    @classmethod
    async def check_endpoint_health(cls, endpoint_url: str, ssl_context: ssl.SSLContext) -> bool:
        """
        Check if the endpoint is healthy by polling /health/model.

        Args:
            endpoint_url: The URL of the endpoint to check.
            ssl_context: The SSL context for secure communication.

        Returns:
            True if the endpoint is healthy, False otherwise.
        """
        logger.info(f"Checking endpoint health: {endpoint_url}")
        attempts = 0
        max_attempts = 240  # 4 minutes

        for _ in range(max_attempts):
            attempts += 1
            try:
                async with httpx.AsyncClient(verify=ssl_context) as client:
                    logger.debug(f"Attempt {attempts}/{max_attempts}: Sending health check request to {endpoint_url}/health/model")
                    resp = await client.get(f"{endpoint_url}/health/model", timeout=3)
                    if resp.status_code == 200:
                        logger.info(f"Endpoint {endpoint_url} is healthy (status code: 200)")
                        return True
                    else:
                        logger.warning(f"Endpoint {endpoint_url} returned non-200 status code: {resp.status_code}")
            except httpx.ConnectError as e:
                logger.debug(f"Connection error while checking endpoint health: {e}")
            except httpx.ReadTimeout as e:
                logger.debug(f"Read timeout while checking endpoint health: {e}")
            except httpx.WriteTimeout as e:
                logger.debug(f"Write timeout while checking endpoint health: {e}")
            except httpx.TimeoutException as e:
                logger.debug(f"Timeout while checking endpoint health: {e}")
            except ssl.SSLError as e:
                logger.warning(f"SSL error while checking endpoint health: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error while checking endpoint health: {e}")

            # Only log every 10 attempts to avoid flooding the logs
            if attempts % 10 == 0:
                logger.info(f"Still waiting for endpoint {endpoint_url} to become healthy (attempt {attempts}/{max_attempts})")

            await asyncio.sleep(1)

        logger.error(f"Endpoint {endpoint_url} is not healthy after {max_attempts} attempts")
        return False  # timeout waiting for model readiness

    @staticmethod
    def _create_ssl_context(client_cert: Certificate, client_key: rsa.RSAPrivateKey, server_cert_pem: str) -> ssl.SSLContext:
        """
        Create an SSL context for secure communication with the Compute Horde job.

        Args:
            client_cert: The client certificate.
            client_key: The client private key.
            server_cert_pem: The server certificate in PEM format.

        Returns:
            An SSL context.
        """
        cert_pem = client_cert.public_bytes(encoding=serialization.Encoding.PEM).decode("utf-8")
        key_pem = client_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode("utf-8")

        with tempfile.NamedTemporaryFile("w+", delete=False) as cert_file, \
             tempfile.NamedTemporaryFile("w+", delete=False) as key_file, \
             tempfile.NamedTemporaryFile("w+", delete=False) as ca_file:
            cert_file.write(cert_pem)
            cert_file.flush()
            key_file.write(key_pem)
            key_file.flush()
            ca_file.write(server_cert_pem)
            ca_file.flush()

            ctx = ssl.create_default_context(cafile=ca_file.name)
            ctx.load_cert_chain(certfile=cert_file.name, keyfile=key_file.name)
            return ctx
