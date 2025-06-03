import pathlib

import bittensor
from compute_horde_sdk.v1 import ComputeHordeClient, ComputeHordeJobSpec, ExecutorClass
from loguru import logger


class ComputeHordeJobCreator:
    DOCKER_IMAGE_REPOSITORY = "backenddevelopersltd"
    DOCKER_IMAGE_TAG = "latest"

    def __init__(self, facilitator_url: str, job_namespace="SN1.VLLM"):
        self.facilitator_url = facilitator_url
        self.job_namespace = job_namespace

        # Initialize wallet and SN12 client.
        self.wallet = bittensor.wallet(
            name="validator", hotkey="default", path=(pathlib.Path(__file__).parent / "wallets").as_posix()
        )
        self.compute_horde_client = ComputeHordeClient(
            hotkey=self.wallet.hotkey,
            compute_horde_validator_hotkey=self.wallet.hotkey.ss58_address,
            facilitator_url=facilitator_url,
        )

    async def create_job_remote_vllm(self, llm_model_id) -> str:
        """
        Creates a job to run the VLLM remotely and returns the accessible URL.
        """
        docker_image = f"{self.DOCKER_IMAGE_REPOSITORY}/{llm_model_id}:{self.DOCKER_IMAGE_TAG}"
        reproducible_vllm_streaming_job_spec = ComputeHordeJobSpec(
            executor_class=ExecutorClass.always_on__llm__a6000,
            job_namespace=self.job_namespace,
            docker_image=docker_image,
            args=["python", "app.py"],
            artifacts_dir="/artifacts",
            streaming=True,
            download_time_limit_sec=15,
            execution_time_limit_sec=300,
            streaming_start_time_limit_sec=120,
            upload_time_limit_sec=5,
        )
        reproducible_vllm_streaming_job = await self.compute_horde_client.create_job(
            reproducible_vllm_streaming_job_spec
        )
        await reproducible_vllm_streaming_job.wait_for_streaming(timeout=120)

        miner_uuid = reproducible_vllm_streaming_job.uuid
        streaming_server_address = reproducible_vllm_streaming_job.streaming_server_address
        streaming_server_port = reproducible_vllm_streaming_job.streaming_server_port

        logger.info(f"Streaming server: {streaming_server_address}:{streaming_server_port}")

        return miner_uuid, f"https://{streaming_server_address}:{streaming_server_port}"
