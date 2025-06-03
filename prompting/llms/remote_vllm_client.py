import asyncio
import pathlib
import ssl
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from random import random
from typing import Any, Dict, List, Tuple

import bittensor
import httpx
from compute_horde_sdk.v1 import ComputeHordeClient, ComputeHordeJobSpec, ExecutorClass
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509 import Certificate
from loguru import logger

from prompting.llms.vllm_llm import ReproducibleVLLM


@dataclass
class LogEntry:
    streaming_job_uuid: str
    endpoint: str
    request: Dict[str, Any]
    response: Dict[str, Any]
    status: int
    timestamp: float = field(default_factory=time.time)


class MemoryLog:
    VERIFICATION_PROBABILITY: float = 0.2  # Probability to verify with TrustedMiner

    def __init__(self):
        self._log: Dict[str, LogEntry] = {}

    def add_entry(
        self, streaming_job_uuid: str, endpoint: str, request: Dict[str, Any], response: Dict[str, Any], status: int
    ):
        # Save some of the requests to be verified with TrustedMiner
        if random() < self.VERIFICATION_PROBABILITY:
            entry_id = str(uuid.uuid4())
            entry = LogEntry(
                streaming_job_uuid=streaming_job_uuid,
                endpoint=endpoint,
                request=request,
                response=response,
                status=status,
            )
            self._log[entry_id] = entry

    def get_entry(self, entry_id: str) -> LogEntry | None:
        return self._log.get(entry_id)


@dataclass
class Tokenizer:
    bos_token: str
    eos_token: str


class RemoteVLLMClient:
    def __init__(self, llm_model_docker_image: str, facilitator_url: str = None, job_namespace: str = "SN1.VLLM"):
        self.llm_model_docker_image = llm_model_docker_image
        self.facilitator_url = facilitator_url
        self.job_namespace = job_namespace

        self.wallet = bittensor.wallet(
            name="validator", hotkey="default", path=(pathlib.Path(__file__).parent / "wallets").as_posix()
        )
        self.compute_horde_client = ComputeHordeClient(
            hotkey=self.wallet.hotkey,
            compute_horde_validator_hotkey=self.wallet.hotkey.ss58_address,
            facilitator_url=self.facilitator_url,
        )

        self.memory_log = MemoryLog()
        self._job_creation_lock = asyncio.Lock()
        self.active = False
        self.streaming_server_url = None
        self.ssl_context = None
        self.streaming_job = None
        self.streaming_job_uuid = None
        self.tokenizer = None

    @staticmethod
    def create_ssl_context_with_tempfiles(
        client_cert: Certificate, client_key: rsa.RSAPrivateKey, server_cert_pem: str
    ) -> ssl.SSLContext:
        cert_pem = client_cert.public_bytes(encoding=serialization.Encoding.PEM).decode("utf-8")
        key_pem = client_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode("utf-8")

        with tempfile.NamedTemporaryFile("w+", delete=False) as cert_file, tempfile.NamedTemporaryFile(
            "w+", delete=False
        ) as key_file, tempfile.NamedTemporaryFile("w+", delete=False) as ca_file:
            cert_file.write(cert_pem)
            cert_file.flush()
            key_file.write(key_pem)
            key_file.flush()
            ca_file.write(server_cert_pem)
            ca_file.flush()

            ctx = ssl.create_default_context(cafile=ca_file.name)
            ctx.load_cert_chain(certfile=cert_file.name, keyfile=key_file.name)
            return ctx

    async def create_job_remote_vllm(self):
        logger.info("[RemoteVLLMClient] Create a new streaming job")
        job_spec = ComputeHordeJobSpec(
            executor_class=ExecutorClass.always_on__llm__a6000,
            job_namespace=self.job_namespace,
            docker_image=self.llm_model_docker_image,
            args=["python", "app.py"],
            artifacts_dir="/artifacts",
            streaming=True,
            download_time_limit_sec=15,
            execution_time_limit_sec=600,
            streaming_start_time_limit_sec=300,
            upload_time_limit_sec=5,
        )
        self.streaming_job = await self.compute_horde_client.create_job(job_spec)
        await self.streaming_job.wait_for_streaming(timeout=300)

        self.ssl_context = self.create_ssl_context_with_tempfiles(
            self.streaming_job.streaming_public_cert,
            self.streaming_job.streaming_private_key,
            self.streaming_job.streaming_server_cert,
        )

        streaming_server_address = self.streaming_job.streaming_server_address
        streaming_server_port = self.streaming_job.streaming_server_port

        self.streaming_server_url = f"https://{streaming_server_address}:{streaming_server_port}"
        logger.info(f"[RemoteVLLMClient] Streaming server: {self.streaming_server_url}")

        self.streaming_job_uuid = self.streaming_job.uuid
        self.active = True

        self.tokenizer = await self.get_tokenizer()

    async def status(self):
        await self.streaming_job.refresh_from_facilitator()
        return self.streaming_job.status

    async def streaming_job_active(self, retries: int = 3, delay: float = 3.0):
        for attempt in range(1, retries + 1):
            try:
                await self.streaming_job.refresh_from_facilitator()

                if self.streaming_job.status.is_streaming_ready():
                    return

                logger.warning(f"[RemoteVLLMClient] Streaming job not in progress (attempt {attempt}).")

                async with self._job_creation_lock:
                    # Create new job
                    await self.create_job_remote_vllm()
                return

            except Exception as e:
                logger.error(f"[RemoteVLLMClient] Failed in job check/create (attempt {attempt}): {e}")
                await asyncio.sleep(delay)

        raise RuntimeError("[RemoteVLLMClient] Could not ensure active job after multiple attempts.")

    @classmethod
    async def get_max_tokens(
        cls,
        sampling_params: dict[str, str | float | int | bool],
        default_value: int = 512,
    ) -> int:
        # Reused classmethod form ReproducibleVLLM
        return await ReproducibleVLLM.get_max_tokens(sampling_params, default_value)

    async def get_tokenizer(self):
        async with httpx.AsyncClient(verify=self.ssl_context, timeout=30) as client:
            resp = await client.get(f"{self.streaming_server_url}/tokenizer")
            data = resp.json()
            logger.info(f"[RemoteVLLMClient] /tokenizer [{self.streaming_job_uuid}] -> Status: {resp.status_code}")
            if resp.status_code != 200:
                self.active = False
                raise RuntimeError(f"[RemoteVLLMClient] Remote tokenizer failed: {data.get('error', 'Unknown error')}")
            return Tokenizer(data.get("bos_token", "<s>"), data.get("eos_token", "</s>"))

    async def generate(
        self,
        messages: List[str] | List[Dict[str, str]],
        sampling_params: Dict[str, Any] | None = None,
        seed: int | None = None,
        continue_last_message: bool = False,
    ) -> str:
        await self.streaming_job_active()
        payload = {
            "messages": messages,
            "sampling_parameters": sampling_params or {},
            "seed": seed,
            "continue_last_message": continue_last_message,
        }
        async with httpx.AsyncClient(verify=self.ssl_context, timeout=30) as client:
            resp = await client.post(f"{self.streaming_server_url}/generate", json=payload)
            data = resp.json()
            self.memory_log.add_entry(
                streaming_job_uuid=self.streaming_job_uuid,
                endpoint="generate",
                request=payload,
                response=data,
                status=resp.status_code,
            )
            logger.info(f"[RemoteVLLMClient] /generate [{self.streaming_job_uuid}] -> Status: {resp.status_code}")
            if resp.status_code != 200:
                self.active = False
                raise RuntimeError(f"[RemoteVLLMClient] Remote generation failed: {data.get('error', 'Unknown error')}")
            return data.get("result", "")

    async def generate_logits(
        self,
        messages: List[str] | List[Dict[str, str]],
        top_logprobs: int = 10,
        sampling_params: Dict[str, Any] | None = None,
        seed: int | None = None,
        continue_last_message: bool = False,
    ) -> Tuple[Dict[str, float], str]:
        await self.streaming_job_active()
        payload = {
            "messages": messages,
            "sampling_parameters": sampling_params or {},
            "seed": seed,
            "continue_last_message": continue_last_message,
            "top_logprobs": top_logprobs,
        }
        async with httpx.AsyncClient(verify=self.ssl_context, timeout=30) as client:
            resp = await client.post(f"{self.streaming_server_url}/generate_logits", json=payload)
            data = resp.json()
            self.memory_log.add_entry(
                streaming_job_uuid=self.streaming_job_uuid,
                endpoint="generate_logits",
                request=payload,
                response=data,
                status=resp.status_code,
            )
            logger.info(
                f"[RemoteVLLMClient] /generate_logits [{self.streaming_job_uuid}] -> Status: {resp.status_code}"
            )
            if resp.status_code != 200:
                self.active = False
                raise RuntimeError(f"[RemoteVLLMClient] Remote generate_logits failed: {data.get('error', 'Unknown error')}")
            return data.get("logits", {}), data.get("prompt", "")

    async def unload_model(self):
        if not self.active or not self.ssl_context:
            return
        try:
            async with httpx.AsyncClient(verify=self.ssl_context) as client:
                resp = await client.post(f"{self.streaming_server_url}/terminate")
                logger.info(f"[RemoteVLLMClient] /terminate -> Status: {resp.status_code}")
        except Exception as e:
            logger.info(f"[RemoteVLLMClient] already shut down: {e}")
        finally:
            self.active = False

    @staticmethod
    def format_messages(messages: list[str] | list[dict[str, str]]) -> list[dict[str, str | list[dict[str, str]]]]:
        # Reused staticmethod form ReproducibleVLLM
        return ReproducibleVLLM.format_messages(messages)
