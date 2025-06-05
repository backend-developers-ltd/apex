import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import aiohttp
from loguru import logger


@dataclass
class LogEntry:
    miner_uuid: str
    endpoint: str
    request: Dict[str, Any]
    response: Dict[str, Any]
    status: int
    timestamp: float = field(default_factory=time.time)


class MemoryLog:
    def __init__(self):
        self._log: Dict[str, LogEntry] = {}

    def add_entry(
        self, miner_uuid: str, endpoint: str, request: Dict[str, Any], response: Dict[str, Any], status: int
    ) -> str:
        entry_id = str(uuid.uuid4())
        entry = LogEntry(miner_uuid=miner_uuid, endpoint=endpoint, request=request, response=response, status=status)
        self._log[entry_id] = entry
        return entry_id

    def get_entry(self, entry_id: str) -> LogEntry | None:
        return self._log.get(entry_id)


class RemoteVLLMClient:
    def __init__(self, base_url: str, miner_uuid: str):
        self.base_url = base_url.rstrip("/")
        self.miner_uuid = miner_uuid
        self.memory_log = MemoryLog()
        self.active = True

    @classmethod
    async def get_max_tokens(
            cls,
            sampling_params: dict[str, str | float | int | bool],
            default_value: int = 512,
    ) -> int:
        # Process max tokens with backward compatibility.
        max_tokens = sampling_params.get("max_tokens")
        if max_tokens is None:
            max_tokens = sampling_params.get("max_new_tokens")
        if max_tokens is None:
            max_tokens = sampling_params.get("max_completion_tokens", default_value)
        return max_tokens

    async def generate(
        self,
        messages: List[str] | List[Dict[str, str]],
        sampling_params: Dict[str, Any] | None = None,
        seed: int | None = None,
        continue_last_message: bool = False,
    ) -> str:
        payload = {
            "messages": messages,
            "sampling_parameters": sampling_params or {},
            "seed": seed,
            "continue_last_message": continue_last_message,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/generate", json=payload) as resp:
                data = await resp.json()
                entry_id = self.memory_log.add_entry(
                    miner_uuid=self.miner_uuid,
                    endpoint="generate",
                    request=payload,
                    response=data,
                    status=resp.status
                )
                logger.info(f"[RemoteVLLMClient] /generate [{entry_id}] -> Status: {resp.status}")
                if resp.status != 200:
                    self.active = False
                    raise RuntimeError(f"Remote generation failed: {data.get('error', 'Unknown error')}")
                return data.get("result", "")
            return None
        return None

    async def generate_logits(
        self,
        messages: List[str] | List[Dict[str, str]],
        top_logprobs: int = 10,
        sampling_params: Dict[str, Any] | None = None,
        seed: int | None = None,
        continue_last_message: bool = False,
    ) -> Tuple[Dict[str, float], str]:
        payload = {
            "messages": messages,
            "sampling_parameters": sampling_params or {},
            "seed": seed,
            "continue_last_message": continue_last_message,
            "top_logprobs": top_logprobs,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/generate_logits", json=payload) as resp:
                data = await resp.json()
                entry_id = self.memory_log.add_entry(
                    miner_uuid=self.miner_uuid,
                    endpoint="generate_logits",
                    request=payload,
                    response=data,
                    status=resp.status,
                )
                logger.info(f"[RemoteVLLMClient] /generate_logits [{entry_id}] -> Status: {resp.status}")
                if resp.status != 200:
                    self.active = False
                    raise RuntimeError(f"Remote logits failed: {data.get('error', 'Unknown error')}")
                return data.get("logits", {}), data.get("prompt", "")
            return None
        return None

    async def unload_model(self):
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(f"{self.base_url}/terminate") as resp:
                    data = await resp.json()
                    entry_id = self.memory_log.add_entry(
                        miner_uuid=self.miner_uuid,
                        endpoint="terminate",
                        request=None,
                        response=data,
                        status=resp.status,
                    )
                    logger.info(
                        f"[RemoteVLLMClient] /terminate [{entry_id}] -> Status: {resp.status}")
            except Exception as e:
                logger.info(f"[RemoteVLLMClient] already shut down: {e}")
            finally:
                self.active = False

    def __del__(self):
        self.unload_model()

    @staticmethod
    def format_messages(messages: list[str] | list[dict[str, str]]) -> list[
        dict[str, str | list[dict[str, str]]]]:
        return messages


