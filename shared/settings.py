import sys
import time

# Need to delete logging from modules and load in standard python logging
if "logging" in sys.modules:
    del sys.modules["logging"]
current_directory = sys.path.pop(0)
import logging

sys.path.insert(0, current_directory)

import os
from functools import cached_property
from typing import Any, Literal, Optional

import bittensor as bt
import dotenv
from bittensor.core.metagraph import Metagraph
from bittensor.core.subtensor import Subtensor
from loguru import logger
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

from shared.misc import is_cuda_available

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


class SharedSettings(BaseSettings):
    _instance: Optional["SharedSettings"] = None
    _instance_mode: Optional[str] = None
    _last_metagraph: Metagraph = None
    _last_update_time: float = 0
    _block_sync_last_time: float = 0
    _block_sync_interval: float = 300
    _subtensor: Subtensor | None = None

    mode: Literal["api", "validator", "miner", "mock"] = Field("validator", env="MODE")
    MOCK: bool = False

    # W&B.
    WANDB_ON: bool = Field(True, env="WANDB_ON")
    WANDB_ENTITY: Optional[str] = Field("macrocosmos", env="WANDB_ENTITY")
    WANDB_PROJECT_NAME: Optional[str] = Field("prompting-validators", env="WANDB_PROJECT_NAME")
    WANDB_API_KEY: Optional[str] = Field(None, env="WANDB_API_KEY")
    WANDB_OFFLINE: bool = Field(False, env="WANDB_OFFLINE")
    WANDB_NOTES: str = Field("", env="WANDB_NOTES")
    MAX_WANDB_DURATION: int = 24

    # Logging.
    LOGGING_DONT_SAVE_EVENTS: bool = Field(True, env="LOGGING_DONT_SAVE_EVENTS")
    LOG_WEIGHTS: bool = Field(False, env="LOG_WEIGHTS")
    LOG_TIMINGS: bool = Field(False, env="LOG_TIMINGS")
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")

    # Neuron parameters.
    NEURON_DEVICE: str | None = Field("cuda", env="NEURON_DEVICE")
    NEURON_TIMEOUT: int = Field(20, env="NEURON_TIMEOUT")
    INFERENCE_TIMEOUT: int = Field(30, env="INFERENCE_TIMEOUT")
    MAX_TIMEOUT: int = Field(300, env="INFERENCE_TIMEOUT")
    NEURON_DISABLE_SET_WEIGHTS: bool = Field(False, env="NEURON_DISABLE_SET_WEIGHTS")
    NEURON_VPERMIT_TAO_LIMIT: int = Field(4096, env="NEURON_VPERMIT_TAO_LIMIT")
    NEURON_QUERY_UNIQUE_COLDKEYS: bool = Field(False, env="NEURON_QUERY_UNIQUE_COLDKEYS")
    NEURON_QUERY_UNIQUE_IPS: bool = Field(False, env="NEURON_QUERY_UNIQUE_IPS")
    NEURON_MAX_TOKENS: int = Field(512, env="NEURON_MAX_TOKENS")
    REWARD_STEEPNESS: float = Field(0.7, env="STEEPNESS")

    TASK_QUEUE_LENGTH_THRESHOLD: int = Field(50, env="TASK_QUEUE_LENGTH_THRESHOLD")
    SCORING_QUEUE_LENGTH_THRESHOLD: int = Field(50, env="SCORING_QUEUE_LENGTH_THRESHOLD")
    GEMMA_API_KEY: Optional[str] = Field(None, env="GEMMA_API_KEY")
    HF_TOKEN: Optional[str] = Field(None, env="HF_TOKEN")

    SCORING_API_PORT: int = Field(8095, env="SCORING_API_PORT")
    # Hard-code MC validator axon, since it might be overwritten in the metagraph.
    MC_VALIDATOR_HOTKEY: str = Field("5CGLCBndTR1BvQZzn429ckT8GyxduzyjMgt4K1UVTYa8gKfb", env="MC_VALIDATOR_HOTKEY")
    MC_VALIDATOR_AXON: str = Field("184.105.215.196:42174", env="MC_VALIDATOR_AXON")

    # ==== API =====
    # Hotkey used to run api, defaults to Macrocosmos
    API_HOTKEY: str = Field("5CGLCBndTR1BvQZzn429ckT8GyxduzyjMgt4K1UVTYa8gKfb", env="API_HOTKEY")
    # Scoring request rate limit in seconds.
    SCORING_RATE_LIMIT_SEC: float = Field(5, env="SCORING_RATE_LIMIT_SEC")
    # Scoring queue threshold when rate-limit start to kick in, used to query validator API with scoring requests.
    SCORING_QUEUE_API_THRESHOLD: int = Field(1, env="SCORING_QUEUE_API_THRESHOLD")
    API_TEST_MODE: bool = Field(False, env="API_TEST_MODE")
    # Top UIDs used for the primary API stream.
    API_TOP_UIDS_TO_STREAM: int = Field(10, env="API_TOP_UIDS_TO_STREAM")
    # Top UIDs to sample from to use for the primary API stream.
    API_TOP_UIDS_SAMPLE: int = Field(300, env="API_TOP_UIDS_SAMPLE")
    # Exploration ratio for UIDs to include into primary API stream besides top incentive.
    API_UIDS_EXPLORE: float = Field(0.0, env="API_UIDS_EXPLORE")
    # Minimum UIDs to query from API (not recommended to set anything lower than 30, since scoring will be slowed down).
    API_EXTRA_UIDS_QUERY: int = Field(50, env="API_MIN_UIDS_QUERY")
    # If enabled, sends request to a random validator, otherwise sends to all available validators.
    API_ENABLE_BALANCE: bool = Field(True, env="API_ENABLE_BALANCE")
    OVERRIDE_AVAILABLE_AXONS: list[str] | None = Field(None, env="OVERRIDE_AVAILABLE_AXONS")

    # Validator scoring API (.env.validator).
    SCORE_ORGANICS: bool = Field(False, env="SCORE_ORGANICS")
    WORKERS: int = Field(1, env="WORKERS")

    # API Management (.env.api).
    API_PORT: int = Field(8005, env="API_PORT")
    API_HOST: str = Field("0.0.0.0", env="API_HOST")
    # Validator scoring API address, also used for miner availabilities.
    # TODO: Choose this dynamically from the network
    VALIDATOR_API: str = Field("0.0.0.0:8094", env="VALIDATOR_API")
    # Default SN1 API address
    DEFAULT_SN1_API: str = Field("http://0.0.0.0:8005/v1", env="DEFAULT_SN1_API")
    # File with keys used to access API.
    API_KEYS_FILE: str = Field("api_keys.json", env="API_KEYS_FILE")
    # Admin key used to generate API keys.
    ADMIN_KEY: str | None = Field(None, env="ADMIN_KEY")
    # ==============

    # Additional Validator Fields.
    NETUID: Optional[int] = Field(61, env="NETUID")
    TEST: bool = False
    OPENAI_API_KEY: Optional[str] = Field(None, env="OPENAI_API_KEY")
    WALLET_NAME: Optional[str] = Field(None, env="WALLET_NAME")
    HOTKEY: Optional[str] = Field(None, env="HOTKEY")
    AXON_PORT: Optional[int] = Field(None, env="AXON_PORT")
    TEST_MINER_IDS: list[int] = Field([], env="TEST_MINER_IDS")
    SUBTENSOR_NETWORK: Optional[str] = Field(None, env="SUBTENSOR_NETWORK")
    MAX_ALLOWED_VRAM_GB: float = Field(62, env="MAX_ALLOWED_VRAM_GB")
    PROXY_URL: Optional[str] = Field(None, env="PROXY_URL")
    LLM_MODEL: list[str] = [
        "mrfakename/mistral-small-3.1-24b-instruct-2503-hf",
    ]
    SAMPLING_PARAMS: dict[str, Any] = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 50,
        "max_tokens": 2048,
    }
    LLM_MODEL_RAM: float = Field(70, env="LLM_MODEL_RAM")
    OPENAI_API_KEY: str | None = Field(None, env="OPENAI_API_KEY")
    SN19_API_KEY: str | None = Field(None, env="SN19_API_KEY")
    SN19_API_URL: str | None = Field(None, env="SN19_API_URL")
    GPT_MODEL_CONFIG: dict[str, dict[str, Any]] = {
        "gpt-3.5-turbo": {
            "context_window": 16_385,
            "max_tokens": 4096,
            "vision": False,
            "score": 100,
            "upgrade": "gpt-4-turbo",
            "input_token_cost": 0.0005,
            "output_token_cost": 0.0015,
        },
        "gpt-4-turbo": {
            "context_window": 128_000,
            "max_tokens": 4096,
            "vision": True,
            "score": 200,
            "upgrade": "gpt-4o",
            "input_token_cost": 0.01,
            "output_token_cost": 0.03,
        },
        "gpt-4o": {
            "context_window": 128_000,
            "max_tokens": 4096,
            "vision": True,
            "score": 300,
            "input_token_cost": 0.005,
            "output_token_cost": 0.015,
        },
    }
    model_config = {"frozen": True, "arbitrary_types_allowed": False}

    @model_validator(mode="before")
    def validate_mode(cls, v):
        if v["mode"] == "api":
            if not dotenv.load_dotenv(".env.api"):
                logger.warning("No .env.api file found. Please create one.")
            if not os.getenv("SCORING_KEY"):
                logger.warning(
                    "No SCORING_KEY found in .env.api file. You must add a scoring key that will allow us to forward miner responses to the validator for scoring."
                )
        elif v["mode"] == "miner":
            if not dotenv.load_dotenv(".env.miner"):
                logger.warning("No .env.miner file found. Please create one.")
        elif v["mode"] == "validator":
            if not dotenv.load_dotenv(".env.validator"):
                logger.warning("No .env.validator file found. Please create one.")

        return v

    @classmethod
    def load_env_file(cls, mode: Literal["miner", "validator", "mock", "api"]):
        """Load the appropriate .env file based on the mode."""
        if mode == "miner":
            dotenv_file = ".env.miner"
        elif mode == "validator":
            dotenv_file = ".env.validator"
        # For mock testing, still make validator env vars available where possible.
        elif mode == "mock":
            dotenv_file = ".env.validator"
        elif mode == "api":
            dotenv_file = ".env.api"
            if os.getenv("API_TEST_MODE"):
                logger.warning("API_TEST_MODE is set to true - THE API IS RUNNING IN TEST MODE.")
        else:
            raise ValueError(f"Invalid mode: {mode}")

        if dotenv_file:
            if not dotenv.load_dotenv(dotenv.find_dotenv(filename=dotenv_file)):
                logger.warning(
                    f"No {dotenv_file} file found. The use of args when running a {mode} will be deprecated "
                    "in the near future."
                )

    @classmethod
    def load(cls, mode: Literal["miner", "validator", "mock", "api"]) -> "SharedSettings":
        """Load or retrieve the Settings instance based on the mode."""
        if cls._instance is not None and cls._instance_mode == mode:
            return cls._instance
        else:
            cls.load_env_file(mode)
            cls._instance = cls(mode=mode)
            cls._instance_mode = mode
            return cls._instance

    @model_validator(mode="before")
    def complete_settings(cls, values: dict[str, Any]) -> dict[str, Any]:
        mode = values["mode"]
        netuid = values.get("NETUID", 61)
        if netuid is None:
            raise ValueError("NETUID must be specified")
        values["TEST"] = netuid != 1
        if values.get("TEST_MINER_IDS"):
            values["TEST_MINER_IDS"] = values["TEST_MINER_IDS"]
        if mode == "mock":
            values["MOCK"] = True
            values["NEURON_DEVICE"] = "cpu"
            logger.info("Running in mock mode. Bittensor objects will not be initialized.")
            return values

        return values

    @cached_property
    def WALLET(self):
        # TODO: Move chain-related stuff out of settings.
        wallet_name = self.WALLET_NAME  # or config().wallet.name
        hotkey = self.HOTKEY  # or config().wallet.hotkey
        logger.info(f"Instantiating wallet with name: {wallet_name}, hotkey: {hotkey}")
        return bt.wallet(name=wallet_name, hotkey=hotkey)

    @cached_property
    def SUBTENSOR(self) -> Subtensor:
        """Lazy subtensor initialization."""
        if self._subtensor is not None:
            return self._subtensor
        # TODO: Move chain-related stuff out of settings.
        subtensor_network = self.SUBTENSOR_NETWORK or os.environ.get("SUBTENSOR_NETWORK", "local")
        # bt_config = config()
        if subtensor_network.lower() == "local":
            subtensor_network = os.environ.get("SUBTENSOR_CHAIN_ENDPOINT")  # bt_config.subtensor.chain_endpoint or
        else:
            subtensor_network = subtensor_network.lower()
        logger.info(f"Instantiating subtensor with network: {subtensor_network}")
        self._subtensor = Subtensor(network=subtensor_network)
        return self._subtensor

    def metagraph_force_sync(self) -> Metagraph:
        self._last_update_time = 0
        return self.METAGRAPH

    @property
    def METAGRAPH(self) -> Metagraph:
        if time.time() - self._last_update_time > 1200:
            try:
                logger.info(f"Fetching new METAGRAPH for NETUID={self.NETUID}")
                meta = self.SUBTENSOR.metagraph(netuid=self.NETUID)
                self._last_metagraph = meta
                self._last_update_time = time.time()
                return meta
            except Exception as e:
                logger.error(f"Failed to fetch new METAGRAPH for NETUID={self.NETUID}: {e}")
                if self._last_metagraph is not None:
                    logger.warning("Falling back to the previous METAGRAPH.")
                    return self._last_metagraph
                else:
                    logger.error("No previous METAGRAPH is available; re-raising exception.")
                    raise
        else:
            return self._last_metagraph

    @cached_property
    def UID(self) -> int:
        # TODO: Move chain-related stuff out of settings.
        return self.METAGRAPH.hotkeys.index(self.WALLET.hotkey.ss58_address)

    @property
    def block(self) -> int:
        # TODO: Move chain-related stuff out of settings.
        time_since_last_block = time.time() - self._block_sync_last_time
        if time_since_last_block > self._block_sync_interval:
            self._block = self.SUBTENSOR.get_current_block()
            self._block_sync_last_time = time.time()
            return self._block

        blocks_passed = time_since_last_block // 12
        return self._block + blocks_passed


try:
    shared_settings = SharedSettings.load(mode="validator" if is_cuda_available() else "mock")
except Exception as e:
    logger.exception(f"Error loading settings: {e}")
    shared_settings = None
