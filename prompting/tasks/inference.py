import random
from typing import ClassVar

from loguru import logger
from pydantic import Field, model_validator

from prompting.datasets.sn13 import ChatEntry
from prompting.rewards.inference_reward_model import InferenceRewardModel
from prompting.rewards.reward import BaseRewardConfig, BaseRewardModel
from prompting.tasks.base_task import BaseTextTask
from shared import settings
from shared.docker_utils import get_generation

shared_settings = settings.shared_settings


class InferenceRewardConfig(BaseRewardConfig):
    reward_definitions: ClassVar[list[BaseRewardModel]] = [
        InferenceRewardModel(weight=1),
    ]


QUERY_PROMPT = """
Ask a question about the following text:

{website_content}

---

Ask a question about the text and nothing else:"""

SYSTEM_PROMPTS = [
    "",
    "You are a helpful AI assistant. Provide concise, accurate answers to any questions asked.",
    "You are a friendly and patient assistant. Communicate your responses in a clear, easy-to-understand way, ensuring the user feels supported.",
    "You are a creative helper. Offer engaging, imaginative responses that keep the user interested, while maintaining accuracy and clarity.",
]


class InferenceTask(BaseTextTask):
    name: ClassVar[str] = "inference"
    # TODO: Once we want to enable the 'actual' inference task with exact models
    query: str | list = []
    reference: str | None = None
    system_prompt: str | None = None
    llm_model_id: str | None = Field(default_factory=lambda: random.choice(settings.shared_settings.LLM_MODEL))
    seed: int = Field(default_factory=lambda: random.randint(0, 1_000_000), allow_mutation=False)
    sampling_params: dict[str, float] = shared_settings.SAMPLING_PARAMS.copy()
    messages: list[dict] | None = None
    timeout: int = shared_settings.INFERENCE_TIMEOUT

    @model_validator(mode="after")
    def random_llm_model_id(self):
        if self.query:  # If we are already defining query, as in the case of organics, we also specify model.
            return self
        # self.sampling_params["temperature"] = random.randint(1, 10) / 10
        # self.sampling_params["max_new_tokens"] = random.choice([256, 512, 1024, 2048])
        return self

    async def make_query(self, dataset_entry: ChatEntry) -> str:
        if self.query:
            return self.query
        system_prompt = random.choice(SYSTEM_PROMPTS)
        system_prompt = [{"role": "system", "content": system_prompt}] if system_prompt else []
        self.messages = system_prompt + dataset_entry.messages
        self.query = self.messages

        return self.query

    async def make_reference(self, dataset_entry: ChatEntry) -> str:
        # With logits scoring there is no reference, and instead we need to generate the logits based
        # on the miner's completions.
        logger.info(f"self.llm_model: {self.llm_model}")
        logger.info(f"self.llm_model_id: {self.llm_model_id}")
        if self.organic or self.llm_model_id:
            self.reference = ""
            return self.reference

        self.reference = await get_generation(
            messages=self.messages,
            model=self.llm_model_id,
            seed=self.seed,
            sampling_params=self.sampling_params,
        )
        return self.reference
