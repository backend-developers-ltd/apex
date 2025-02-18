from loguru import logger

from prompting.llms.apis.gpt_wrapper import openai_client
from prompting.llms.apis.llm_messages import LLMMessages
from prompting.llms.apis.sn19_wrapper import chat_complete
from shared import settings

shared_settings = settings.shared_settings


class LLMWrapper:
    @staticmethod
    def chat_complete(
        messages: LLMMessages,
        model="chat-llama-3-1-70b",
        temperature=0.5,
        max_tokens=500,
        top_p=1,
        stream=False,
        logprobs=True,
    ) -> str:
        response: str | None = None
        if (
            shared_settings.SN19_API_KEY
            and shared_settings.SN19_API_URL
            and (model is None or "gpt" not in model.lower())
        ):
            try:
                response = chat_complete(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stream=stream,
                    logprobs=logprobs,
                )

            except Exception:
                logger.error(
                    "Failed to use SN19 API, falling back to GPT-3.5. "
                    "Make sure to specify 'SN19_API_KEY' and 'SN19_API_URL' in .env.validator"
                )

        if response is None:
            model = "gpt-3.5-turbo"
            response, _ = openai_client.chat_complete(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=stream,
                logprobs=logprobs,
            )
            response = response.choices[0].message.content
        return response
