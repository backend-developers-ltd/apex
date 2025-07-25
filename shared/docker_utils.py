from typing import Any

import requests
from loguru import logger

from shared import constants


async def get_generation(
    messages: list[str] | list[dict],
    roles: list[str] | None = None,
    model: str | None = None,
    seed: int = None,
    sampling_params: dict[str, float] = None,
) -> str:
    if messages and isinstance(messages[0], dict):
        dict_messages = messages
    else:
        dict_messages = [
            {"content": message, "role": role} for message, role in zip(messages, roles or ["user"] * len(messages))
        ]
    url = f"{constants.DOCKER_BASE_URL}/v1/chat/generate"
    headers = {"Content-Type": "application/json"}
    payload = {"messages": dict_messages, "seed": seed, "sampling_params": sampling_params}
    response = requests.post(url, headers=headers, json=payload)
    try:
        json_response = response.json()
        logger.info(f"Response: {json_response}")
        return json_response["choices"][0]["message"]["content"]
    except requests.exceptions.JSONDecodeError:
        logger.error(f"Error generating response. Status: {response.status_code}, Body: {response.text}")
        return ""


async def get_logits(
    messages: list[str],
    model: None = None,
    sampling_params: dict[str, float] = None,
    seed: int = None,
    continue_last_message: bool = False,
    top_logprobs: int = 10,
) -> dict[str, Any] | None:
    try:
        url = f"{constants.DOCKER_BASE_URL}/v1/chat/generate_logits"
        headers = {"Content-Type": "application/json"}
        payload = {
            "messages": messages,
            "seed": seed,
            "sampling_params": sampling_params,
            "top_logprobs": top_logprobs,
            "continue_last_message": continue_last_message,
        }
        response = requests.post(url, headers=headers, json=payload)
        json_response = response.json()
        return json_response
    except BaseException as exc:
        logger.error(f"Error generating logits: {exc}")
        return None


def get_embeddings(inputs):
    """
    Sends a POST request to the local embeddings endpoint and returns the response.

    Args:
        inputs (str or list of str): A single input string or a list of input strings to embed.

    Returns:
        dict: JSON response from the embeddings server.
    """
    if isinstance(inputs, str):
        inputs = [inputs]  # convert single string to list

    url = f"{constants.DOCKER_BASE_URL}/v1/embeddings"
    headers = {"Content-Type": "application/json"}
    payload = {"input": inputs}

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e)}


if __name__ == "__main__":
    print(get_embeddings("Hello, world!"))
