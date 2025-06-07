import json
import re
import traceback
from functools import wraps
from typing import Any

from fastapi.responses import StreamingResponse
from loguru import logger


def parse_llm_json(json_str: str, allow_empty: bool = True) -> dict[str, Any]:
    """Parse JSON output from LLM that may contain code blocks, newlines and other formatting.

    Extracts JSON from code blocks if present, or finds JSON objects/arrays within text.

    Args:
        json_str (str): The JSON string to parse.
        allow_empty (bool): Whether to allow empty JSON objects.

    Returns:
        dict: The parsed JSON object
    """
    # First try to extract JSON from code blocks if they exist.
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    code_block_matches = re.findall(code_block_pattern, json_str)
    if code_block_matches:
        # Use the first code block found.
        json_str = code_block_matches[0]
    else:
        # Try to find JSON objects or arrays within the string.
        json_candidates = []

        # Look for JSON objects {...}.
        brace_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        object_matches = re.findall(brace_pattern, json_str)
        json_candidates.extend(object_matches)

        # Look for JSON arrays [...].
        bracket_pattern = r"\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]"
        array_matches = re.findall(bracket_pattern, json_str)
        json_candidates.extend(array_matches)

        # Try to parse each candidate and use the first valid one.
        for candidate in json_candidates:
            try:
                candidate = candidate.strip()
                json.loads(candidate)
                json_str = candidate
                break
            except json.JSONDecodeError:
                continue
        else:
            # If no valid JSON found in candidates, try the original string.
            pass

    # Replace escaped newlines with actual newlines.
    json_str = json_str.replace("\\n", "\n")

    # Remove any redundant newlines/whitespace while preserving content.
    json_str = " ".join(line.strip() for line in json_str.splitlines())

    # Parse the cleaned JSON string.
    result = json.loads(json_str)

    if not allow_empty and not result:
        raise json.JSONDecodeError("Empty JSON string", json_str, 0)

    return result


def with_retries(max_retries: int = 3):
    """
    A decorator that retries a function on failure and logs attempts using loguru.

    Args:
        max_retries (int): Maximum number of retry attempts before giving up
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Get the full stack trace
                    stack_trace = traceback.format_exc()
                    # If this is the last attempt, log as critical with full stack trace
                    if attempt == max_retries - 1:
                        logger.exception(
                            f"Function '{func.__name__}' failed on final attempt {attempt + 1}/{max_retries}. "
                            f"Error: {str(e)}\nStack trace:\n{stack_trace}"
                        )
                        raise  # Re-raise the exception after logging
                    # Otherwise log as error without stack trace
                    logger.error(
                        f"Function '{func.__name__}' failed on attempt {attempt + 1}/{max_retries}. "
                        f"Error: {str(e)}. Retrying..."
                    )
            return None  # In case all retries fail

        return wrapper

    return decorator


def convert_to_gemma_messages(messages):
    """Convert a list of messages to a list of gemma messages by alternating roles and adding empty messages."""
    gemma_messages = []
    for message in messages:
        if gemma_messages and gemma_messages[-1]["role"] == message["role"]:
            # Gemma requires alternating roles, so we need to add an empty message with the opposite role
            gemma_messages.append(
                {"type": "text", "content": "", "role": "assistant" if message["role"] == "user" else "user"}
            )
        gemma_messages.append({"type": "text", "role": message["role"], "content": message["content"]})
    return gemma_messages


async def extract_content_from_stream(streaming_response: StreamingResponse) -> str:
    full_content = ""

    async for chunk in streaming_response.body_iterator:
        # Decode bytes to string if necessary
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8")

        # Remove any 'data: ' prefixes and skip empty lines
        for line in chunk.splitlines():
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue

            try:
                data = json.loads(line.removeprefix("data:").strip())
                delta = data.get("choices", [{}])[0].get("delta", {})
                content_piece = delta.get("content")
                if content_piece:
                    full_content += content_piece
            except json.JSONDecodeError:
                continue  # Optionally log/handle malformed chunks

    return full_content
