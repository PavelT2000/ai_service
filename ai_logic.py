import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import HTTPException, status
from google import genai
from google.genai import types

from schemas import EmbeddingRequest, ProxyRequest

logger = logging.getLogger("ai_service.logic")

load_dotenv()

proxy_url = os.getenv("PROXY_URL")
if os.getenv("USE_PROXY") == "True" and proxy_url:
    os.environ["HTTP_PROXY"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url
    logger.info("Using proxy: %s", proxy_url)

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY") or "",
    vertexai=False,
)

MODELS_FILE = Path(__file__).resolve().parent / "models_config.json"


def _load_models_config() -> Dict[str, Any]:
    with MODELS_FILE.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


MODELS_CONFIG = _load_models_config()
CHAT_MODELS_PRIORITY = MODELS_CONFIG["chat"]["priority"]
EMBEDDING_MODEL = MODELS_CONFIG["embedding"]["default"]
FALLBACK_DELAY_SECONDS = 1.5
_chat_cooldown_until = 0.0
_chat_cooldown_lock = threading.Lock()


def _extract_retry_seconds(error_message: str) -> int | None:
    float_retry_match = re.search(r"retry in (\d+(?:\.\d+)?)s", error_message, re.IGNORECASE)
    if float_retry_match:
        return max(1, int(float(float_retry_match.group(1))))

    int_retry_match = re.search(r"'retryDelay': '(\d+)s'", error_message)
    if int_retry_match:
        return max(1, int(int_retry_match.group(1)))

    return None


def _is_temporary_error(error_message: str) -> bool:
    lowered = error_message.lower()
    temporary_markers = (
        "resource_exhausted",
        "unavailable",
        "too many requests",
        "retry in",
        "retrydelay",
        "timed out",
        "timeout",
        "server disconnected",
    )
    return any(marker in lowered for marker in temporary_markers)


def _get_chat_cooldown_remaining() -> float:
    with _chat_cooldown_lock:
        remaining = _chat_cooldown_until - time.monotonic()
    return max(0.0, remaining)


def _set_chat_cooldown(seconds: int) -> None:
    if seconds <= 0:
        return
    target_time = time.monotonic() + seconds
    with _chat_cooldown_lock:
        global _chat_cooldown_until  # pylint: disable=global-statement
        _chat_cooldown_until = max(_chat_cooldown_until, target_time)


def _sleep_before_next_fallback(previous_model: str) -> None:
    logger.info(
        "Waiting %.1f sec before next fallback model after %s",
        FALLBACK_DELAY_SECONDS,
        previous_model,
    )
    time.sleep(FALLBACK_DELAY_SECONDS)


def _build_tool_declarations(raw_tools: list[dict[str, Any]] | None) -> list[types.Tool]:
    if not raw_tools:
        return []

    processed_tools: list[types.Tool] = []
    for tool_item in raw_tools:
        if not isinstance(tool_item, dict):
            logger.warning("Skipping invalid tool item (not an object): %r", tool_item)
            continue

        # Backward-compatible input: {"name": "...", "description": "...", "parameters": {...}}
        if "name" in tool_item:
            try:
                declaration = types.FunctionDeclaration(
                    name=str(tool_item["name"]),
                    description=str(tool_item.get("description", "")),
                    parameters=tool_item.get("parameters", {}),
                )
                processed_tools.append(types.Tool(function_declarations=[declaration]))
            except Exception as tool_error:  # pylint: disable=broad-exception-caught
                logger.warning("Skipping invalid flat tool declaration: %s", tool_error)
            continue

        # Newer input: {"function_declarations": [ ... ]}
        raw_declarations = tool_item.get("function_declarations")
        if isinstance(raw_declarations, list):
            declarations = []
            for raw_declaration in raw_declarations:
                if not isinstance(raw_declaration, dict):
                    continue
                if "name" not in raw_declaration:
                    continue
                try:
                    declarations.append(
                        types.FunctionDeclaration(
                            name=str(raw_declaration["name"]),
                            description=str(raw_declaration.get("description", "")),
                            parameters=raw_declaration.get("parameters", {}),
                        )
                    )
                except Exception as tool_error:  # pylint: disable=broad-exception-caught
                    logger.warning("Skipping invalid nested tool declaration: %s", tool_error)
            if declarations:
                processed_tools.append(types.Tool(function_declarations=declarations))
            continue

        logger.warning("Skipping unknown tool format: %r", tool_item)

    return processed_tools


def ask_gemini(request: ProxyRequest) -> Dict[str, Any]:
    config_params = request.model_dump(exclude={"contents", "tools"})
    processed_tools = _build_tool_declarations(request.tools)

    config = types.GenerateContentConfig(
        **config_params,
        tools=processed_tools if processed_tools else None,
    )

    cooldown_remaining = _get_chat_cooldown_remaining()
    if cooldown_remaining > 0:
        wait_seconds = max(1, int(cooldown_remaining))
        wait_message = (
            f"Сервис сейчас перегружен. Пожалуйста, подождите {wait_seconds} сек. "
            "и повторите запрос."
        )
        logger.warning(
            "Chat generation is cooling down for %.1f sec due to previous 429.",
            cooldown_remaining,
        )
        return {
            "answer": wait_message,
            "function_calls": None,
            "model_used": "none",
            "finish_reason": "RETRY_LATER",
        }

    attempt_details = []
    should_stop_fallbacks = False
    for model_id in CHAT_MODELS_PRIORITY:
        try:
            logger.info("Attempting model: %s", model_id)
            response = client.models.generate_content(
                model=model_id,
                contents=[c.model_dump() for c in request.contents],
                config=config,
            )

            answer_text = ""
            f_calls = []
            finish_reason = "UNKNOWN"

            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                finish_reason = candidate.finish_reason or "UNKNOWN"

                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.text:
                            answer_text += part.text
                        if part.function_call:
                            f_calls.append(
                                {
                                    "name": part.function_call.name,
                                    "args": part.function_call.args,
                                }
                            )

            logger.info("Success with %s. Finish reason: %s", model_id, finish_reason)
            if f_calls:
                logger.info("Model generated %s function calls", len(f_calls))

            return {
                "answer": answer_text or "Пустой ответ",
                "function_calls": f_calls if f_calls else None,
                "model_used": model_id,
                "finish_reason": str(finish_reason),
            }

        except Exception as e:  # pylint: disable=broad-exception-caught
            error_msg = str(e)
            logger.error("Error with model %s: %s", model_id, error_msg)
            attempt_details.append(
                {"model": model_id, "error": error_msg, "type": type(e).__name__}
            )

            retry_seconds = _extract_retry_seconds(error_msg)
            if retry_seconds is not None:
                # Respect provider retry window and block further outgoing chat requests.
                _set_chat_cooldown(retry_seconds)
                should_stop_fallbacks = True
                logger.warning(
                    "Applying chat cooldown for %s sec after model %s error.",
                    retry_seconds,
                    model_id,
                )

            if should_stop_fallbacks:
                break

            if _is_temporary_error(error_msg):
                _sleep_before_next_fallback(model_id)
            continue

    logger.critical("All models failed to respond: %s", attempt_details)

    temporary_errors = [
        item["error"] for item in attempt_details if _is_temporary_error(item["error"])
    ]
    retry_candidates = [
        _extract_retry_seconds(item["error"])
        for item in attempt_details
        if _is_temporary_error(item["error"])
    ]
    retry_seconds = max((seconds for seconds in retry_candidates if seconds is not None), default=None)

    if temporary_errors:
        wait_hint = f"{retry_seconds} сек." if retry_seconds is not None else "30-60 сек."
        wait_message = (
            f"Сервис сейчас перегружен. Пожалуйста, подождите {wait_hint} "
            "и повторите запрос."
        )
        logger.warning("Temporary failure, returning wait response: %s", wait_message)
        return {
            "answer": wait_message,
            "function_calls": None,
            "model_used": "none",
            "finish_reason": "RETRY_LATER",
        }

    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Service unavailable",
    )


def get_embedding(request: EmbeddingRequest) -> Dict[str, Any]:
    try:
        logger.info("Generating embedding for text length: %s", len(request.text))

        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=request.text,
            config=types.EmbedContentConfig(
                task_type=request.task_type,
                title=request.title,
            ),
        )
        if not result or not result.embeddings:
            logger.error("API returned empty embeddings")
            raise ValueError("Empty response from Gemini API")

        return {
            "embedding": result.embeddings[0].values,
            "model_used": EMBEDDING_MODEL,
        }
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Embedding error: %s", str(e))
        raise
