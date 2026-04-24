"""Бизнес-логика взаимодействия с Google Gemini API."""
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import HTTPException
from google import genai
from google.genai import types

from logger_config import get_logger
from schemas import EmbeddingRequest, ProxyRequest

logger = get_logger("ai_service.logic")
load_dotenv()

# Настройка прокси
PROXY_URL = os.getenv("PROXY_URL")
if os.getenv("USE_PROXY") == "True" and PROXY_URL:
    os.environ["HTTP_PROXY"] = PROXY_URL
    os.environ["HTTPS_PROXY"] = PROXY_URL
    logger.info("Using proxy: %s", PROXY_URL)

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY") or "", vertexai=False)

# Конфигурация моделей
MODELS_FILE = Path(__file__).resolve().parent / "models_config.json"

def _load_models_config() -> Dict[str, Any]:
    with MODELS_FILE.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)

MODELS_CONFIG = _load_models_config()
CHAT_MODELS_PRIORITY = MODELS_CONFIG["chat"]["priority"]
EMBEDDING_MODEL = MODELS_CONFIG["embedding"]["default"]
FALLBACK_DELAY_SECONDS = 1.5

# Состояние кулдауна (используем UPPER_CASE для соответствия Pylint)
CHAT_COOLDOWN_UNTIL = 0.0
COOLDOWN_LOCK = threading.Lock()

def _extract_retry_seconds(error_message: str) -> Optional[int]:
    """Извлекает время ожидания из ошибки API."""
    float_match = re.search(r"retry in (\d+(?:\.\d+)?)s", error_message, re.I)
    if float_match:
        return max(1, int(float(float_match.group(1))))
    return None

def _is_temporary_error(error_message: str) -> bool:
    """Определяет, является ли ошибка временной."""
    lowered = error_message.lower()
    markers = ("resource_exhausted", "unavailable", "too many requests", "timeout")
    return any(m in lowered for m in markers)

def _build_tool_declarations(raw_tools: Optional[List[Dict[str, Any]]]) -> List[types.Tool]:
    """Безопасно преобразует описание инструментов в объекты SDK."""
    if not raw_tools:
        return []
    processed = []
    for tool in raw_tools:
        if isinstance(tool, dict) and "name" in tool:
            try:
                declaration = types.FunctionDeclaration(
                    name=str(tool["name"]),
                    description=str(tool.get("description", "")),
                    parameters=tool.get("parameters", {}),
                )
                processed.append(types.Tool(function_declarations=[declaration]))
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Invalid tool format: %s", exc)
    return processed

def ask_gemini(request: ProxyRequest) -> Dict[str, Any]:
    """Основной цикл генерации контента с перебором моделей."""
    global CHAT_COOLDOWN_UNTIL  # Объявление в самом начале для Pylint

    config = types.GenerateContentConfig(
        **request.model_dump(exclude={"contents", "tools"}),
        tools=_build_tool_declarations(request.tools) or None,
    )

    with COOLDOWN_LOCK:
        remaining = CHAT_COOLDOWN_UNTIL - time.monotonic()

    if remaining > 0:
        return {
            "answer": f"Сервис перегружен. Подождите {int(remaining)} сек.",
            "function_calls": None,
            "model_used": "none",
            "finish_reason": "RETRY_LATER",
        }

    attempt_details = []
    for model_id in CHAT_MODELS_PRIORITY:
        try:
            logger.info("Attempting model: %s", model_id)
            response = client.models.generate_content(
                model=model_id,
                contents=[c.model_dump() for c in request.contents],
                config=config,
            )

            answer_text, f_calls, finish_reason = "", [], "UNKNOWN"

            if response.candidates:
                candidate = response.candidates[0]
                finish_reason = str(candidate.finish_reason or "UNKNOWN")
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.text:
                            answer_text += part.text
                        if part.function_call:
                            f_calls.append({
                                "name": part.function_call.name,
                                "args": part.function_call.args
                            })

            return {
                "answer": answer_text or "Пустой ответ",
                "function_calls": f_calls if f_calls else None,
                "model_used": model_id,
                "finish_reason": finish_reason,
            }

        except Exception as exc:  # pylint: disable=broad-exception-caught
            err_msg = str(exc)
            logger.error("Error with %s: %s", model_id, err_msg)
            attempt_details.append({"model": model_id, "error": err_msg})

            retry_sec = _extract_retry_seconds(err_msg)
            if retry_sec:
                with COOLDOWN_LOCK:
                    CHAT_COOLDOWN_UNTIL = time.monotonic() + retry_sec
                break

            if _is_temporary_error(err_msg):
                time.sleep(FALLBACK_DELAY_SECONDS)
            continue

    raise HTTPException(
        status_code=503,
        detail={"status": "error", "attempts": attempt_details}
    )

def get_embedding(request: EmbeddingRequest) -> Dict[str, Any]:
    """Генерация эмбеддингов."""
    try:
        logger.info("Generating embedding for text length: %s", len(request.text))
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=request.text,
            config=types.EmbedContentConfig(task_type=request.task_type, title=request.title),
        )
        if not result or not result.embeddings:
            raise ValueError("Empty embeddings from API")
        return {"embedding": result.embeddings[0].values, "model_used": EMBEDDING_MODEL}
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Embedding error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc