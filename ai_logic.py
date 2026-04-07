import json
import logging
import os
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


def ask_gemini(request: ProxyRequest) -> Dict[str, Any]:
    config_params = request.model_dump(exclude={"contents", "tools"})

    processed_tools = []
    if request.tools:
        for tool_dict in request.tools:
            processed_tools.append(
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=tool_dict["name"],
                            description=tool_dict.get("description", ""),
                            parameters=tool_dict.get("parameters", {}),
                        ),
                    ]
                )
            )

    config = types.GenerateContentConfig(
        **config_params,
        tools=processed_tools if processed_tools else None,
    )

    attempt_details = []
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
            continue

    logger.critical("All models failed to respond: %s", attempt_details)
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