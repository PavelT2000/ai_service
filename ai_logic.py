"""Модуль для взаимодействия с Google Gemini API."""
import os
from typing import Dict, Any
from google import genai
from google.genai import types
from dotenv import load_dotenv
from schemas import ProxyRequest

load_dotenv()

# Прокси
proxy_url = os.getenv("PROXY_URL")
if os.getenv("USE_PROXY") == "True" and proxy_url:
    os.environ['HTTP_PROXY'] = proxy_url
    os.environ['HTTPS_PROXY'] = proxy_url

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY") or "",
    vertexai=False
)

MODELS_PRIORITY = [
    'models/gemini-2.5-flash',      
    'models/gemini-2.0-flash-lite', 
    'models/gemini-1.5-flash'
]

def ask_gemini(request: ProxyRequest) -> Dict[str, Any]:
    """Отправляет запрос в Gemini и возвращает структурированный словарь с ответом."""
    config_params = request.model_dump(exclude={"contents"})
    config = types.GenerateContentConfig(**config_params)

    for model_id in MODELS_PRIORITY:
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=[c.model_dump() for c in request.contents],
                config=config
            )

            # Безопасно извлекаем текст и части сообщения
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
                            f_calls.append({
                                "name": part.function_call.name,
                                "args": part.function_call.args
                            })

            return {
                "answer": answer_text or "Пустой ответ",
                "function_calls": f_calls if f_calls else None,
                "model_used": model_id,
                "finish_reason": str(finish_reason)
            }

        except Exception as e:
            print(f"Ошибка на прокси-узле ({model_id}): {e}")
            continue

    return {
        "answer": "Error: All models failed",
        "model_used": "none",
        "finish_reason": "ERROR",
        "function_calls": None
    }