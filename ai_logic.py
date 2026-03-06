import os
import logging
from typing import Dict, Any
from google import genai
from google.genai import types
from dotenv import load_dotenv
from schemas import ProxyRequest

# Настройка логера для этого модуля
logger = logging.getLogger("ai_service.logic")

load_dotenv()

# Прокси
proxy_url = os.getenv("PROXY_URL")
if os.getenv("USE_PROXY") == "True" and proxy_url:
    os.environ['HTTP_PROXY'] = proxy_url
    os.environ['HTTPS_PROXY'] = proxy_url
    logger.info(f"Using proxy: {proxy_url}")

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY") or "",
    vertexai=False
)

# Исправлен список (gemini-2.5 пока не существует)
MODELS_PRIORITY = [
    'models/gemini-2.5-flash',      
    'models/gemini-2.0-flash-lite', 
    'models/gemini-1.5-flash'
]

def ask_gemini(request: ProxyRequest) -> Dict[str, Any]:
    # 1. Извлекаем базовые параметры (temperature, tokens и т.д.)
    config_params = request.model_dump(exclude={"contents", "tools"})
    
    # 2. Обрабатываем инструменты (tools), если они есть
    processed_tools = []
    if request.tools:
        for tool_dict in request.tools:
            # Превращаем словарь в объект декларации функции
            # Gemini SDK ожидает, что функция лежит внутри объекта Tool как declaration
            processed_tools.append(
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=tool_dict["name"],
                            description=tool_dict.get("description", ""),
                            parameters=tool_dict.get("parameters", {})
                        )
                    ]
                )
            )
    
    # 3. Собираем конфиг
    config = types.GenerateContentConfig(
        **config_params,
        tools=processed_tools if processed_tools else None
    )

    for model_id in MODELS_PRIORITY:
        try:
            logger.info(f"Attempting model: {model_id}")
            response = client.models.generate_content(
                model=model_id,
                contents=[c.model_dump() for c in request.contents],
                config=config
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
                            f_calls.append({
                                "name": part.function_call.name,
                                "args": part.function_call.args
                            })
            
            logger.info(f"Success with {model_id}. Finish reason: {finish_reason}")
            if f_calls:
                logger.info(f"Model generated {len(f_calls)} function calls")

            return {
                "answer": answer_text or "Пустой ответ",
                "function_calls": f_calls if f_calls else None,
                "model_used": model_id,
                "finish_reason": str(finish_reason)
            }

        except Exception as e:
            logger.error(f"Error with model {model_id}: {str(e)}")
            continue

    logger.critical("All models failed to respond!")
    return {
        "answer": "Error: All models failed",
        "model_used": "none",
        "finish_reason": "ERROR",
        "function_calls": None
    }