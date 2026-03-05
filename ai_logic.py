import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from schemas import UserRequest

load_dotenv()

proxy_url = os.getenv("PROXY_URL")
if os.getenv("USE_PROXY") == "True" and proxy_url:
    # Теперь Pylance знает, что proxy_url здесь точно не None
    os.environ['HTTP_PROXY'] = proxy_url
    os.environ['HTTPS_PROXY'] = proxy_url
    
# Важно: переменные окружения должны быть установлены ДО создания клиента
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY"),
    vertexai=False
)

MODELS_PRIORITY = [
    'models/gemini-2.5-flash',      
    'models/gemini-2.0-flash-lite', 
    'models/gemini-1.5-flash'       
]

def ask_gemini(request: UserRequest):
    """Отправляет запрос к Gemini с учетом настроек безопасности и параметров."""
    # Приводим строковые значения из request к объектам Enum библиотеки
    config = types.GenerateContentConfig(
        system_instruction=request.system_instruction,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        max_output_tokens=request.max_output_tokens,
        safety_settings=[
            types.SafetySetting(
                category="HATE_SPEECH",  # type: ignore
                threshold=request.hate_threshold or "BLOCK_MEDIUM_AND_ABOVE" # type: ignore
            ),
            types.SafetySetting(
                category="HARASSMENT",  # type: ignore
                threshold=request.harassment_threshold or "BLOCK_MEDIUM_AND_ABOVE" # type: ignore
            ),
            types.SafetySetting(
                category="DANGEROUS_CONTENT",  # type: ignore
                threshold="BLOCK_MEDIUM_AND_ABOVE" # type: ignore
            ),
            types.SafetySetting(
                category="SEXUALLY_EXPLICIT",  # type: ignore
                threshold="BLOCK_MEDIUM_AND_ABOVE" # type: ignore
            ),
        ]
    )

    for model_id in MODELS_PRIORITY:
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=request.prompt,
                config=config
            )
            text = response.text if response.text else "Пустой ответ"
            return text, model_id
        except Exception as e:
            print(f"Ошибка модели {model_id}: {e}")
            continue
    return "Все модели заняты", "none"