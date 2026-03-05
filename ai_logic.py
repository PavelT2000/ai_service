import os
from google import genai
from dotenv import load_dotenv

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

def ask_gemini(prompt: str):
    for model_id in MODELS_PRIORITY:
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt
            )
            # Добавляем проверку: если текст пустой или None, берем пустую строку
            text = response.text if response.text else "Модель вернула пустой ответ"
            return text, model_id
            
        except Exception as e:
            # ... ваш код обработки ошибок ...
            continue
    return "Все модели заняты", "none"