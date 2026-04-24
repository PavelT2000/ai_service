"""Точка входа в приложение FastAPI."""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ai_logic import ask_gemini, get_embedding
from logger_config import get_logger, setup_logging
from schemas import EmbeddingRequest, EmbeddingResponse, ProxyRequest, ProxyResponse

# Настраиваем логирование перед созданием приложения
setup_logging()
logger = get_logger("ai_service.main")

app = FastAPI(title="Independent AI Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Проверка доступности сервиса."""
    return {"status": "ok", "port": 8001}


@app.post("/api/chat", response_model=ProxyResponse)
async def proxy_chat(request: ProxyRequest):
    """Обработка чат-запросов."""
    logger.info("Chat request: %s messages", len(request.contents))
    result = ask_gemini(request)
    return ProxyResponse(**result)


@app.post("/api/embed", response_model=EmbeddingResponse)
async def proxy_embedding(request: EmbeddingRequest):
    """Генерация эмбеддингов."""
    logger.info("Embedding request received")
    result = get_embedding(request)
    return EmbeddingResponse(**result)


if __name__ == "__main__":
    # Запуск сервера
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)