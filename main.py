import uvicorn
import logging
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ai_logic import ask_gemini, get_embedding
from schemas import ProxyRequest, ProxyResponse, EmbeddingRequest, EmbeddingResponse

# Настройка корневого логирования для вывода в stdout (важно для systemd)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ai_service.main")

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
    return {"status": "ok", "port": 8001}

@app.post("/api/chat", response_model=ProxyResponse)
async def proxy_chat(request: ProxyRequest):
    # Логируем входящий запрос
    logger.info(f"Incoming request: {len(request.contents)} messages")
    
    result = ask_gemini(request)

    # Логируем краткий результат (первые 50 символов ответа)
    short_answer = result["answer"][:50].replace("\n", " ")
    logger.info(f"Sending response. Model: {result['model_used']}. Preview: {short_answer}...")

    return ProxyResponse(
        answer=result["answer"],
        function_calls=result["function_calls"],
        model_used=result["model_used"],
        finish_reason=result["finish_reason"]
    )
    
@app.post("/api/embed", response_model=EmbeddingResponse)
async def proxy_embedding(request: EmbeddingRequest):
    logger.info("Incoming embedding request")
    
    result = get_embedding(request)
    
    return EmbeddingResponse(
        embedding=result["embedding"],
        model_used=result["model_used"]
    )
    
if __name__ == "__main__":
    # reload=True на сервере лучше убрать, если это продакшн на Orange Pi
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)