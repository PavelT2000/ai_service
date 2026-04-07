import logging
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ai_logic import ask_gemini, get_embedding
from schemas import EmbeddingRequest, EmbeddingResponse, ProxyRequest, ProxyResponse

# Корневой логгер в stdout (удобно для контейнеров и сервисов).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
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
    logger.info("Incoming request: %s messages", len(request.contents))

    result = ask_gemini(request)

    return ProxyResponse(**result)


@app.post("/api/embed", response_model=EmbeddingResponse)
async def proxy_embedding(request: EmbeddingRequest):
    logger.info("Incoming embedding request")

    result = get_embedding(request)

    return EmbeddingResponse(
        embedding=result["embedding"],
        model_used=result["model_used"],
    )


if __name__ == "__main__":
    # Для прод-среды не используем авто-перезапуск.
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)
