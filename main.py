"""Основной сервис FastAPI для проксирования запросов к ИИ."""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ai_logic import ask_gemini
from schemas import ProxyRequest, ProxyResponse

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
    """Проксирует запрос к Gemini и возвращает результат."""
    print(f"--- Proxying request: {len(request.contents)} messages ---")

    result = ask_gemini(request)

    return ProxyResponse(
        answer=result["answer"],
        function_calls=result["function_calls"],
        model_used=result["model_used"],
        finish_reason=result["finish_reason"]
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)