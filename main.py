import uvicorn
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from ai_logic import ask_gemini
from schemas import UserRequest, AIResponse

app = FastAPI(title="Independent AI Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # В продакшене лучше указать конкретный домен
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "ok", "port": 8001}

@app.post("/api/chat", response_model=AIResponse)
async def chat_with_ai(request: UserRequest):
    print(f"--- AI Service Log ---")
    print(f"Prompt: {request.prompt[:50]}...")
    ai_text, model_used = ask_gemini(request)
    return AIResponse(answer=ai_text, model=model_used)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)