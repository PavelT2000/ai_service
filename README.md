# `ai_service`

A lightweight, production-ready FastAPI proxy and intelligent wrapper for Google's Gemini AI models. Designed for seamless integration into microservice architectures, it provides automatic model fallback, native function calling support, and dedicated embedding generation.

---

## 🌟 Features

| Feature | Description |
|---------|-------------|
| 🔄 **Multi-Model Fallback** | Automatically iterates through a prioritized list of Gemini models if the primary endpoint fails or times out. |
| 🛠️ **Function Calling** | Full support for tool declarations, argument extraction, and structured response parsing. |
| 📊 **Embedding Generation** | Dedicated `/api/embed` endpoint using `gemini-embedding-2-preview` for semantic search & RAG pipelines. |
| 🌐 **Proxy Support** | Configurable HTTP/HTTPS proxy routing for restricted or corporate network environments. |
| 📝 **Structured Logging** | Stdout-optimized, namespaced logging ready for `systemd`, Docker, or cloud log aggregators. |
| 🔒 **Type-Safe Contracts** | Pydantic v2 schemas ensure strict request/response validation and auto-generated OpenAPI docs. |

---

## 📁 Project Structure

```
ai_service/
├── .aiignore
├── .gitignore
├── ai_logic.py      # Core AI logic, fallback routing, function call parsing
├── main.py          # FastAPI application, routes, CORS, logging config
├── requirements.txt # Pinned Python dependencies
└── schemas.py       # Pydantic models for API request/response contracts
```

---

## 🛠️ Prerequisites

- Python `3.10+`
- Valid Google AI API Key ([Get one here](https://aistudio.google.com/app/apikey))
- (Optional) Proxy server credentials for restricted networks

---

## 📦 Installation

```bash
# 1. Clone & navigate
git clone <your-repo-url>
cd ai_service

# 2. Create & activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ⚙️ Configuration

Create a `.env` file in the project root:

```env
# Required: Google AI API Key
GOOGLE_API_KEY=your_api_key_here

# Optional: Proxy Configuration
USE_PROXY=False
PROXY_URL=http://proxy.example.com:8080
```

> 💡 **Note on Model IDs:** The `MODELS_PRIORITY` list in `ai_logic.py` contains placeholder/future model names. Update them to currently available Google AI model IDs (e.g., `gemini-1.5-flash`, `gemini-2.0-flash-lite`) before production deployment.

---

## 🌐 API Reference

Interactive documentation is available at `http://localhost:8001/docs` after startup.

### `GET /health`
Returns service status and listening port.

**Response:**
```json
{ "status": "ok", "port": 8001 }
```

---

### `POST /api/chat`
Generates a response using the Gemini API with support for conversation history, system instructions, and function calling.

**Request Payload:**
```json
{
  "contents": [
    {
      "role": "user",
      "parts": [{ "text": "What's the weather in Tokyo?" }]
    }
  ],
  "system_instruction": "You are a helpful assistant.",
  "tools": [
    {
      "name": "get_weather",
      "description": "Fetch current weather for a city",
      "parameters": {
        "type": "object",
        "properties": { "city": { "type": "string" } },
        "required": ["city"]
      }
    }
  ],
  "temperature": 0.7,
  "max_output_tokens": 1000
}
```

**Response:**
```json
{
  "answer": "I'll check the weather for Tokyo right now.",
  "function_calls": [
    { "name": "get_weather", "args": { "city": "Tokyo" } }
  ],
  "model_used": "models/gemini-1.5-flash",
  "finish_reason": "STOP"
}
```

---

### `POST /api/embed`
Generates a dense vector embedding for semantic search, clustering, or RAG indexing.

**Request Payload:**
```json
{
  "text": "Machine learning models require high-quality training data.",
  "task_type": "RETRIEVAL_DOCUMENT",
  "title": "ML Data Quality"
}
```

**Response:**
```json
{
  "embedding": [0.012, -0.045, 0.112, ...],
  "model_used": "gemini-embedding-2-preview"
}
```

---

## 🚀 Running the Service

### Development
```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### Production (Systemd / Docker)
```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --workers 2 --log-level info
```
> ⚠️ `reload=True` is disabled by default in `main.py` for production safety.

---

## 📊 Logging & Observability

The service uses Python's built-in `logging` module with a standardized format optimized for stdout streaming:

```
2024-05-20 14:32:10,123 [INFO] ai_service.main: Incoming request: 3 messages
2024-05-20 14:32:10,456 [INFO] ai_service.logic: Attempting model: models/gemini-1.5-flash
2024-05-20 14:32:11,789 [INFO] ai_service.logic: Success with models/gemini-1.5-flash. Finish reason: STOP
```

- **Namespaces:** `ai_service.main` (routing/middleware), `ai_service.logic` (AI execution)
- **Integration:** Drop-in compatible with `journalctl`, Docker logging drivers, Prometheus Loki, or CloudWatch.

---

## 🔧 Maintenance & Notes

| Area | Recommendation |
|------|----------------|
| **Model IDs** | Regularly verify `MODELS_PRIORITY` against [Google AI Model Garden](https://ai.google.dev/gemini-api/docs/models/gemini) |
| **Rate Limits** | Implement client-side retry/backoff if hitting `429 Too Many Requests` |
| **Security** | Never commit `.env` or expose `GOOGLE_API_KEY` in logs. Use secret managers in production. |
| **Scaling** | For high-throughput, deploy behind a reverse proxy (Nginx/Traefik) and scale workers via Gunicorn/Uvicorn. |

---

## 📄 License

[MIT License](LICENSE) *(Replace with your actual license)*

---

*Built with FastAPI, Pydantic v2, and the Google GenAI SDK.* 🤖✨