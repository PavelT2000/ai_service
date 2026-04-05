# 🤖 AI Service

A lightweight, production-ready FastAPI proxy service for Google Gemini AI models. Designed for edge deployment (e.g., Orange Pi), it provides intelligent model fallback, native function calling support, and text embedding capabilities.

## ✨ Features
- 🚀 **FastAPI REST API** with automatic OpenAPI/Swagger documentation
- 🔄 **Smart Model Fallback** – Automatically retries with alternative Gemini models if the primary fails
- 🛠️ **Function Calling** – Full support for tool declarations and execution routing
- 📐 **Embeddings API** – Generate vector embeddings for RAG or semantic search workflows
- 🌐 **Proxy & Env Config** – Flexible `.env` configuration for API keys and network proxies
- 📦 **CI/CD Ready** – GitHub Actions workflow for automated deployment to Orange Pi via Tailscale & SSH
- ❤️ **Health & CORS** – Built-in `/health` endpoint and permissive CORS for frontend integration

## 📦 Installation

### Prerequisites
- Python 3.10+
- `pip` or `uv`
- A valid [Google AI API Key](https://aistudio.google.com/app/apikey)

### Local Setup
```bash
# 1. Clone the repository
git clone https://github.com/your-username/ai_service.git
cd ai_service

# 2. Create & activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
touch .env
```

## ⚙️ Configuration

Create a `.env` file in the project root:

```env
# Required
GOOGLE_API_KEY=your_google_ai_api_key_here

# Optional: Network Proxy
USE_PROXY=False
PROXY_URL=http://your-proxy-server:port
```

> 💡 **Note:** Model fallback order is defined in `ai_logic.py` (`MODELS_PRIORITY`). Adjust the list based on your quota limits or model availability.

## 🚀 Usage

### Start the Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```
Interactive API documentation will be available at: `http://localhost:8001/docs`

### API Endpoints

#### 1. Health Check
```bash
curl http://localhost:8001/health
# Response: {"status": "ok", "port": 8001}
```

#### 2. Chat / Text Generation (`POST /api/chat`)
Supports conversation history, system instructions, temperature, and function calling.

```bash
curl -X POST http://localhost:8001/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [
      {"role": "user", "parts": [{"text": "What is the capital of France?"}]}
    ],
    "temperature": 0.7,
    "max_output_tokens": 500
  }'
```

#### 3. Embeddings (`POST /api/embed`)
Generates vector embeddings for semantic search or RAG pipelines.

```bash
curl -X POST http://localhost:8001/api/embed \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Machine learning enables computers to learn from data.",
    "task_type": "RETRIEVAL_QUERY"
  }'
```

## 🌍 Deployment (Orange Pi + GitHub Actions)

This project includes a production-ready CI/CD pipeline that deploys to an Orange Pi via **Tailscale** and **SSH**.

### Prerequisites for Deployment
1. **Target Machine:** Orange Pi (or similar Linux SBC) with `git`, `python3`, `pip`, and `systemd` installed.
2. **Initial Setup:** Clone the repo to `~/My/ai_service` on the target machine and create a systemd service (`ai-service.service`).
3. **GitHub Secrets:** Configure the following in your repository settings:
   - `TAILSCALE_AUTH_KEY`
   - `SERVER_IP` (Tailscale IP)
   - `SERVER_USER`
   - `SSH_PRIVATE_KEY`

### How it Works
On every push to `main`, the workflow:
1. Connects to your private network via Tailscale
2. SSHs into the Orange Pi
3. Pulls the latest code (`git reset --hard origin/main`)
4. Installs/updates Python dependencies
5. Restarts the `ai-service.service` systemd unit
6. Verifies service status

> 📝 **Systemd Service Example** (`/etc/systemd/system/ai-service.service`):
> ```ini
> [Unit]
> Description=AI Service Proxy
> After=network.target
>
> [Service]
> Type=simple
> User=your_user
> WorkingDirectory=/home/your_user/My/ai_service
> ExecStart=/home/your_user/My/ai_service/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8001
> Restart=always
> Environment=PATH=/home/your_user/My/ai_service/.venv/bin
>
> [Install]
> WantedBy=multi-user.target
> ```

## 📁 Project Structure
```
ai_service/
├── .github/workflows/deploy.yml  # CI/CD pipeline for Orange Pi
├── ai_logic.py                   # Core Gemini API logic & fallback routing
├── main.py                       # FastAPI application & route definitions
├── schemas.py                    # Pydantic request/response models
├── requirements.txt              # Pinned Python dependencies
├── .env                          # Environment variables (gitignored)
└── .gitignore                    # Standard Python/FastAPI ignore rules
```

## 📄 License
[MIT License](LICENSE) *(Replace with your actual license)*