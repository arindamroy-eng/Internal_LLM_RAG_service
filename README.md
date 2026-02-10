# Internal LLM RAG Service

A self-hosted, multi-tenant LLM platform with per-user RAG (Retrieval-Augmented Generation) — designed for 1000+ users on a cluster of 8× AMD Instinct MI350X GPUs.

## Architecture

```
Users (OpenAI-compatible SDK)
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│           Agent Platform API (FastAPI) :8080                │
│  Auth │ Agent CRUD │ Chat │ Context Mgmt │ Doc Ingestion   │
└──────┬──────────┬──────────────────┬───────────────────────┘
       │          │                  │
 ┌─────▼────┐ ┌──▼──────┐  ┌───────▼────────┐
 │ Postgres │ │  Redis   │  │   Qdrant       │
 │  :5432   │ │  :6379   │  │   :6333        │
 └──────────┘ └─────────┘  └───────┬────────┘
                                    │
                        ┌───────────┘
                        ▼
               ┌────────────────┐
               │   LiteLLM      │  :4000
               │   Proxy        │
               └───────┬────────┘
                       │
           ┌───────────┼───────────┐
           ▼           ▼           ▼
        vLLM        vLLM        vLLM
        405B        Coder       Embed
        :8000       :8001       :8002
        GPU 0-5     GPU 6       GPU 7
```

## Hardware Requirements

- **8× AMD Instinct MI350X** (288 GB HBM3e each, 2,304 GB total)
- **ROCm 7.0+** with AITER support
- **CPU**: AMD EPYC recommended (64+ cores for handling 1000 users)
- **System RAM**: 256 GB+
- **Storage**: 2 TB+ NVMe for models, 1 TB+ for vector DB

## Quick Start

### 1. Clone and configure

```bash
git clone <your-repo-url>
cd Internal_LLM_RAG_service
cp .env.example .env
# Edit .env with your settings
```

### 2. Install ROCm and prepare GPU server

```bash
chmod +x scripts/*.sh
sudo ./scripts/setup_rocm.sh
```

### 3. Download models

```bash
# Install huggingface CLI
pip install huggingface-hub[cli]

# Download models (ensure you have access to gated models)
huggingface-cli download amd/Llama-3.1-405B-Instruct-MXFP4 --local-dir /models/llama-405b-mxfp4
huggingface-cli download Qwen/Qwen2.5-Coder-32B-Instruct --local-dir /models/qwen-coder-32b
huggingface-cli download BAAI/bge-large-en-v1.5 --local-dir /models/bge-large
```

### 4. Start the infrastructure stack

```bash
# Start Postgres, Redis, Qdrant, LiteLLM
docker compose up -d

# Run database migrations
docker compose exec app python -m app.db.migrations.run
```

### 5. Start vLLM serving on GPUs

```bash
./scripts/start_vllm.sh
```

### 6. Create user API keys

```bash
# Single user
./scripts/create_user_keys.sh alice

# Bulk from file
./scripts/create_user_keys.sh --bulk users.txt
```

### 7. Verify

```bash
# Health check
curl http://localhost:8080/health

# Test chat
curl -X POST http://localhost:8080/api/v1/agents/{agent_id}/chat \
  -H "Authorization: Bearer sk-alice-key" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, world!"}'
```

## User Guide

### Creating an Agent

```python
from openai import OpenAI

# Platform API client
import requests

API_BASE = "http://llm-gateway.internal:8080/api/v1"
headers = {"Authorization": "Bearer sk-alice-key"}

# Create an agent
agent = requests.post(f"{API_BASE}/agents", headers=headers, json={
    "name": "Research Assistant",
    "system_prompt": "You are a helpful research assistant...",
    "model": "gpt-4",
    "temperature": 0.7,
    "enable_rag": True
}).json()

# Upload documents to the agent
with open("paper.pdf", "rb") as f:
    requests.post(
        f"{API_BASE}/agents/{agent['id']}/documents",
        headers=headers,
        files={"file": f}
    )

# Chat with the agent (RAG-augmented)
response = requests.post(
    f"{API_BASE}/agents/{agent['id']}/chat",
    headers=headers,
    json={"message": "Summarize the key findings from the paper"}
).json()
```

### Direct OpenAI-Compatible Access

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://llm-gateway.internal:4000/v1",
    api_key="sk-alice-litellm-key"
)

# This hits the 405B model on GPUs 0-5
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Explain quantum entanglement"}]
)
```

## GPU Allocation

| GPUs | Model | Precision | Purpose | Port |
|------|-------|-----------|---------|------|
| 0-5 | Llama 3.1 405B | MXFP4 | Primary chat/reasoning | 8000 |
| 6 | Qwen 2.5 Coder 32B | FP8 | Code generation | 8001 |
| 7 | BGE-large-en-v1.5 | FP16 | Embeddings for RAG | 8002 |

## Configuration

All configuration is via environment variables (see `.env.example`).

## Project Structure

```
Internal_LLM_RAG_service/
├── app/
│   ├── main.py                 # FastAPI application entry
│   ├── config.py               # Settings / env config
│   ├── dependencies.py         # Dependency injection
│   ├── auth/
│   │   └── auth.py             # JWT + API key authentication
│   ├── models/
│   │   └── schemas.py          # Pydantic request/response models
│   ├── db/
│   │   ├── database.py         # Postgres connection + queries
│   │   └── migrations/
│   │       └── 001_initial.sql # Schema migration
│   ├── services/
│   │   ├── llm_client.py       # OpenAI-compatible LLM client
│   │   ├── context_manager.py  # Context window assembly
│   │   ├── message_cache.py    # Redis message caching
│   │   ├── vector_db.py        # Qdrant per-user vector DB
│   │   └── ingestion.py        # Document chunking + embedding
│   └── routers/
│       ├── agents.py           # Agent CRUD endpoints
│       ├── conversations.py    # Conversation management
│       ├── documents.py        # Document upload + management
│       └── chat.py             # Chat endpoint with streaming
├── scripts/
│   ├── setup_rocm.sh           # ROCm + driver installation
│   ├── start_vllm.sh           # Launch vLLM instances on GPUs
│   └── create_user_keys.sh     # User + API key provisioning
├── nginx/
│   └── nginx.conf              # Reverse proxy config
├── docker-compose.yml          # Full stack orchestration
├── litellm_config.yaml         # LiteLLM proxy routing config
├── Dockerfile                  # App container image
├── requirements.txt            # Python dependencies
├── .env.example                # Environment template
└── README.md
```

## License

Internal use only.
