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
               │   LiteLLM      │  :4000   ← decides WHICH MODEL
               │   Proxy        │            (aliases, keys, budgets)
               └───────┬────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
 ┌─────────────┐    vLLM           vLLM
 │ llm-d Envoy │    Coder          Embed
 │    :8081    │    :8001          :8002
 │      │      │    GPU 6          GPU 7
 │   EPP:9002  │  ← decides WHICH REPLICA
 └──────┬──────┘    (prefix-cache + load aware)
        │
  ┌─────┼─────┐
  ▼     ▼     ▼
 vLLM  vLLM  vLLM        405B MXFP4, TP=2 each
 :8000 :8010 :8020       GPUs 0,1 / 2,3 / 4,5
```

### Two layers of routing

The split matters, because they are different problems:

| Layer | Question | Owner |
|---|---|---|
| Model selection | *Which model serves this request?* | LiteLLM alias, from `agents.model` |
| Replica selection | *Which replica of that model?* | llm-d Endpoint Picker |

**llm-d does not route between different models.** It picks among replicas of
one model, using prefix-cache locality and queue/KV pressure. There is no
"send code questions to the coder model" capability in llm-d — that decision
stays where it already was.

llm-d is optional and gated. Bring it up only after measuring; see
`docs/` and the gates below.

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

> **The model names are aliases for self-hosted models — no OpenAI model is
> ever called.** They keep the OpenAI SDKs working as a drop-in, but the name
> tells you nothing about what actually serves the request:

| Alias you request | What actually serves it | GPUs |
|---|---|---|
| `gpt-4`, `gpt-4-turbo`, `gpt-4o` | Llama 3.1 405B Instruct (MXFP4) | 0–5 |
| `gpt-3.5-turbo` | **Qwen 2.5 Coder 32B** — a *code* model, despite the name | 6 |
| `text-embedding-ada-002`, `text-embedding-3-small` | BGE-large-en-v1.5 | 7 |

> Pick the alias by the **backend you want**, not by what the name suggests.
> An agent doing code generation should set `"model": "gpt-3.5-turbo"` even
> though that name reads like a downgrade.
>
> Unknown aliases are rejected at agent-creation time with a `422` listing the
> valid options. The accepted set is derived from `CHAT_MODEL_ALIAS`,
> `CODE_MODEL_ALIAS` and `EXTRA_AGENT_MODEL_ALIASES`, and **must be kept in
> sync with the `model_list` in `litellm_config.yaml`** — a name accepted here
> but absent there would still fail later as a 502.

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
    "model": "gpt-4",          # alias -> Llama 3.1 405B (see table above)
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

# "gpt-4" resolves to Llama 3.1 405B. With the sharded layout this is a
# POOL of three TP=2 replicas on GPUs 0-5, not a single server — LiteLLM
# (or llm-d, if enabled) picks which replica serves the request.
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Explain quantum entanglement"}]
)

# Code generation goes to Qwen 2.5 Coder, addressed as "gpt-3.5-turbo".
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Write a binary search in Rust"}]
)
```

## GPU Allocation

Production layout — `scripts/start_vllm_sharded.sh`:

| GPUs | Model | Precision | TP | Purpose | Port |
|------|-------|-----------|----|---------|------|
| 0,1 | Llama 3.1 405B | MXFP4 | 2 | Chat replica 0 | 8000 |
| 2,3 | Llama 3.1 405B | MXFP4 | 2 | Chat replica 1 | 8010 |
| 4,5 | Llama 3.1 405B | MXFP4 | 2 | Chat replica 2 | 8020 |
| 6 | Qwen 2.5 Coder 32B | FP8 | 1 | Code generation | 8001 |
| 7 | BGE-large-en-v1.5 | FP16 | 1 | Embeddings for RAG | 8002 |

Single-replica fallback — `scripts/start_vllm.sh`: 405B at TP=4 on GPUs 0-3,
GPUs 4-5 idle.

> **Tensor-parallel size is constrained.** Llama 3.1 405B has 128 attention
> heads, **8 KV heads** and `intermediate_size` 53248. vLLM requires TP to
> divide all three, so only **TP ∈ {2, 4, 8}** are valid. An earlier config
> used TP=6, which fails on two of the three — that server could not start.

> **Replication is not free.** Three copies of the weights (~223 GB each)
> cost roughly 34% of aggregate KV capacity versus one large shard, and
> per-token decode for a *single* request is ~3× slower at TP=2 than at TP=6
> because each GPU streams a larger shard per token. This layout wins on
> aggregate throughput under concurrency and loses at low concurrency.
> Measure before adopting it.

## Enabling llm-d replica routing

llm-d is **off by default** (Compose profile `llmd`). It is worth installing
only if requests actually share prompt prefixes — its documented benefit on
low-prefix-sharing traffic is approximately zero.

### Prerequisites

1. **A replica pool.** `./scripts/start_vllm_sharded.sh` — with one replica
   there is nothing to pick between.
2. **A reusable prefix.** The context builder places the volatile RAG block
   *after* conversation history precisely so a stable prefix exists. Verify
   with `pytest tests/test_context_prefix_stability.py`.

### Gates

Do not skip these; each one can end the project cheaply.

| Gate | Check | If it fails |
|---|---|---|
| **A** | Chat prefix-cache hit rate ≥ 0.25 under a real multi-turn replay | Stop. llm-d has nothing to route on. |
| **B** | Does `litellm_config.roundrobin.yaml` already meet the SLO? | Stop. Plain least-busy balancing is free. |
| **C** | llm-d beats round-robin by ≥20% p95 TTFT at equal throughput | Roll back to round-robin. |

Measure with a replay of **real conversations**, not synthetic prompts —
random prompts have zero prefix sharing by construction and will pre-decide
Gate A against llm-d.

### Start

```bash
./scripts/start_vllm_sharded.sh          # 3 chat replicas
docker compose --profile llmd up -d      # epp + envoy
curl -s localhost:19000/clusters | grep ext_proc   # expect health_flags::healthy
```

Then point the chat aliases at Envoy in `litellm_config.yaml`
(`api_base: http://host.docker.internal:8081/v1`) and restart LiteLLM.

### Rollback

```bash
cp litellm_config.roundrobin.yaml litellm_config.yaml   # bypass llm-d  (~10s)
docker compose restart litellm
docker compose --profile llmd down                      # remove llm-d   (~5s)
./scripts/start_vllm.sh                                 # single replica (~15min)
```

To drain one replica without restarting the EPP, remove its entry from
`llmd/epp/endpoints.yaml` by **atomic rename** (`mv tmp endpoints.yaml`);
`watchFile: true` reloads it live.

## Observability

Prometheus on `:9091`, Grafana on `:3000`. Key metrics for the gates:

| Signal | Metric |
|---|---|
| Prefix cache hit rate | `vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total` |
| Queue depth | `vllm:num_requests_waiting` |
| TTFT | `vllm:time_to_first_token_seconds` |
| KV pressure | `vllm:gpu_cache_usage_perc` |
| **Preemption thrash** | `vllm:num_preemptions_total` |

Check metric names against your image first — vLLM V0 exposed a
`vllm:gpu_prefix_cache_hit_rate` gauge where V1 uses counters:

```bash
curl -s localhost:8000/metrics | grep -i prefix
```

End-to-end TTFT (including context assembly and the LiteLLM/Envoy hops, which
vLLM cannot see) is logged by the app as `ttft_seconds=`.

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
│   ├── start_vllm.sh           # Single-replica fallback (405B TP=4)
│   ├── start_vllm_sharded.sh   # Production: 3x 405B TP=2 chat pool
│   └── create_user_keys.sh     # User + API key provisioning
├── llmd/                       # llm-d replica router (Compose profile: llmd)
│   ├── epp/config.yaml         # Endpoint Picker: scorers + weights
│   ├── epp/endpoints.yaml      # Chat replica inventory (hot-reloadable)
│   └── envoy/envoy.yaml        # ext_proc proxy in front of the EPP
├── monitoring/
│   └── prometheus.yml          # Scrape config for vLLM / LiteLLM / EPP
├── tests/
│   └── test_context_prefix_stability.py  # Prefix-reuse invariants
├── nginx/
│   └── nginx.conf              # Reverse proxy config
├── docker-compose.yml          # Full stack orchestration
├── litellm_config.yaml         # LiteLLM routing (model selection)
├── litellm_config.roundrobin.yaml  # llm-d-free control arm / rollback
├── Dockerfile                  # App container image
├── requirements.txt            # Python dependencies
├── requirements-dev.txt        # + test dependencies
├── .env.example                # Environment template
└── README.md
```

## License

Internal use only.
