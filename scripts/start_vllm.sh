#!/usr/bin/env bash
# ──────────────────────────────────────────────
# Launch vLLM instances on 8× MI350X GPUs
#
# GPU 0-5: Llama 3.1 405B MXFP4 (chat/reasoning)
# GPU 6:   Qwen 2.5 Coder 32B FP8 (code gen)
# GPU 7:   BGE-large-en-v1.5 (embeddings for RAG)
# ──────────────────────────────────────────────
set -euo pipefail

# Load env vars if .env exists
if [ -f .env ]; then
    set -a; source .env; set +a
fi

VLLM_IMAGE="${VLLM_IMAGE:-rocm/vllm:latest}"
VLLM_API_KEY="${VLLM_CHAT_API_KEY:-vllm-internal-key}"
MODEL_DIR="${MODEL_STORAGE_PATH:-/models}"

echo "═══════════════════════════════════════════"
echo "  Starting vLLM on 8× MI350X"
echo "═══════════════════════════════════════════"

# ── Verify GPUs ──
echo "[Pre-check] Verifying GPUs..."
GPU_COUNT=$(rocm-smi --showid 2>/dev/null | grep -c GPU || echo 0)
if [ "$GPU_COUNT" -lt 8 ]; then
    echo "WARNING: Found $GPU_COUNT GPUs, expected 8. Proceeding anyway."
fi

# ── Common Docker flags ──
COMMON_FLAGS=(
    --network=host
    --device=/dev/kfd
    --device=/dev/dri
    --group-add=video
    --ipc=host
    --cap-add=SYS_PTRACE
    --security-opt seccomp=unconfined
    --shm-size 16G
    -v "${MODEL_DIR}:${MODEL_DIR}"
    -e HIP_FORCE_DEV_KERNARG=1
    -e VLLM_ROCM_USE_AITER=1
    -e NCCL_MIN_NCHANNELS=112
    --restart unless-stopped
)

# ── Stop existing containers ──
echo "[Cleanup] Stopping existing vLLM containers..."
docker rm -f vllm-chat vllm-code vllm-embed 2>/dev/null || true
sleep 2

# ──────────────────────────────────────────────
# Instance 1: Chat Model — Llama 3.1 405B MXFP4
# GPUs 0-5, Tensor Parallel = 6
# ──────────────────────────────────────────────
echo ""
echo "[1/3] Starting Chat Model (Llama 3.1 405B MXFP4) on GPUs 0-5..."
docker run -d \
    --name vllm-chat \
    "${COMMON_FLAGS[@]}" \
    -e ROCR_VISIBLE_DEVICES=0,1,2,3,4,5 \
    "${VLLM_IMAGE}" \
    vllm serve "${VLLM_CHAT_MODEL:-amd/Llama-3.1-405B-Instruct-MXFP4}" \
        --dtype auto \
        --tensor-parallel-size 6 \
        --port 8000 \
        --api-key "${VLLM_API_KEY}" \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.92 \
        --max-num-seqs 256 \
        --enable-chunked-prefill \
        --enable-prefix-caching

echo "  → Chat model starting on :8000"

# ──────────────────────────────────────────────
# Instance 2: Code Model — Qwen 2.5 Coder 32B
# GPU 6, FP8 quantization
# ──────────────────────────────────────────────
echo ""
echo "[2/3] Starting Code Model (Qwen 2.5 Coder 32B FP8) on GPU 6..."
docker run -d \
    --name vllm-code \
    "${COMMON_FLAGS[@]}" \
    -e ROCR_VISIBLE_DEVICES=6 \
    "${VLLM_IMAGE}" \
    vllm serve "${VLLM_CODE_MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct}" \
        --quantization ptpc_fp8 \
        --dtype auto \
        --port 8001 \
        --api-key "${VLLM_API_KEY}" \
        --max-model-len 16384 \
        --gpu-memory-utilization 0.90 \
        --max-num-seqs 128 \
        --enable-chunked-prefill \
        --enable-prefix-caching

echo "  → Code model starting on :8001"

# ──────────────────────────────────────────────
# Instance 3: Embedding Model — BGE-large
# GPU 7
# ──────────────────────────────────────────────
echo ""
echo "[3/3] Starting Embedding Model (BGE-large-en-v1.5) on GPU 7..."
docker run -d \
    --name vllm-embed \
    "${COMMON_FLAGS[@]}" \
    -e ROCR_VISIBLE_DEVICES=7 \
    "${VLLM_IMAGE}" \
    vllm serve "${VLLM_EMBED_MODEL:-BAAI/bge-large-en-v1.5}" \
        --dtype auto \
        --port 8002 \
        --api-key "${VLLM_API_KEY}" \
        --max-num-seqs 512

echo "  → Embedding model starting on :8002"

# ── Wait and verify ──
echo ""
echo "Waiting for models to load (this may take several minutes for 405B)..."
echo ""

check_endpoint() {
    local name=$1 port=$2 max_wait=$3
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
            echo "  ✓ ${name} is ready on :${port}"
            return 0
        fi
        sleep 10
        waited=$((waited + 10))
        echo "    ... ${name} loading (${waited}s / ${max_wait}s)"
    done
    echo "  ✗ ${name} did not start within ${max_wait}s. Check: docker logs ${name}"
    return 1
}

check_endpoint "Embedding" 8002 120 &
check_endpoint "Code" 8001 300 &
check_endpoint "Chat (405B)" 8000 900 &
wait

echo ""
echo "═══════════════════════════════════════════"
echo "  vLLM instances launched."
echo ""
echo "  Chat  (405B) → http://localhost:8000"
echo "  Code  (32B)  → http://localhost:8001"
echo "  Embed (BGE)  → http://localhost:8002"
echo ""
echo "  Monitor: docker logs -f vllm-chat"
echo "═══════════════════════════════════════════"
