#!/usr/bin/env bash
# ──────────────────────────────────────────────
# Launch vLLM on 8× MI350X with a REPLICATED chat pool.
#
#   GPUs 0,1 → Llama 3.1 405B MXFP4, TP=2, :8000   ┐
#   GPUs 2,3 → Llama 3.1 405B MXFP4, TP=2, :8010   ├ chat pool (llm-d routes here)
#   GPUs 4,5 → Llama 3.1 405B MXFP4, TP=2, :8020   ┘
#   GPU  6   → Qwen 2.5 Coder 32B FP8,      :8001   (single replica, direct)
#   GPU  7   → BGE-large-en-v1.5,           :8002   (single replica, direct)
#
# WHY TP=2 AND NOT TP=6
#   Llama 3.1 405B has 128 attention heads, 8 KV heads and intermediate_size
#   53248. vLLM requires the tensor-parallel size to divide all three. TP=6
#   fails on two of them (8 % 6 = 2, 53248 % 6 = 4) and the server exits at
#   startup — the previous single-replica config could never have run. Only
#   TP ∈ {2, 4, 8} are valid for this model.
#
# WHY REPLICATE AT ALL
#   The llm-d endpoint picker selects among replicas of the SAME model using
#   prefix-cache and load awareness. With one replica there is nothing to
#   pick, so a pool is the precondition for llm-d doing anything.
#
# COST — READ BEFORE RUNNING
#   Replication triples the resident weights (~223 GB each) and costs roughly
#   34% of aggregate KV capacity versus a single large shard. Per-token decode
#   is also ~3x slower for a SINGLE request, because each GPU streams a 112 GB
#   shard per token instead of ~37 GB. This layout wins on aggregate
#   throughput under concurrency and loses at low concurrency. Measure before
#   committing — see GATE B in the plan.
#
# Rollback: scripts/start_vllm.sh (single replica, TP=4 on GPUs 0-3).
# ──────────────────────────────────────────────
set -euo pipefail

# Load env vars if .env exists
if [ -f .env ]; then
    set -a; source .env; set +a
fi

VLLM_IMAGE="${VLLM_IMAGE:-rocm/vllm:latest}"
VLLM_API_KEY="${VLLM_CHAT_API_KEY:-vllm-internal-key}"
MODEL_DIR="${MODEL_STORAGE_PATH:-/models}"
CHAT_MODEL="${VLLM_CHAT_MODEL:-amd/Llama-3.1-405B-Instruct-MXFP4}"

# Per-replica concurrency cap.
#
# The old value of 256 was set for a single large shard. At TP=2 each replica
# holds roughly 138 GB of KV per GPU, about 560k tokens, which is only ~93
# concurrent sequences at a realistic ~6k average context. Admitting 256 makes
# vLLM accept requests it cannot hold and preempt-and-recompute — which also
# evicts the prefix cache that the whole llm-d routing story depends on.
# Tune from the vllm:num_preemptions_total metric.
CHAT_MAX_NUM_SEQS="${CHAT_MAX_NUM_SEQS:-128}"

echo "═══════════════════════════════════════════"
echo "  Starting vLLM — 3× TP=2 chat pool"
echo "═══════════════════════════════════════════"

# ── Pre-checks ──
echo "[Pre-check] Verifying GPUs..."
GPU_COUNT=$(rocm-smi --showid 2>/dev/null | grep -c GPU || echo 0)
if [ "$GPU_COUNT" -lt 8 ]; then
    echo "WARNING: Found $GPU_COUNT GPUs, expected 8. Proceeding anyway."
fi

# Three TP=2 replicas contend for the host's /dev/shm where one shard did
# before. Note --ipc=host makes --shm-size a no-op, so the HOST value is what
# matters. RCCL exhausting shm presents as a hang, not an error message.
if [ -d /dev/shm ]; then
    SHM_AVAIL_GB=$(df -BG --output=avail /dev/shm 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
    echo "[Pre-check] /dev/shm available: ${SHM_AVAIL_GB}G (want >= 60G for 3 replicas)"
    if [ "${SHM_AVAIL_GB:-0}" -lt 60 ]; then
        echo "  WARNING: /dev/shm may be too small. If replicas hang during"
        echo "           init, remount larger: mount -o remount,size=64G /dev/shm"
    fi
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
    --restart unless-stopped
)

# NOTE: NCCL_MIN_NCHANNELS=112 was carried over from the 6-way config and is
# almost certainly wrong for a 2-GPU XGMI pair. It is deliberately NOT set
# here. Benchmark before reintroducing it.

# ── Stop existing containers ──
echo "[Cleanup] Stopping existing vLLM containers..."
docker rm -f vllm-chat vllm-chat-0 vllm-chat-1 vllm-chat-2 vllm-code vllm-embed 2>/dev/null || true
sleep 2

# ──────────────────────────────────────────────
# Chat pool — 3 replicas, TP=2
# ──────────────────────────────────────────────
start_chat_replica() {
    local idx=$1 gpus=$2 port=$3
    echo ""
    echo "[chat ${idx}] Llama 3.1 405B MXFP4, TP=2, GPUs ${gpus} → :${port}"
    docker run -d \
        --name "vllm-chat-${idx}" \
        "${COMMON_FLAGS[@]}" \
        -e ROCR_VISIBLE_DEVICES="${gpus}" \
        "${VLLM_IMAGE}" \
        vllm serve "${CHAT_MODEL}" \
            --dtype auto \
            --tensor-parallel-size 2 \
            --port "${port}" \
            --api-key "${VLLM_API_KEY}" \
            --max-model-len 32768 \
            --gpu-memory-utilization 0.92 \
            --max-num-seqs "${CHAT_MAX_NUM_SEQS}" \
            --enable-chunked-prefill \
            --enable-prefix-caching
    # NOTE: --block-size is deliberately left at the default. The llm-d
    # APPROXIMATE prefix scorer hashes prompt text in its own blocks and has
    # no coupling to vLLM's paged-attention block size; matching them only
    # matters for PRECISE (KV-event) routing, which is not enabled here. The
    # ROCm paged-attention kernel is also tuned for block sizes 16/32, so
    # changing it risks falling back to a slower Triton path.
}

start_chat_replica 0 "0,1" 8000
start_chat_replica 1 "2,3" 8010
start_chat_replica 2 "4,5" 8020

# ──────────────────────────────────────────────
# Code Model — Qwen 2.5 Coder 32B, GPU 6 (unchanged)
# ──────────────────────────────────────────────
echo ""
echo "[code] Qwen 2.5 Coder 32B FP8, GPU 6 → :8001"
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

# ──────────────────────────────────────────────
# Embedding Model — BGE-large, GPU 7 (unchanged)
# ──────────────────────────────────────────────
echo ""
echo "[embed] BGE-large-en-v1.5, GPU 7 → :8002"
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

# ── Wait and verify ──
echo ""
echo "Waiting for models to load (405B replicas take several minutes)..."
echo ""

check_endpoint() {
    local name=$1 port=$2 max_wait=$3
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
            echo "  ✓ ${name} ready on :${port}"
            return 0
        fi
        sleep 10
        waited=$((waited + 10))
    done
    echo "  ✗ ${name} did not start within ${max_wait}s. Check: docker logs ${name}"
    return 1
}

rc=0
check_endpoint "vllm-embed"  8002 120 || rc=1 &
check_endpoint "vllm-code"   8001 300 || rc=1 &
check_endpoint "vllm-chat-0" 8000 900 || rc=1 &
check_endpoint "vllm-chat-1" 8010 900 || rc=1 &
check_endpoint "vllm-chat-2" 8020 900 || rc=1 &
wait

echo ""
echo "═══════════════════════════════════════════"
echo "  Chat pool  → :8000 :8010 :8020  (llm-d routes among these)"
echo "  Code       → :8001"
echo "  Embed      → :8002"
echo ""
echo "  Next: confirm all three chat replicas are listed in"
echo "        llmd/epp/endpoints.yaml, then start the router:"
echo "          docker compose up -d epp envoy"
echo "═══════════════════════════════════════════"
