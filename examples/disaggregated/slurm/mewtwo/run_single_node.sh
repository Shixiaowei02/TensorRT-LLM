#!/bin/bash
# Disaggregated serving launch script for DeepSeek-V3.2 (Mewtwo)
# Context server: GPU 0-3 (prefill)
# Generation server: GPU 4-7 (decode)
# Orchestrator: port 8000

set -e

# ── Cleanup on exit ──────────────────────────────────────────────────────────
cleanup() {
  echo ""
  echo "Shutting down servers..."
  kill "${CTX_PID}" "${GEN_PID}" "${DISAGG_PID}" 2>/dev/null || true
  wait "${CTX_PID}" "${GEN_PID}" "${DISAGG_PID}" 2>/dev/null || true
  echo "Done."
}
trap cleanup EXIT INT TERM
unset SLURM_JOBID SLURM_NODELIST

TOKENIZER_DIR="${TOKENIZER_DIR:-/llm-models/DeepSeek-V3.2-Exp-hf}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}"

CTX_NODE=localhost; CTX_PORT=8001
GEN_NODE=localhost; GEN_PORT=8002
PROXY_PORT=8000

CTX_CONFIG=${SCRIPT_DIR}/ctx_config_tep4.yaml
GEN_CONFIG=${SCRIPT_DIR}/gen_config_tep4.yaml
RUN_DIR="${SCRIPT_DIR}/run"; mkdir -p "${RUN_DIR}"
DISAGG_CONFIG="${RUN_DIR}/disagg_config_runtime.yaml"
sed -e "s|\${CTX_NODE}|${CTX_NODE}|g" \
    -e "s|\${CTX_PORT}|${CTX_PORT}|g" \
    -e "s|\${GEN_NODE}|${GEN_NODE}|g" \
    -e "s|\${GEN_PORT}|${GEN_PORT}|g" \
    -e "s|\${PROXY_PORT}|${PROXY_PORT}|g" \
    "${SCRIPT_DIR}/disagg_config_1ctx_1gen.yaml" > "${DISAGG_CONFIG}"

# ── Launch context server (GPU 0-3) ─────────────────────────────────────────
# Note: trtllm-llmapi-launch is for SLURM/srun only; on a standalone machine
# trtllm-serve spawns MPI workers internally when tp_size > 1.
echo "[1/3] Starting context server on port 8001 (GPU 0-3)..."
UCX_TLS=tcp,cuda_copy,self CUDA_VISIBLE_DEVICES=0,1,2,3 trtllm-serve "${MODEL_DIR}" \
    --tokenizer "${TOKENIZER_DIR}" \
    --backend pytorch \
    --host localhost --port 8001 \
    --tp_size 4 \
    --config "${CTX_CONFIG}" &> "${RUN_DIR}/log_ctx.txt" &
CTX_PID=$!
echo "  context server PID: ${CTX_PID}"

# ── Launch generation server (GPU 4-7) ──────────────────────────────────────
echo "[2/3] Starting generation server on port 8002 (GPU 4-7)..."
UCX_TLS=tcp,cuda_copy,self CUDA_VISIBLE_DEVICES=4,5,6,7 trtllm-serve "${MODEL_DIR}" \
    --tokenizer "${TOKENIZER_DIR}" \
    --backend pytorch \
    --host localhost --port 8002 \
    --tp_size 4 \
    --config "${GEN_CONFIG}" &> "${RUN_DIR}/log_gen.txt" &
GEN_PID=$!
echo "  generation server PID: ${GEN_PID}"

# ── Wait for both workers to be ready ───────────────────────────────────────
echo "Waiting for context and generation servers to be ready..."
for PORT in 8001 8002; do
  until curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; do
    sleep 5
  done
  echo "  port ${PORT} ready."
done

# ── Launch disaggregated proxy ───────────────────────────────────────────────
echo "[3/3] Starting disaggregated proxy on port 8000..."
trtllm-serve disaggregated -c "${DISAGG_CONFIG}" &> "${RUN_DIR}/log_disagg.txt" &
DISAGG_PID=$!
echo "  disaggregated proxy PID: ${DISAGG_PID}"

echo "Waiting for disaggregated proxy (port 8000) to be ready..."
until curl -s "http://localhost:8000/health" > /dev/null 2>&1; do
  sleep 5
done
echo "  port 8000 ready."

# ── Send fake request and print output ──────────────────────────────────────
echo ""
echo "====== Fake completion request ======"
MODEL_NAME=$(basename "${MODEL_DIR}")
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL_NAME}\",
    \"prompt\": \"The future of AI is\",
    \"max_tokens\": 32,
    \"temperature\": 0
  }" | python3 -c "
import sys, json
resp = json.load(sys.stdin)
print('prompt :', 'The future of AI is')
print('output :', resp['choices'][0]['text'])
print('tokens :', resp['usage'])
"
echo "====== Done ======"

# Keep servers alive until Ctrl-C
wait "${DISAGG_PID}"
