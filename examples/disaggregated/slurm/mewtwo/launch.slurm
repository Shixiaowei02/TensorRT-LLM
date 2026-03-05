#!/bin/bash
#SBATCH --partition=batch
#SBATCH --account=coreai_comparch_trtllm
#SBATCH --job-name=coreai_comparch_trtllm-mewtwo.disagg
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --time=02:00:00

# Disaggregated serving multi-node SLURM script for Mewtwo (dummy model)
# Node layout:
#   NODES[0] — context server  (tp_size=4, port 8001)
#   NODES[1] — generation server (tp_size=4, port 8002)
#   NODES[2] — disaggregated proxy (port 8000)

set -e

# ── User-configurable ────────────────────────────────────────────────────────
CONTAINER_IMAGE="urm.nvidia.com#sw-tensorrt-docker/tensorrt-llm:pytorch-25.12-py3-x86_64-ubuntu24.04-trt10.14.1.48-skip-tritondevel-202602011118-10901"          # set to container image path if using containers
MOUNT_PATHS="/lustre/fsw/coreai_comparch_trtllm/xiaoweis:/lustre/fsw/coreai_comparch_trtllm/xiaoweis,/lustre/fsw/coreai_comparch_trtllm/llm_data:/lustre/fsw/coreai_comparch_trtllm/llm_data,/home/xiaoweis:/home/xiaoweis"              # e.g. "/llm-models:/llm-models,/tmp:/tmp"
TOKENIZER_DIR="${TOKENIZER_DIR:-/lustre/fsw/coreai_comparch_trtllm/xiaoweis/llm-models/DeepSeek-V3.2-Exp-hf/}"

CTX_PORT=8001
GEN_PORT=8002
PROXY_PORT=8000

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
MODEL_DIR="${SCRIPT_DIR}"
CTX_CONFIG="${SCRIPT_DIR}/ctx_config_tep4.yaml"
GEN_CONFIG="${SCRIPT_DIR}/gen_config_tep4.yaml"
RUN_DIR="${SCRIPT_DIR}/run/${SLURM_JOB_ID}"; mkdir -p "${RUN_DIR}"
DISAGG_CONFIG="${RUN_DIR}/disagg_config_runtime.yaml"

# ── Resolve SLURM job partition/account from script's own #SBATCH directives ─
SLURM_JOB_PARTITION=$(grep '^#SBATCH --partition=' "${BASH_SOURCE[0]}" | sed 's/.*--partition=//')
SLURM_JOB_ACCOUNT=$(grep '^#SBATCH --account=' "${BASH_SOURCE[0]}" | sed 's/.*--account=//')

# ── Resolve node assignments from SLURM allocation ──────────────────────────
readarray -t NODES < <(scontrol show hostnames "$SLURM_NODELIST")
CTX_NODE="${NODES[0]}"
GEN_NODE="${NODES[1]}"
PROXY_NODE="${NODES[2]}"

echo "Context server node  : ${CTX_NODE}"
echo "Generation server node: ${GEN_NODE}"
echo "Proxy node           : ${PROXY_NODE}"

# ── Generate disagg_config with actual node hostnames ───────────────────────
sed -e "s|\${CTX_NODE}|${CTX_NODE}|g" \
    -e "s|\${CTX_PORT}|${CTX_PORT}|g" \
    -e "s|\${GEN_NODE}|${GEN_NODE}|g" \
    -e "s|\${GEN_PORT}|${GEN_PORT}|g" \
    -e "s|\${PROXY_PORT}|${PROXY_PORT}|g" \
    "${SCRIPT_DIR}/disagg_config_1ctx_1gen.yaml" > "${DISAGG_CONFIG}"
echo "disagg_config_runtime.yaml:"; cat "${DISAGG_CONFIG}"
echo "Run dir: ${RUN_DIR}"

# ── Helper: build srun prefix (with or without container) ───────────────────
srun_cmd() {
  local node="$1"; shift
  if [[ -n "${CONTAINER_IMAGE}" ]]; then
    echo "srun --account=${SLURM_JOB_ACCOUNT} --partition=${SLURM_JOB_PARTITION} --nodelist=${node} --container-image=${CONTAINER_IMAGE} --container-mounts=${MOUNT_PATHS} $*"
  else
    echo "srun --account=${SLURM_JOB_ACCOUNT} --partition=${SLURM_JOB_PARTITION} --nodelist=${node} $*"
  fi
}

# ── Launch context server ────────────────────────────────────────────────────
echo "[1/3] Starting context server on ${CTX_NODE}:${CTX_PORT}..."
$(srun_cmd "${CTX_NODE}" -N1 --ntasks-per-node=4 --mpi=pmix) \
  bash -c "bash ${SCRIPT_DIR}/install_trtllm.sh && trtllm-llmapi-launch trtllm-serve ${MODEL_DIR} \
      --tokenizer ${TOKENIZER_DIR} \
      --backend pytorch \
      --host 0.0.0.0 --port ${CTX_PORT} \
      --tp_size 4 \
      --config ${CTX_CONFIG}" &> "${RUN_DIR}/log_ctx.txt" &
CTX_PID=$!

# ── Launch generation server ─────────────────────────────────────────────────
echo "[2/3] Starting generation server on ${GEN_NODE}:${GEN_PORT}..."
$(srun_cmd "${GEN_NODE}" -N1 --ntasks-per-node=4 --mpi=pmix) \
  bash -c "bash ${SCRIPT_DIR}/install_trtllm.sh && trtllm-llmapi-launch trtllm-serve ${MODEL_DIR} \
      --tokenizer ${TOKENIZER_DIR} \
      --backend pytorch \
      --host 0.0.0.0 --port ${GEN_PORT} \
      --tp_size 4 \
      --config ${GEN_CONFIG}" &> "${RUN_DIR}/log_gen.txt" &
GEN_PID=$!

# ── Wait for context and generation servers to be ready ─────────────────────
echo "Waiting for context and generation servers..."
for HOSTPORT in "${CTX_NODE}:${CTX_PORT}" "${GEN_NODE}:${GEN_PORT}"; do
  H="${HOSTPORT%%:*}"; P="${HOSTPORT##*:}"
  until curl -s "http://${H}:${P}/health" > /dev/null 2>&1; do sleep 5; done
  echo "  ${HOSTPORT} ready."
done

# ── Launch disaggregated proxy ───────────────────────────────────────────────
echo "[3/3] Starting disaggregated proxy on ${PROXY_NODE}:${PROXY_PORT}..."
$(srun_cmd "${PROXY_NODE}" -N1 --ntasks-per-node=1 --mpi=pmix) \
  bash -c "bash ${SCRIPT_DIR}/install_trtllm.sh && trtllm-llmapi-launch trtllm-serve disaggregated -c ${DISAGG_CONFIG}" \
  &> "${RUN_DIR}/log_disagg.txt" &
DISAGG_PID=$!

echo "Waiting for disaggregated proxy..."
until curl -s "http://${PROXY_NODE}:${PROXY_PORT}/health" > /dev/null 2>&1; do sleep 5; done
echo "  ${PROXY_NODE}:${PROXY_PORT} ready."

# ── Send test completion request ─────────────────────────────────────────────
echo ""
echo "====== Fake completion request ======"
MODEL_NAME=$(basename "${MODEL_DIR}")
curl -s "http://${PROXY_NODE}:${PROXY_PORT}/v1/completions" \
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

wait "${CTX_PID}" "${GEN_PID}" "${DISAGG_PID}"
