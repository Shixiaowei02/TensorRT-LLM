#!/bin/bash
# Install TensorRT-LLM via pip install -e . if not already installed.
# Usage: bash install_trtllm.sh [TRTLLM_SRC_DIR]
#   TRTLLM_SRC_DIR defaults to the value below if not specified.
#
# When running under MPI (SLURM_LOCALID set), only rank 0 runs pip install;
# other ranks wait for it to finish.

TRTLLM_SRC_DIR="${1:-/lustre/fsw/coreai_comparch_trtllm/xiaoweis/tekit}"
LOCAL_RANK="${SLURM_LOCALID:-0}"
DONE_FLAG="/tmp/trtllm_install_done_${SLURM_JOB_ID:-$$}"

if [[ "${LOCAL_RANK}" -ne 0 ]]; then
    # Non-zero ranks wait for rank 0 to finish install
    echo "[rank ${LOCAL_RANK}] Waiting for rank 0 to finish install..."
    for i in $(seq 120); do
        [[ -f "${DONE_FLAG}" ]] && break
        sleep 5
    done
    if [[ ! -f "${DONE_FLAG}" ]]; then
        echo "ERROR: rank ${LOCAL_RANK}: install timed out" >&2
        exit 1
    fi
    echo "[rank ${LOCAL_RANK}] Install done, continuing."
    exit 0
fi

# Rank 0: run the install
rm -f "${DONE_FLAG}"

if which trtllm-llmapi-launch > /dev/null 2>&1; then
    echo "trtllm-llmapi-launch already available, skipping install."
    touch "${DONE_FLAG}"
    exit 0
fi

echo "Installing from ${TRTLLM_SRC_DIR} ..."
if [[ ! -d "${TRTLLM_SRC_DIR}" ]]; then
    echo "ERROR: Source directory not found: ${TRTLLM_SRC_DIR}" >&2
    touch "${DONE_FLAG}"  # unblock other ranks
    exit 1
fi

cd "${TRTLLM_SRC_DIR}"

# Install dependencies first (skips already-installed packages)
if [[ -f requirements.txt ]]; then
    echo "Installing requirements.txt ..."
    pip install -r requirements.txt 2>&1
fi

# Install tekit itself without re-resolving deps (avoids z3/etc conflicts)
pip install --no-deps -e . 2>&1
if [[ $? -ne 0 ]]; then
    echo "ERROR: pip install failed" >&2
    touch "${DONE_FLAG}"  # unblock other ranks
    exit 1
fi

echo "tensorrt_llm installed successfully."
touch "${DONE_FLAG}"
