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

[ -d /tmp/mewtwo_dummy ] || mkdir -p /tmp/mewtwo_dummy
cat <<EOF > /tmp/mewtwo_dummy/config.json
{
  "architectures": [
    "MewtwoForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 0,
  "eos_token_id": 1,
  "ep_size": 1,
  "first_k_dense_replace": 3,
  "hidden_act": "silu",
  "hidden_size": 1024,
  "index_head_dim": 128,
  "index_n_heads": 64,
  "index_topk": 512,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "kv_lora_rank": 448,
  "max_position_embeddings": 65536,
  "model_type": "mewtwo",
  "moe_intermediate_size": 256,
  "moe_layer_freq": 1,
  "n_group": 8,
  "n_routed_experts": 256,
  "n_shared_experts": 1,
  "norm_topk_prob": true,
  "num_attention_heads": 64,
  "num_experts_per_tok": 6,
  "num_hidden_layers": 8,
  "num_key_value_heads": 64,
  "num_nextn_predict_layers": 0,
  "q_lora_rank": 1024,
  "qk_nope_head_dim": 448,
  "qk_rope_head_dim": 64,
  "quantization_config": {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "scale_fmt": "ue8m0",
    "weight_block_size": [
      128,
      128
    ]
  },
  "rms_norm_eps": 1e-06,
  "rope_scaling": {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 4,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 65536,
    "type": "yarn"
  },
  "rope_theta": 10000,
  "routed_scaling_factor": 1.5,
  "scoring_func": "sqrtsoftplus",
  "tie_word_embeddings": false,
  "topk_group": 4,
  "topk_method": "noaux_tc",
  "torch_dtype": "bfloat16",
  "transformers_version": "4.44.2",
  "use_cache": true,
  "v_head_dim": 512,
  "vocab_size": 129280,
  "o_groups": 8,
  "o_lora_rank": 1024,
  "n_hash_layers": 3,
  "hc_mult": 4,
  "hc_sinkhorn_iters": 20,
  "hc_eps": 1e-6,
  "window_size": 128,
  "compress_rope_theta": 40000,
  "compress_ratios": [1, 1, 4, 128, 4, 128, 4, 128]
}
EOF

MODEL_DIR=/tmp/mewtwo_dummy
TOKENIZER_DIR="${TOKENIZER_DIR:-/llm-models/DeepSeek-V3.2-Exp-hf}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CTX_CONFIG=${SCRIPT_DIR}/ctx_config.yaml
GEN_CONFIG=${SCRIPT_DIR}/gen_config.yaml
DISAGG_CONFIG=${SCRIPT_DIR}/disagg_config.yaml

# ── Write ctx_config.yaml ───────────────────────────────────────────────────
cat > "${CTX_CONFIG}" << 'EOF'
disable_overlap_scheduler: true
load_format: dummy
enable_attention_dp: true
moe_expert_parallel_size: 4
moe_config:
  backend: DEEPGEMM
kv_cache_config:
  dtype: fp8
  enable_block_reuse: false
  tokens_per_block: 128
cuda_graph_config: {}
cache_transceiver_config:
  backend: NIXL
  max_tokens_in_buffer: 2048
  transceiver_runtime: PYTHON
EOF

# ── Write gen_config.yaml ───────────────────────────────────────────────────
cat > "${GEN_CONFIG}" << 'EOF'
load_format: dummy
enable_attention_dp: true
moe_expert_parallel_size: 4
moe_config:
  backend: DEEPGEMM
kv_cache_config:
  dtype: fp8
  enable_block_reuse: false
  tokens_per_block: 128
cuda_graph_config: {}
cache_transceiver_config:
  backend: NIXL
  max_tokens_in_buffer: 2048
  transceiver_runtime: PYTHON
EOF

# ── Write disagg_config.yaml ────────────────────────────────────────────────
cat > "${DISAGG_CONFIG}" << 'EOF'
hostname: localhost
port: 8000
backend: pytorch
context_servers:
  num_instances: 1
  urls:
    - "localhost:8001"
generation_servers:
  num_instances: 1
  urls:
    - "localhost:8002"
EOF

# ── Launch context server (GPU 0-3) ─────────────────────────────────────────
# Note: trtllm-llmapi-launch is for SLURM/srun only; on a standalone machine
# trtllm-serve spawns MPI workers internally when tp_size > 1.
echo "[1/3] Starting context server on port 8001 (GPU 0-3)..."
UCX_TLS=tcp,cuda_copy,self CUDA_VISIBLE_DEVICES=0,1,2,3 trtllm-serve "${MODEL_DIR}" \
    --tokenizer "${TOKENIZER_DIR}" \
    --backend pytorch \
    --host localhost --port 8001 \
    --tp_size 4 \
    --config "${CTX_CONFIG}" &> log_ctx.txt &
CTX_PID=$!
echo "  context server PID: ${CTX_PID}"

# ── Launch generation server (GPU 4-7) ──────────────────────────────────────
echo "[2/3] Starting generation server on port 8002 (GPU 4-7)..."
UCX_TLS=tcp,cuda_copy,self CUDA_VISIBLE_DEVICES=4,5,6,7 trtllm-serve "${MODEL_DIR}" \
    --tokenizer "${TOKENIZER_DIR}" \
    --backend pytorch \
    --host localhost --port 8002 \
    --tp_size 4 \
    --config "${GEN_CONFIG}" &> log_gen.txt &
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
trtllm-serve disaggregated -c "${DISAGG_CONFIG}" &> log_disagg.txt &
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
