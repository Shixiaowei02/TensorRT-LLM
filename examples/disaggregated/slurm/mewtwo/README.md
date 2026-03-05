# Mewtwo Disaggregated Serving — Smoke Test

Uses a **dummy model** (random weights, `load_format: dummy`) to verify the
disaggregated prefill/decode pipeline without loading real checkpoints.

## Files

| File | Description |
|------|-------------|
| `config.json` | Dummy Mewtwo model architecture |
| `ctx_config_tep4.yaml` | Prefill server config (tp=4, ep=4, fp8 KV, NIXL) |
| `gen_config_tep4.yaml` | Decode server config (tp=4, ep=4, fp8 KV, NIXL) |
| `disagg_config_1ctx_1gen.yaml` | Disagg proxy topology template (1 ctx + 1 gen) |
| `run_single_node.sh` | Single-node launch (8 GPUs: 0-3 prefill, 4-7 decode) |
| `launch.slurm` | Multi-node SLURM launch (3 nodes) |

Runtime output (logs + generated configs) goes to `run/`.

## Single-node

```bash
TOKENIZER_DIR=/path/to/tokenizer bash run_single_node.sh
```

Requires 8 GPUs on one machine. On success the script sends a test completion
request and prints the output, then keeps the servers alive until Ctrl-C.

Logs: `run/log_ctx.txt`, `run/log_gen.txt`, `run/log_disagg.txt`

## Multi-node (SLURM)

Edit the `#SBATCH` header in `launch.slurm` (partition, account), then:

```bash
TOKENIZER_DIR=/path/to/tokenizer sbatch launch.slurm
```

Allocates **3 nodes** (4 GPUs each):

| Node | Role | Port |
|------|------|------|
| `NODES[0]` | Prefill server (tp=4) | 8001 |
| `NODES[1]` | Decode server (tp=4) | 8002 |
| `NODES[2]` | Disagg proxy | 8000 |

Logs: `run/<SLURM_JOB_ID>/log_ctx.txt`, `log_gen.txt`, `log_disagg.txt`

## Cleanup

```bash
rm -rf run/
```
