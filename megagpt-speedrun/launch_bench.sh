#!/usr/bin/env zsh
# Throughput BENCHMARK launcher: short synthetic-data run that measures pure-compute tokens/sec +
# MFU for a candidate config, then exits. No storage cold-start (SP_SYNTH_DATA=1), no checkpoint
# (SP_STEPS small). Use for the SWEEP.md throughput sweep.
#
# Usage:
#   ./launch_bench.sh <name> <gpus:1|8> [extra -e KEY VAL ...]
# e.g. 8-GPU deep-sparse:
#   ./launch_bench.sh bench-T4 8 -e SP_EXPERTS 512 -e SP_TOPK 2 -e SP_SEQ 2048 -e SP_BATCH 64
set -euo pipefail
NAME="${1:?name}"; GPUS="${2:?gpus 1 or 8}"; shift 2 2>/dev/null || true
: "${WANDB_API_KEY:?}"; : "${HF_TOKEN:?}"

ARGS=(
  --gpu "H100x${GPUS}" --enable-extra-resources --extra gpu --cpu 16 --memory 256GB --disk 200GB
  --max-retries 0 --job-name "$NAME"
  -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN"
  # synthetic data: datakit config supplies the tokenizer only (no cache load); train uses random.
  -e SP_DATA datakit -e SP_SYNTH_DATA 1
  -e RAGGED_DOT_IMPL triton -e SP_ATTN gpu_fa4_cute
  -e SP_LOG_EVERY 1 -e SP_STEPS 80 -e SP_TOKENS 2000000000
  # default geometry (override per config via extra -e); mesh = pure expert-parallel over the node.
  -e SP_HIDDEN 1536 -e SP_EMBED 512 -e SP_EXPERTS 64 -e SP_TOPK 8 -e SP_SEQ 4096
  -e SP_EP "$GPUS" -e SP_TP 1 -e SP_REPLICA 1
  -e SP_GROUP megagpt-bench
)
ARGS+=( "$@" )

KUBECONFIG=~/.kube/coreweave-iris-gpu uv run iris --cluster=cw-us-east-02a job run --no-wait \
  "${ARGS[@]}" -- python launch.py
