#!/usr/bin/env zsh
# megagpt-speedrun CoreWeave H100x8 launcher.
#
# Usage:
#   ./launch_cw.sh <job-name> <SP_TOKENS> <SP_STEPS|""> <SP_GROUP> [extra -e KEY VAL ...]
#
# Headline geometry is baked in: D=1536 d_e=512 E=256 K=8 seq=4096 EP=8 (15B total /
# 0.75B active, factorized embedding, vocab 128256). Data = R2 nemotron tokenized mix.
# SP_TOKENS sets batch/steps/LR via the heuristic (iso-token). Pass SP_STEPS="" to use the
# heuristic step count (the real WSD schedule); pass a number only for short fit/throughput
# smokes (overriding desyncs the LR schedule).
set -euo pipefail
NAME="${1:?job-name}"; TOKENS="${2:?SP_TOKENS}"; STEPS="${3:-}"; GROUP="${4:-megagpt}"
shift 4 2>/dev/null || true

ARGS=(
  --gpu H100x8 --enable-extra-resources --extra gpu --cpu 32 --memory 512GB --disk 400GB
  --max-retries 3 --job-name "$NAME"  # retry resumes same run_id from the 30-min checkpoint (3x for long 24h runs)
  -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN"
  -e SP_HIDDEN 1536 -e SP_EMBED 512 -e SP_EXPERTS 256 -e SP_TOPK 8 -e SP_SEQ 4096
  -e SP_EP 8 -e SP_REPLICA 1 -e SP_DATA datakit
  # RAGGED_DOT_IMPL=triton is REQUIRED: the MoE grouped matmul defaults to "auto", whose
  # GPU path silently falls back to the XLA dense ragged_dot_general -- a per-device [M,G,N]
  # materialization that OOMs (hundreds of GiB) at this deeply-sparse 15B geometry. The
  # triton kernel streams it. (dispatch.py forwards RAGGED_DOT_IMPL to the train task.)
  -e RAGGED_DOT_IMPL triton
  -e SP_TOKENS "$TOKENS" -e SP_GROUP "$GROUP"
)
[[ -n "$STEPS" ]] && ARGS+=( -e SP_STEPS "$STEPS" )
ARGS+=( "$@" )

KUBECONFIG=~/.kube/coreweave-iris-gpu uv run iris --cluster=cw-us-east-02a job run --no-wait \
  "${ARGS[@]}" -- python launch.py
