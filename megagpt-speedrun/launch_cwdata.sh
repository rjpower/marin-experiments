#!/usr/bin/env zsh
# megagpt-speedrun launcher that reads training data from the CoreWeave cluster-local
# **cwobject** mirror instead of remote R2 (much faster; no cross-run contention -> run up to
# 4 concurrent experiments). Same baked geometry as launch_cw.sh.
#
# Usage:
#   ./launch_cwdata.sh <job-name> <SP_TOKENS> <SP_STEPS|""> <SP_GROUP> [extra -e KEY VAL ...]
#
# How it works: the whole training process is pointed at cwobject by overriding the s3 env
# (AWS_* = CW creds, AWS_ENDPOINT_URL = cwobject, LEVANTER_S3_VIRTUAL_HOSTED=1 so the
# build_kvstore_spec patch uses virtual-hosted addressing) AND MARIN_PREFIX -> the cwobject
# bucket so checkpoints/output land there too (tensorstore reads ONE global cred set, so data
# and checkpoints must share a store). iris's own R2 state is a separate process, unaffected.
# Needs: the mirror (mirror_to_cw.py) to have completed the components in SP_CW_COMPONENTS,
# and tensorstore>=0.1.84 (pinned via override-dependencies).
set -euo pipefail
NAME="${1:?job-name}"; TOKENS="${2:?SP_TOKENS}"; STEPS="${3:-}"; GROUP="${4:-megagpt}"
shift 4 2>/dev/null || true
: "${CW_KEY_ID:?set CW_KEY_ID}"; : "${CW_KEY_SECRET:?set CW_KEY_SECRET}"
# Default to the components the bootstrap mirror copies first (smallest-first): math + code.
CWCOMP="${SP_CW_COMPONENTS:-proofpile_2,starcoderdata}"

ARGS=(
  --gpu H100x8 --enable-extra-resources --extra gpu --cpu 32 --memory 512GB --disk 400GB
  --max-retries 3 --job-name "$NAME"
  -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN"
  -e SP_HIDDEN 1536 -e SP_EMBED 512 -e SP_EXPERTS 256 -e SP_TOPK 8 -e SP_SEQ 4096
  -e SP_EP 8 -e SP_REPLICA 1
  -e RAGGED_DOT_IMPL triton
  -e LEVANTER_TS_CACHE_LIMIT 34359738368
  # --- cwobject data path ---
  # NB: iris injects AWS_ENDPOINT_URL (R2) with precedence over -e, so we pass the cwobject
  # endpoint + creds under non-AWS_ names (CW_S3_ENDPOINT / CW_KEY_*); cw_patch applies them
  # IN-PROCESS (before the marin executor runs) so the whole process talks cwobject.
  -e SP_DATA cw -e SP_CW_COMPONENTS "$CWCOMP"
  -e LEVANTER_S3_VIRTUAL_HOSTED 1
  -e CW_S3_ENDPOINT https://cwobject.com
  -e CW_KEY_ID "$CW_KEY_ID" -e CW_KEY_SECRET "$CW_KEY_SECRET"
  -e AWS_DEFAULT_REGION us-east-1
  -e MARIN_PREFIX s3://marin-us-east-02a/marin   # checkpoints/output -> cwobject too
  -e SP_TOKENS "$TOKENS" -e SP_GROUP "$GROUP"
)
[[ -n "$STEPS" ]] && ARGS+=( -e SP_STEPS "$STEPS" )
ARGS+=( "$@" )

KUBECONFIG=~/.kube/coreweave-iris-gpu uv run iris --cluster=cw-us-east-02a job run --no-wait \
  "${ARGS[@]}" -- python launch.py
