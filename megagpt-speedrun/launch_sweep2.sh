#!/usr/bin/env zsh
# SWEEP 2 — 8-GPU (H100x8, EP=8) deep-sparse throughput, the EP-REAL numbers single-GPU can't get
# (the all-to-all collective is THE deep-sparse cost). Answers: does a big-E MoE actually go fast at
# EP=8? Re-measures the E512/K2 config that earlier mis-read as 352 s/it (compile artifact). Measure
# steady-state with: uv run python analyze_sweep.py <prefix> 8 20
#
# Each GRID line: <suffix>|<SP_ overrides KEY=VAL ...>. Base: D1536 d_e512 EP8 seq2048 save_moe,
# synthetic data, FA4, 60 steps. Batch tuned per-config (E512 OOMs above ~b8).
set -euo pipefail
: "${WANDB_API_KEY:?}"; : "${HF_TOKEN:?}"
PREFIX="${1:?prefix}"; ONLY="${2:-}"

typeset -A BASE
BASE=(
  SP_DATA datakit  SP_SYNTH_DATA 1  RAGGED_DOT_IMPL triton  SP_ATTN gpu_fa4_cute
  SP_LOG_EVERY 1   SP_STEPS 60      SP_TOKENS 2000000000
  SP_HIDDEN 1536   SP_EMBED 512     SP_EXPERTS 128  SP_TOPK 8  SP_SEQ 2048  SP_BATCH 16
  SP_EP 8          SP_TP 1          SP_REPLICA 1   SP_REMAT save_moe
  SP_GROUP megagpt-sw2grp
)

GRID=(
  # --- E-ladder at EP=8 (does the collective + router stay cheap as E grows?) ---
  "e64k8|SP_EXPERTS=64 SP_TOPK=8 SP_BATCH=16"
  "e128k8|SP_EXPERTS=128 SP_TOPK=8 SP_BATCH=16"
  "e256k4|SP_EXPERTS=256 SP_TOPK=4 SP_BATCH=8"
  "e512k2|SP_EXPERTS=512 SP_TOPK=2 SP_BATCH=8"
  # --- deep-sparse + cheap attn (the 100B-1T candidate) ---
  "e256k4gl|SP_EXPERTS=256 SP_TOPK=4 SP_BATCH=8 SP_GLOBAL_EVERY=6 SP_LOCAL_WINDOW=1024"
  "e512k2gl|SP_EXPERTS=512 SP_TOPK=2 SP_BATCH=8 SP_GLOBAL_EVERY=6 SP_LOCAL_WINDOW=1024"
  # --- dispatch impl + approx-router on the deep config ---
  "e256rata|SP_EXPERTS=256 SP_TOPK=4 SP_BATCH=8 SP_MOE_IMPL=ragged_all_to_all"
  "e256fqb|SP_EXPERTS=256 SP_TOPK=4 SP_BATCH=8 SP_FAST_QB=1"
  # --- OPTIMIZE THE WINNER (E64/K8, 3.6B fits 1 GPU): kill EP overhead (DP, no all-to-all),
  #     cheapen attn (global/local), push batch for MFU. Baseline EP8 b16 = 111K tok/s, 8.1% MFU. ---
  "e64dp|SP_EXPERTS=64 SP_TOPK=8 SP_BATCH=16 SP_EP=1"
  "e64dpgl|SP_EXPERTS=64 SP_TOPK=8 SP_BATCH=16 SP_EP=1 SP_GLOBAL_EVERY=6 SP_LOCAL_WINDOW=1024"
  "e64dpb32|SP_EXPERTS=64 SP_TOPK=8 SP_BATCH=32 SP_EP=1"
  "e64ep8gl|SP_EXPERTS=64 SP_TOPK=8 SP_BATCH=16 SP_GLOBAL_EVERY=6 SP_LOCAL_WINDOW=1024"
)

echo "launching deep-sparse 8-GPU jobs (prefix=$PREFIX)"
for spec in "${GRID[@]}"; do
  suffix="${spec%%|*}"; overrides="${spec#*|}"
  if [[ -n "$ONLY" && ",$ONLY," != *",$suffix,"* ]]; then continue; fi
  typeset -A env; env=("${(@kv)BASE}")
  for kv in ${(s: :)overrides}; do env[${kv%%=*}]="${kv#*=}"; done
  args=( --gpu H100x8 --enable-extra-resources --extra gpu --cpu 32 --memory 512GB --disk 200GB
         --max-retries 0 --job-name "${PREFIX}-${suffix}"
         -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" )
  for k in ${(k)env}; do args+=( -e "$k" "${env[$k]}" ); done
  KUBECONFIG=~/.kube/coreweave-iris-gpu uv run iris --cluster=cw-us-east-02a job run --no-wait \
    "${args[@]}" -- python launch.py >/dev/null 2>&1
  echo "  submitted ${PREFIX}-${suffix}   [${overrides}]"
done
echo "done"
