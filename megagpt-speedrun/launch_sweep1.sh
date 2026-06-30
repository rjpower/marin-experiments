#!/usr/bin/env zsh
# SWEEP 1 — single-GPU (H100x1) lever-ranking + E-scan, massively parallel (~20 jobs at once).
#
# Rationale: a single device skips the 8-way SPMD partition (which made the 8-GPU benchmarks take
# ~15min to compile) AND lets us run ~8x more configs at once. It ranks the *per-device* levers
# (seq, batch ceiling, global:local attn, remat, fast_qb, K, E-scaling of router/ragged-dot) that
# transfer directly to the big model. It does NOT capture the EP all-to-all collective overhead --
# that needs the 8-GPU confirm runs. The E-scan here isolates whether high-E slowness is per-device
# compute (visible at EP=1) vs the collective (only at EP=8).
#
# Each line of GRID is:  <job-suffix>|<extra SP_ overrides as KEY=VAL space-separated>
# Base proxy (overridable): D1536 d_e512 E32 K4 seq2048 batch8 EP1, synthetic data, fa4, 60 steps.
set -euo pipefail
: "${WANDB_API_KEY:?}"; : "${HF_TOKEN:?}"
PREFIX="${1:-sw1}"
# optional $2 = comma-separated suffix filter (relaunch a subset), e.g. "e32,e64,k1"
ONLY="${2:-}"

typeset -A BASE
BASE=(
  SP_DATA datakit  SP_SYNTH_DATA 1  RAGGED_DOT_IMPL triton  SP_ATTN gpu_fa4_cute
  SP_LOG_EVERY 1   SP_STEPS 60      SP_TOKENS 2000000000
  SP_HIDDEN 1536   SP_EMBED 512     SP_EXPERTS 32  SP_TOPK 4  SP_SEQ 2048  SP_BATCH 8
  SP_EP 1          SP_TP 1          SP_REPLICA 1   SP_REMAT recompute_all
  SP_NO_DIST 1     SP_GROUP megagpt-sw1
)

# grid: suffix | overrides
GRID=(
  # --- E-scan (K4): per-device router/ragged-dot scaling, isolates collective-free cost ---
  "e16|SP_EXPERTS=16"
  "e32|"
  "e48|SP_EXPERTS=48"
  "e64|SP_EXPERTS=64"
  "e96|SP_EXPERTS=96"
  # --- K-scan (E32): active-compute knob ---
  "k1|SP_TOPK=1"
  "k2|SP_TOPK=2"
  "k8|SP_TOPK=8"
  # --- routing impl: does fast_qb (approx top-k) speed the sort + work on GPU? ---
  "fqb|SP_FAST_QB=1"
  "fqbE64|SP_EXPERTS=64 SP_FAST_QB=1"
  "moering|SP_MOE_IMPL=ring"
  "morata|SP_MOE_IMPL=ragged_all_to_all"
  # --- attention global:local (the 5:1 lever) ---
  "gfull|SP_GLOBAL_EVERY=1"
  "g6w1024|SP_GLOBAL_EVERY=6 SP_LOCAL_WINDOW=1024"
  "g8w512|SP_GLOBAL_EVERY=8 SP_LOCAL_WINDOW=512"
  # --- seq / batch: attn O(seq^2) + tokens/step ---
  "seq4096|SP_SEQ=4096"
  "seq1024|SP_SEQ=1024"
  "seq1024b16|SP_SEQ=1024 SP_BATCH=16"
  "b16|SP_BATCH=16"
  # --- remat ---
  "savemoe|SP_REMAT=save_moe"
)

echo "launching ${#GRID[@]} single-GPU jobs (prefix=$PREFIX)"
for spec in "${GRID[@]}"; do
  suffix="${spec%%|*}"; overrides="${spec#*|}"
  if [[ -n "$ONLY" && ",$ONLY," != *",$suffix,"* ]]; then continue; fi
  typeset -A env; env=("${(@kv)BASE}")
  if [[ -n "$overrides" ]]; then
    for kv in ${(s: :)overrides}; do env[${kv%%=*}]="${kv#*=}"; done
  fi
  args=( --gpu H100x1 --enable-extra-resources --extra gpu --cpu 12 --memory 200GB --disk 120GB
         --max-retries 0 --job-name "${PREFIX}-${suffix}"
         -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" )
  for k in ${(k)env}; do args+=( -e "$k" "${env[$k]}" ); done
  KUBECONFIG=~/.kube/coreweave-iris-gpu uv run iris --cluster=cw-us-east-02a job run --no-wait \
    "${args[@]}" -- python launch.py >/dev/null 2>&1
  echo "  submitted ${PREFIX}-${suffix}   [${overrides:-base}]"
done
echo "all ${#GRID[@]} submitted"
