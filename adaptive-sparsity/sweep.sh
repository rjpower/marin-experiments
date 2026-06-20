#!/usr/bin/env bash
# Submit a sparsity sweep to the shared marin cluster. One iris job per arm.
#
# Usage:
#   ./sweep.sh baseline                  # fixed-K arms: K in {1,2,4,8} at E=128
#   ./sweep.sh aggressive                # adaptive arms: K_max=8, K_min=0, vary penalty lambda
#   SP_STEPS=5000 SP_TPU=v6e-8 ./sweep.sh baseline
#
# No --region / MARIN_PREFIX: the worker derives its own region bucket from VM
# metadata and the FineWeb-Edu cache (HF-backed) downloads there, so arms schedule
# wherever v6e capacity is. Every arm is iso-token (same batch/steps/LR); only the
# routing sparsity varies.
set -euo pipefail

MODE="${1:?usage: sweep.sh <baseline|aggressive|kmin1>}"
TPU="${SP_TPU:-v6e-8}"
GROUP="${SP_GROUP:-adaptive-sparsity-$MODE}"
STEPS_ENV=()
[ -n "${SP_STEPS:-}" ] && STEPS_ENV=(-e SP_STEPS "$SP_STEPS")

# Optional region pin. Unpinned by default (arms take v6e anywhere). The FineWeb-Edu
# cache is HF-backed and built per-region on first use; when many arms land together in
# a region that has NOT built it yet they race on the build and the losers see an empty
# dataset. For a large concurrent sweep, pin SP_REGION to a region that already holds a
# complete cache (e.g. us-east5) so every arm just reads it — no build, no race.
REGION_ARG=()
[ -n "${SP_REGION:-}" ] && REGION_ARG=(--region "$SP_REGION")

submit() {
  # submit <tag> <KEY VALUE>...
  local tag="$1"; shift
  echo ">>> submitting arm: $tag"
  # max-retries 0: on a preemptible v6e a retry restarts the run and trips the
  # MixtureDataset RESTART_STRATEGY empty-finite-dataset path; cleaner to let a
  # preempted arm fail and re-submit it than to retry into that error.
  uv run iris --cluster=marin job run --no-wait \
    --tpu "$TPU" --enable-extra-resources --extra marin-core:tpu "${REGION_ARG[@]}" \
    --max-retries 0 --cpu 32 --memory 128GB --disk 50GB \
    -e WANDB_API_KEY "$WANDB_API_KEY" \
    -e SP_TPU "$TPU" -e SP_GROUP "$GROUP" "${STEPS_ENV[@]}" "$@" \
    -- python launch.py 2>&1 | grep -E 'Job submitted|Dashboard' || true
}

case "$MODE" in
  baseline)
    # Fixed top-K sweep at E=128: active fraction 0.78% -> 6.25%.
    for K in 1 2 4 8; do
      submit "fixed-E128-k$K" -e SPARSITY_MODE fixed -e SP_EXPERTS 128 -e SP_TOPK "$K"
    done
    ;;
  aggressive)
    # Adaptive variable-k, NO floor (K_min=0): a token may keep zero routed experts and
    # fall back to the shared expert alone. Sweep the penalty lambda over a wide geometric
    # range. This is the "how far can a blunt penalty push" probe; it tends to collapse.
    for C in 0 4 16 64 256; do
      submit "adapt-E128-k8-c$C" \
        -e SPARSITY_MODE adaptive -e SP_EXPERTS 128 -e SP_TOPK 8 -e SP_MIN_K 0 -e SP_COEF "$C"
    done
    ;;
  kmin1)
    # Adaptive variable-k WITH a one-expert floor (K_min=1): every token keeps >=1 routed
    # expert (0.78% floor) and the penalty only trims the other 7 slots. This isolates the
    # "spend more experts on hard tokens, fewer on easy ones" question against the fixed-K
    # frontier, without the total-collapse failure mode of K_min=0.
    for C in 0 1 4 16; do
      submit "adapt-E128-k8-min1-c$C" \
        -e SPARSITY_MODE adaptive -e SP_EXPERTS 128 -e SP_TOPK 8 -e SP_MIN_K 1 -e SP_COEF "$C"
    done
    ;;
  *)
    echo "unknown mode: $MODE (expected baseline|aggressive|kmin1)"; exit 1;;
esac
echo "all $MODE arms submitted to group=$GROUP tpu=$TPU"
