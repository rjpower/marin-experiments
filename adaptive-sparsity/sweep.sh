#!/usr/bin/env bash
# Submit a sparsity sweep on the real marin nemotron mixture. One iris job per arm.
#
# Usage:
#   SP_TOKENS=10e9  ./sweep.sh frontier        # fixed-K: K in {1,2,4,8,16,40} at E=1024
#   SP_TOKENS=100e9 SP_TPU=v6e-32 ./sweep.sh frontier
#   SP_TOKENS=100e9 SP_TPU=v6e-32 ./sweep.sh adapt   # adaptive variable-k (K_max=16, K_min=1, penalty sweep)
#
# E=1024 reaches 1/1024 (0.098%) active at K=1 and 1/25.6 (3.9%) at K=40 -- the 1/1000..1/25
# sparsity range. Every arm is iso-token: batch/steps/LR are sized once from SP_TOKENS, so the
# only variable across arms is routing sparsity.
#
# Region: unlike the earlier FineWeb-Edu phase (HF-backed, per-region download, region-agnostic),
# the nemotron caches are pre-built in GCS and replicated across regions. We pin BOTH the run
# (--region) and the data region (SP_DATA_REGION) to us-east5 (verified complete set) so every
# read is same-region. Override with SP_REGION to use another region that holds the caches.
set -euo pipefail

MODE="${1:?usage: sweep.sh <frontier|adapt>}"
TPU="${SP_TPU:-v6e-8}"
TOKENS="${SP_TOKENS:?set SP_TOKENS, e.g. SP_TOKENS=10e9}"
REGION="${SP_REGION:-us-east5}"
EXPERTS="${SP_EXPERTS:-1024}"
HIDDEN="${SP_HIDDEN:-512}"
GROUP="${SP_GROUP:-sparsity-$MODE-E$EXPERTS-t$TOKENS}"

submit() {
  # submit <tag> <KEY VALUE>...
  local tag="$1"; shift
  echo ">>> submitting arm: $tag  (tokens=$TOKENS tpu=$TPU region=$REGION)"
  # max-retries 3: these runs checkpoint every 30 min (launch.py) and the nemotron caches are
  # pre-built (no per-region build race), so an auto-retry after preemption RESUMES from the
  # latest checkpoint instead of restarting -- essential for the multi-hour 10B-100B arms.
  uv run iris --cluster=marin job run --no-wait \
    --tpu "$TPU" --enable-extra-resources --extra marin-core:tpu --region "$REGION" \
    --max-retries 3 --cpu 32 --memory 128GB --disk 100GB \
    -e WANDB_API_KEY "$WANDB_API_KEY" \
    -e SP_TPU "$TPU" -e SP_GROUP "$GROUP" -e SP_TOKENS "$TOKENS" \
    -e SP_DATA nemotron -e SP_DATA_REGION "$REGION" \
    -e SP_HIDDEN "$HIDDEN" -e SP_EXPERTS "$EXPERTS" "$@" \
    -- python launch.py 2>&1 | grep -E 'Job submitted|Dashboard' || true
}

case "$MODE" in
  frontier)
    # Fixed top-K frontier at E=1024: active fraction 1/1024 (0.098%) -> 1/25.6 (3.9%).
    for K in 1 2 4 8 16 40; do
      submit "fixed-E${EXPERTS}-k$K" -e SPARSITY_MODE fixed -e SP_TOPK "$K"
    done
    ;;
  adapt)
    # Adaptive variable-k with a one-expert floor (K_min=1): every token keeps >=1 routed
    # expert (1/1024 floor) and the penalty trims the rest of the K_max=16 budget. Isolates
    # "spend more experts on hard tokens, fewer on easy ones" against the fixed-K frontier.
    for C in 0 0.25 1 4; do
      submit "adapt-E${EXPERTS}-k16-min1-c$C" \
        -e SPARSITY_MODE adaptive -e SP_TOPK 16 -e SP_MIN_K 1 -e SP_COEF "$C"
    done
    ;;
  *)
    echo "unknown mode: $MODE (expected frontier|adapt)"; exit 1;;
esac
echo "all $MODE arms submitted to group=$GROUP tpu=$TPU region=$REGION tokens=$TOKENS"
