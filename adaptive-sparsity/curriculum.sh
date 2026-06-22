#!/usr/bin/env bash
# Active-expert curriculum experiment: does ramping the routed-expert width k late in
# training recover near-K16 quality at ~K1 cost? Launches three iso-token arms in one
# wandb group so iso_step.py can compare them directly:
#
#   fixed K=1   -- the cheap floor (worst loss, least compute)
#   curriculum  -- K=1 for 80% of tokens, then 2/4/8/16 over the last 20% (one run,
#                  the dispatch width is actually widened in place; see train._swap_active_k)
#   fixed K=16  -- the expensive ceiling (best loss, most compute)
#
# Same data / budget / geometry as the M4 fixed-K frontier and the M5 adaptive sweep
# (nemotron, 10B tokens, E=1024, hidden=512), so the curriculum overlays on that frontier.
#
# Usage:
#   SP_TOKENS=10e9 ./curriculum.sh
#   SP_TOKENS=10e9 SP_SCHED="1:0.8,2:0.05,4:0.05,8:0.05,16:0.05" ./curriculum.sh
set -euo pipefail

TPU="${SP_TPU:-v6e-8}"
TOKENS="${SP_TOKENS:?set SP_TOKENS, e.g. SP_TOKENS=10e9}"
REGION="${SP_REGION:-us-east5}"
DATA_REGION="${SP_DATA_REGION:-$REGION}"   # GCS data bucket region; decouple from TPU region
                                           # (e.g. TPU europe-west4 reads bucket marin-eu-west4)
EXPERTS="${SP_EXPERTS:-1024}"
HIDDEN="${SP_HIDDEN:-512}"
INTER="${SP_INTERMEDIATE:-0}"   # 0 = heuristic I=D/2 (thin); set e.g. 2048 for fat experts
BATCH="${SP_BATCH:-}"           # empty = heuristic batch; set e.g. 128 to cut HBM (desyncs LR uniformly)
SCHED="${SP_SCHED:-1:0.8,2:0.05,4:0.05,8:0.05,16:0.05}"
GROUP="${SP_GROUP:-sparsity-curric-E$EXPERTS-t$TOKENS}"

submit() {
  # submit <tag> <KEY VALUE>...
  local tag="$1"; shift
  echo ">>> submitting arm: $tag  (tokens=$TOKENS tpu=$TPU region=$REGION)"
  uv run iris --cluster=marin job run --no-wait \
    --tpu "$TPU" --enable-extra-resources --extra marin-core:tpu --region "$REGION" \
    --max-retries 3 --cpu 32 --memory 128GB --disk 100GB \
    -e WANDB_API_KEY "$WANDB_API_KEY" \
    -e SP_TPU "$TPU" -e SP_GROUP "$GROUP" -e SP_TOKENS "$TOKENS" \
    -e SP_DATA nemotron -e SP_DATA_REGION "$DATA_REGION" \
    -e SP_HIDDEN "$HIDDEN" -e SP_EXPERTS "$EXPERTS" -e SP_INTERMEDIATE "$INTER" \
    ${BATCH:+-e SP_BATCH $BATCH} "$@" \
    -- python launch.py 2>&1 | grep -E 'Job submitted|Dashboard' || true
}

# Fixed-K frontier: cheap floor (K=1), a mid point (K=4), and the high-K ceiling (K=16).
submit "fixed-k1"  -e SPARSITY_MODE fixed -e SP_TOPK 1
submit "fixed-k4"  -e SPARSITY_MODE fixed -e SP_TOPK 4
submit "fixed-k16" -e SPARSITY_MODE fixed -e SP_TOPK 16
# The curriculum (SP_CURRICULUM forces fixed routing and starts at the first phase's k).
submit "curric"    -e SP_CURRICULUM "$SCHED"

echo "all curriculum arms submitted to group=$GROUP tpu=$TPU region=$REGION tokens=$TOKENS"
