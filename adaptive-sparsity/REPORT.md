# Strong sparsity in a 1B MoE: how few experts per token can you afford?

## TL;DR

A ~1.1B-total grug MoE (E=128 experts, hidden 768) trained on FineWeb-Edu is heavily
over-provisioned at this token budget. Fixed top-K train loss falls only **0.088** from K=1
(0.78% of experts active) to K=8 (6.25% active), measured iso-step at ~1.05B tokens. One
routed expert per token reaches 97% of the K=8 loss reduction.

Adaptive variable-k routing — a learned per-layer threshold trained with a straight-through
estimator and a global sparsity penalty — gives smooth control of the average active fraction
(λ from 0 to 16 sweeps it from 5.6% down to the 0.8% floor) but does not beat fixed top-K at
any matched active fraction. The λ=0 control matches fixed K=8; as λ rises, adaptive sits
above the fixed-K frontier everywhere, and the gap widens with sparsity, from ~0 at 5.6%
active to 0.156 loss at 0.8% active.

Pushing below one expert per token (no floor, K_min=0) degrades prediction sharply: loss
3.8–4.9 at under 0.5% active. At this scale and budget, plain fixed K=1 is the cheapest good
operating point; neither adaptive allocation nor sub-one-expert sparsity improves on it.

## Setup

All runs train the same grug MoE variant: a QB-routed (DeepSeek-V3-style loss-free bias
balancing) mixture of experts with GatedNorm, exclusive self-attention (XSA), sigmoid combine
weights, a router z-loss, and a float32 router. Every layer is an MoE layer; there are no
dense layers. The base configuration is copied from the `delayed-gradient-pp` grug template
and changed only where adaptive routing required it.

- **Model**: hidden 768, E=128 experts, K_max=8 dispatch capacity, ~1.1B total parameters.
  One shared expert is always active; the routed experts are what sparsity varies.
- **Data**: FineWeb-Edu, pretokenized with the `marin-community/marin-tokenizer` (vocab
  128256), pulled region-locally via `download_pretokenized_cache`
  (`marin-community/fineweb-edu-pretokenized-10B`). No held-out validation set is wired in;
  the convergence metric is `train/loss` (cross-entropy) plus the router metrics.
- **Training**: iso-token across every arm. Batch size, step count, and the AdamH learning
  rate are computed once from a fixed reference geometry (E=128, K=4) so the only variable
  across arms is routing sparsity, not the token budget or LR. Arms run 6000 steps
  (~1.6B tokens) on one `v6e-8` slice (~1.3h/arm).
- **Reported step**: train/loss at **step 4000 (~1.05B tokens)**. Some wandb streams drop
  before step 6000 (state `crashed`) even when the iris job finishes, freezing their summary
  at a smaller step; step 4000 is the largest step every arm reached, so comparing there is
  iso-step (`iso_step.py`). An earlier 10304-step (~2.7B-token) pass agrees on the trend.
- **Cluster**: marin `iris`, `v6e-8` preemptible slices, one iris job per arm via `sweep.sh`.

## Method: adaptive variable-k routing

Fixed top-K MoE spends the same number of experts on every token. Adaptive (variable-k)
routing lets the per-token expert count float between a floor and the dispatch capacity, so
the router can spend more experts on hard tokens and fewer on easy ones. The grug dispatch
path uses a fixed `[T, K_max]` shape, so variable-k is a *keep mask* on the combine weights
rather than a ragged kernel: an expert with zero combine weight contributes nothing to the
output. This is quality-sparse (the loss sees true sparsity) but still pays K_max FLOPs in
the current kernel; realized FLOP savings would need a ragged dispatch.

The keep gate (`MoEMLP._adaptive_gate`, `model.py`) works as follows. Each token selects its
top-K_max candidates on the biased router logits. A single learned per-layer scalar threshold
`θ` (`router_threshold`) decides which survive:

- **Forward (truly sparse)**: keep candidate *i* iff its biased logit clears `θ`
  (`hard_keep = topk_biased > θ`), always keeping the strongest `K_min` (the floor).
- **Backward (straight-through)**: route the gradient through a soft sigmoid surrogate
  `soft_keep = σ((topk_biased − θ)/temp)`, via
  `keep_st = hard_keep + (soft_keep − stop_gradient(soft_keep))`.

The straight-through estimator is the load-bearing design choice. Without it, `θ` sees only
the sparsity-penalty gradient — which always pushes `θ` up — and collapses to the floor
regardless of prediction quality. With it, cross-entropy can pull `θ` back down to recover an
expert that reduces loss, while the penalty pushes `θ` up.

The sparsity penalty is `sparsity_loss_coef · E[active_fraction]`, added to the training loss,
where the expected active fraction is the mean soft-keep mass over the expert pool (the
differentiable surrogate). `λ` (`SP_COEF`) sets how hard the model is pushed toward sparsity;
`K_min` (`SP_MIN_K`) sets the per-token floor. The threshold initializes low (`θ₀ = −4.0`) so
all K_max experts start active and the run anneals from dense to sparse (the ReMoE recipe).

New config fields on `GrugModelConfig`: `adaptive_routing`, `min_experts_per_token`,
`sparsity_loss_coef`, `sparsity_temp`. New logged metrics:
`train/router/sparsity/realized_active_frac`, `.../expected_active_frac`,
`.../threshold_mean`, `.../penalty_weighted`, plus per-layer thresholds.

## Result: fixed-K is nearly flat

| K | active frac | loss@4000 | CE@4000 |
|---|---|---|---|
| 1 | 0.78% | 3.3046 | 3.3027 |
| 2 | 1.56% | 3.2686 | 3.2659 |
| 4 | 3.12% | 3.2316 | 3.2276 |
| 8 | 6.25% | 3.2169 | 3.2116 |

8× more routed experts (K=1→K=8) buys 0.088 train loss. The marginal value of an expert is
~0.01 loss. This is the over-provisioned regime that Abnar et al. (2501.12370) predict at
fixed training compute, and it sets up the rest of the result: when extra experts barely help,
there is little for an adaptive router to gain by reallocating them across tokens.

## Result: adaptive variable-k (K_min=1) tracks the fixed-K frontier from above

K_max=8, K_min=1 (every token keeps at least one routed expert), penalty `λ`:

| λ | active frac | loss@4000 | CE@4000 |
|---|---|---|---|
| 0 | 5.63% | 3.2176 | 3.2124 |
| 0.1 | 4.12% | 3.2356 | 3.2271 |
| 0.25 | 1.97% | 3.3092 | 3.2928 |
| 0.5 | 0.89% | 3.3911 | 3.3640 |
| 1 | 0.81% | 3.4609 | 3.4495 |
| 4 | 0.79% | 3.4728 | 3.4407 |
| 16 | 0.80% | 4.0833 | 3.9547 |

λ gives smooth control of the average active fraction: 5.6% at λ=0, 4.1% at λ=0.1, 2.0% at
λ=0.25, then the K_min=1 floor (~0.8%) by λ=1. The λ=0 control matches fixed K=8 (3.218 vs
3.217), so the threshold gate and straight-through estimator do not themselves hurt. At every
matched active fraction, though, adaptive is worse than fixed-K, by a gap that grows as the
allocation gets sparser:

| active frac | fixed-K loss | adaptive loss | gap |
|---|---|---|---|
| ~5.6% | 3.217 (K=8) | 3.218 (λ=0) | ~0 |
| ~4.1% | ~3.227 (interp.) | 3.236 (λ=0.1) | 0.009 |
| ~2.0% | ~3.259 (interp.) | 3.309 (λ=0.25) | 0.050 |
| ~0.9% | ~3.300 (interp.) | 3.391 (λ=0.5) | 0.091 |
| ~0.8% | 3.305 (K=1) | 3.461 (λ=1) | 0.156 |

(Fixed-K losses at non-measured fractions are linearly interpolated between the bracketing K
points.) Spending more experts on some tokens and fewer on others — the adaptive premise — does
not pay off here: a uniform budget at the same average active fraction is at least as good, and
strictly better below ~4% active. This fits the over-provisioning result: when even K=1 fits
the data, there is no hard-token subset for adaptive allocation to exploit, so the variable
per-token count only adds variance and the threshold/penalty optimization only adds cost.
Beyond λ=1 the penalty keeps pushing a threshold that the floor has already pinned, and loss
degrades further (λ=16: 4.08) without buying more sparsity.

## Result: below one expert per token (K_min=0), prediction breaks

K_max=8, K_min=0 (a token may keep zero routed experts and fall back to the shared expert
alone), penalty `λ`:

| λ | active frac | loss@4000 | CE@4000 |
|---|---|---|---|
| 0 | 5.47% | 3.2174 | 3.2122 |
| 4 | 0.56% | 4.9247 | 4.8897 |
| 16 | 0.11% | 4.5482 | 4.5220 |
| 64 | 0.04% | 3.8006 | 3.7740 |
| 256 | 0.006% | 3.8881 | 3.8649 |

With no floor the penalty drives tokens to zero routed experts. Loss lands at 3.8–4.9, far
above fixed K=1 (3.305), and the ordering is non-monotonic (λ=4 is both the least sparse and
the worst), consistent with the threshold thrashing rather than tracing a clean frontier.

## Related work

- **Abnar et al., Optimal Sparsity (2501.12370)** — at fixed training compute, increasing MoE
  sparsity is loss-neutral to loss-positive up to a point. The flat fixed-K baseline here is
  that regime.
- **AdaMoE (2406.13233)**, **Harder Tasks Need More Experts (2403.07652)** — per-token
  variable expert count via null/threshold experts. The K_min=1 adaptive arm tests this
  against a fixed budget; it does not win at equal active fraction in this setup.
- **ReMoE (2412.14711)** — ReLU-routed MoE with a dense-to-sparse anneal and an L1 sparsity
  penalty. The `θ₀ = −4.0` init and `λ·E[active_fraction]` penalty follow this recipe; the
  K_min=0 collapse is the failure mode when the anneal has no floor and the loss landscape is
  flat.
- **Mixture of Parrots (2410.19034)** — train loss is not reasoning ability. These results are
  train cross-entropy on FineWeb-Edu only, with no downstream eval.

## Reproduction

```bash
# fixed-K baseline frontier (K in {1,2,4,8} at E=128)
SP_REGION=us-east5 SP_STEPS=6000 ./sweep.sh baseline
# adaptive, one-expert floor, penalty sweep
SP_REGION=us-east5 SP_STEPS=6000 ./sweep.sh kmin1
# adaptive, no floor (collapse probe)
SP_REGION=us-east5 SP_STEPS=6000 ./sweep.sh aggressive
# iso-step quality<->sparsity table from a wandb group
uv run python iso_step.py --group adaptive-sparsity-kmin1 --step 4000
```

`SP_REGION` is unpinned by default so arms take v6e capacity in any region. The FineWeb-Edu
cache is HF-backed and built per-region on first use; a large concurrent sweep landing
together in a region that has not built the cache yet races on the build and the losers hit
`MixtureDataset ... empty finite dataset`. Pinning `SP_REGION` to a region that already holds
a complete cache (us-east5) avoids the race.

## Caveats

- Train cross-entropy on FineWeb-Edu only; no held-out or downstream evaluation.
- One model scale (~1.1B total, hidden 768, E=128) and one token budget (~1.05B tokens at the
  reported step). The over-provisioning is budget-relative; the floor on useful sparsity will
  move with the token budget, and the adaptive comparison could change at a scale where extra
  experts carry more marginal loss.
- The keep-mask is quality-sparse but still pays K_max dispatch FLOPs. Wall-clock/FLOP savings
  require a ragged kernel and are not measured here.
- The penalty is a single global `λ` with a fixed schedule. A `λ` warmup, a per-layer or
  per-token target-fraction objective, or a different sparsity regularizer could behave
  differently; only the global-penalty form is tested.
- `v6e-8` preemption: preempted arms are re-submitted (`--max-retries 0`); reported numbers
  are from arms that ran past the step-4000 comparison point.
