# Strong sparsity in a ~1B grug MoE: how few experts per token, and is the speedup real?

## TL;DR

We asked how aggressively a ~1B-class grug MoE can be sparsified — how few routed experts
per token it can run before train loss degrades — and whether that sparsity buys real
throughput. The answer across two model geometries (E=128 and E=1024), two datasets
(FineWeb-Edu and nemotron_mix), and ~1–10B tokens:

- **The fixed top-K loss frontier is shallow.** Going from 1 to 16 routed experts buys
  0.07–0.25 train-loss nats depending on geometry. At E=128 on FineWeb-Edu, K=1→K=8 buys
  0.088. At E=1024 on nemotron, K=1→K=16 buys 0.25 but at under 1.6% of experts active. K=1
  is a good operating point in every setup we ran.
- **The "0.1% activation → ~1000× faster" intuition does not hold at small hidden size.**
  Throughput is bound by the LM head, attention, and the router top-K sort, not the experts.
  The FLOP ceiling from sparsity is **1.33×** at the thin geometry (D=512, E=1024,
  I_expert=256) and **~2.9×** at a fat geometry (D=1024, E=32, I_expert=1024). The router
  top-K is ~35% of device time; `jax.lax.approx_max_k` for the loss-free balancing top-K is
  the ~2× lever that does materialize.
- **Adaptive variable-K gives a smooth active-fraction knob.** A learned per-layer threshold
  with a straight-through estimator and a global sparsity penalty sweeps the active fraction
  continuously. On nemotron at E=1024, λ=0.25 reaches near-K=16 loss (2.885 at 1.07% active
  vs K=16's 2.857 at 1.56%) — adaptive is competitive with the fixed-K frontier here. On
  FineWeb-Edu at E=128 it sat strictly above the frontier. Finer-grained experts help
  adaptive.
- **A K-curriculum does not pay off.** Training at K=1 for 80% of steps, then ramping
  K=1→2→4→8→16 over the last 20%, lands above the fixed-K frontier where it would matter. In
  the fat geometry (experts expensive), the curriculum averages ~2.3 active experts and ends
  at 2.959 — worse than fixed-K=4 (2.929) and short of K=16 (2.910). In the thin geometry, the
  late ramp does help, but experts are cheap there, so there is little compute to save.

The through-line: at ~1B params and ~1–10B tokens the model is over-provisioned, so extra
experts barely reduce loss. When extra experts barely help, neither a per-token adaptive
budget nor a late curriculum has much to reallocate, and the throughput win has to come from
the kernel (`approx_max_k`, ragged dispatch), not the routing policy.

## Setup

All runs train the same grug MoE variant: a QB-routed (DeepSeek-V3-style loss-free bias
balancing) mixture of experts with GatedNorm, exclusive self-attention (XSA), sigmoid combine
weights, a router z-loss, and a float32 router. Every layer is an MoE layer; there are no dense
layers. One shared expert is always active; the routed experts are what sparsity varies. The
base configuration is the `delayed-gradient-pp` grug template, changed only where adaptive
routing, the curriculum, or the geometry sweep required it.

We ran in two regimes, in this order:

1. **FineWeb-Edu, E=128, hidden 768** (~1.1B total). Initial baseline and the first adaptive
   sweep. Data: `marin-community/fineweb-edu-pretokenized-10B`, HF-backed and rebuilt
   per-region on first use, so arms take v6e capacity in any region. Arms run 6000 steps
   (~1.6B tokens) on one `v6e-8` slice; reported at step 4000 (~1.05B tokens), the largest step
   every arm reached.
2. **nemotron_mix, E=1024, D=512** (the "thin" geometry) **and E=32, D=1024** (the "fat"
   geometry). Data: the region-replicated `nemotron_mix` tokenized cache (7 nemotron_cc
   quality splits + starcoderdata + proofpile_2, llama3 vocab 128256). Arms run 9537 steps
   (~10B tokens) on one `v6e-8` slice, reported iso-step at step 9500.

Training is iso-token across every arm within a regime: batch size, step count, and the AdamH
learning rate are computed once from a fixed reference geometry, so the only variable across
arms is routing sparsity. We compare `train/loss` (cross-entropy); no held-out or downstream
eval is wired in.

**Iso-step caveat and `iso_step.py`.** Some wandb streams drop before the final step (state
`crashed`) even when the iris job finishes, freezing their summary at a smaller step. We pull
full history and read every arm at a common reference step (`iso_step.py`). For the thin
nemotron frontier this matters: K1/K2/K4/K8 froze between steps 8333 and 8911 while K16 and
the curriculum ran to 9500, so the thin endpoint comparisons carry a "fixed-K baselines are
slightly stale" caveat (their true final loss is a little lower than the frozen value, since
loss was still descending). The fat-geometry arms all logged cleanly to 9500.

### The two geometries differ in what an expert costs

The thin and fat geometries were chosen to move routed experts from a negligible FLOP fraction
to a real compute lever:

| geometry | D | E | I_expert | routed FLOP frac @K=1 | routed FLOP frac @K=16 | K=1 vs K=16 FLOPs |
|---|---|---|---|---|---|---|
| thin | 512 | 1024 | 256 | 2.2% | 26.5% | 1.33× |
| fat | 1024 | 32 | 1024 | ~13% | ~50% | ~2.9× |

In the thin geometry, sparsity is almost free but also almost worthless as a speedup lever:
K=1 is only 1.33× cheaper than K=16. In the fat geometry an expert is worth ~2.9× — running
K=1 instead of K=16 is a ~2.9× FLOP saving — so the fat geometry is where a sparsity-saving
policy (adaptive or curriculum) has something to win.

## Result 1: the fixed-K frontier is shallow

**Takeaway: more routed experts monotonically lower loss, but the marginal value of an expert
is ~0.005–0.02 nats. The model is over-provisioned at this budget.**

FineWeb-Edu, E=128, step 4000 (~1.05B tokens):

| K | active frac | loss | CE |
|---|---|---|---|
| 1 | 0.78% | 3.3046 | 3.3027 |
| 2 | 1.56% | 3.2686 | 3.2659 |
| 4 | 3.12% | 3.2316 | 3.2276 |
| 8 | 6.25% | 3.2169 | 3.2116 |

8× more routed experts (K=1→K=8) buys 0.088 loss. One routed expert per token reaches 97% of
the K=8 loss reduction.

nemotron, thin E=1024, step 9500 (frozen step in parens where the stream crashed early):

| K | active frac | loss | (step) |
|---|---|---|---|
| 1 | 0.0977% | 3.1099 | 8333 |
| 2 | 0.195% | 3.0138 | 8848 |
| 4 | 0.391% | 2.9989 | 8517 |
| 8 | 0.781% | 2.9769 | 8911 |
| 16 | 1.5625% | 2.8574 | 9500 |

K=1→K=16 buys 0.25 loss here — a steeper frontier than E=128, because at 1024-way granularity
each expert is narrow and K=1 (0.0977% of params active) is genuinely starved. But the whole
sweep happens under 1.6% active.

nemotron, fat E=32 I=1024, step 9500 (all arms clean):

| K | active frac | loss |
|---|---|---|
| 1 | 3.125% | 2.9839 |
| 4 | 12.5% | 2.9287 |
| 16 | 50% | 2.9103 |

K=1→K=16 buys 0.074 loss for 16× the routed FLOPs. The marginal value of an expert is ~0.005
nats. This is the over-provisioned regime Abnar et al. (2501.12370) predict at fixed training
compute, and it frames the rest of the report: when extra experts barely help, there is little
for an adaptive router or a curriculum to gain by reallocating them.

## Result 2: the speedup is geometry-blocked, and bound by the router sort

**Takeaway: at D=512, sparsity cannot give a large speedup. The LM head (61% of K=1 FLOPs),
attention (27%), and the router top-K sort dominate; the routed experts are 2.2% of FLOPs at
K=1. The sparsity ceiling is 1.33× (thin) to ~2.9× (fat), not ~1000×.**

Per-token forward FLOP fractions, thin geometry (D=512, E=1024, I_expert=256, seq 4096, vocab
128256), cross-checked against `levanter.utils.flop_utils.lm_flops_per_token`:

| component | K=1 | K=4 | K=16 |
|---|---|---|---|
| lm_head (D×vocab) | 61.4% | 57.6% | 46.1% |
| attention (QKVO + scores/AV) | 27.3% | 25.6% | 20.5% |
| shared expert (always-on) | 4.4% | 4.1% | 3.3% |
| router proj (D×E, dense) | 2.9% | 2.8% | 2.2% |
| routed experts | 2.2% | 8.3% | 26.5% |
| gated-norms | 1.7% | 1.6% | 1.3% |

The fixed floor (LM head + attention + shared + router + norms) is 97.8% of K=1 FLOPs and
never shrinks with K, so K=1 vs K=16 is only 1.33×. The "1000×" would need a geometry where
routed experts are most of the FLOPs — large D and intermediate, small relative vocab and seq.
The fat geometry moves partway there (routed 13%→50%), lifting the K=1-vs-K=16 ceiling to
~2.9×, which is why we built it for the curriculum and adaptive comparisons.

Measured throughput at K=1, batch 256 was 0.55M tok/s, MFU ≈ 4.8% — the model demands so few
FLOPs/token that it is not compute-bound. Profiling put the router's `top_k` sorts at ~35% of
device time. Swapping the loss-free-balancing top-K to `jax.lax.approx_max_k` (`fast_qb_beta`,
opt-in, default off) is the ~2× throughput lever that actually exists at this geometry; the
expert FLOPs are not where the time goes. (Implemented and committed; a clean
throughput/loss-parity A/B on TPU is not yet run — see caveats.)

## Result 3: adaptive variable-K — a smooth knob, competitive at E=1024

**Takeaway: the learned-threshold gate gives continuous control of the active fraction. On
nemotron E=1024, λ=0.25 reaches near-K=16 loss at ~2/3 the active fraction. On FineWeb-Edu
E=128 it never beat fixed-K. Finer-grained experts make adaptive competitive.**

Fixed top-K spends the same number of experts on every token. Adaptive (variable-K) routing
lets the per-token count float between a floor `K_min` and the dispatch capacity `K_max`. The
grug dispatch path uses a fixed `[T, K_max]` shape, so variable-K is a keep mask on the combine
weights, not a ragged kernel: a dropped expert contributes nothing to the output and to the
loss, but still costs K_max dispatch FLOPs. The gate (`MoEMLP._adaptive_gate`, `model.py`):

- **Forward (truly sparse):** keep candidate *i* iff its biased router logit clears a learned
  per-layer scalar threshold θ, always keeping the strongest `K_min`.
- **Backward (straight-through):** route the gradient through a soft sigmoid surrogate
  `σ((logit − θ)/temp)`, so cross-entropy can pull θ down to recover an expert that lowers
  loss while the sparsity penalty pushes θ up.

The straight-through estimator is load-bearing. Without it θ sees only the penalty gradient,
which always pushes θ up, and the gate collapses to the floor regardless of prediction quality.
The penalty is `λ · E[active_fraction]` added to the loss; θ initializes low (θ₀ = −4.0) so all
K_max experts start active and the run anneals dense→sparse (the ReMoE recipe).

nemotron, thin E=1024, K_max=16, K_min=1, step 9500:

| λ | active frac | loss |
|---|---|---|
| 0 | 1.5625% | 2.8604 |
| 0.25 | 1.0704% | 2.8853 |
| 1 | 0.5402% | 2.9581 |
| 4 | 0.1077% | 3.2475 |

λ=0 matches fixed-K=16 (2.860 vs 2.857), so the gate and straight-through estimator do not
themselves hurt. λ=0.25 holds 2.885 at 1.07% active. A fixed-K budget interpolated to 1.07%
active (between K=8 and K=16) sits near 2.93, so adaptive is on or slightly below the fixed-K
frontier here — within the stale-baseline caveat, a tie-to-small-win. λ=4 over-penalizes and
collapses toward the K=1 starved regime (3.25).

FineWeb-Edu, E=128, K_max=8, K_min=1, step 4000:

| λ | active frac | loss |
|---|---|---|
| 0 | 5.63% | 3.2176 |
| 0.1 | 4.12% | 3.2356 |
| 0.25 | 1.97% | 3.3092 |
| 0.5 | 0.89% | 3.3911 |
| 1 | 0.81% | 3.4609 |
| 4 | 0.79% | 3.4728 |
| 16 | 0.80% | 4.0833 |

At E=128 adaptive sits above the fixed-K frontier everywhere, and the gap widens with sparsity
(from ~0 at 5.6% active to 0.156 at 0.8% active). With only 128 coarse experts there is no
hard-token subset for per-token allocation to exploit, so the variable count adds variance
without a matching gain. Moving to 1024 fine-grained experts is what makes adaptive competitive
in Result 3's first table.

Pushing below one expert per token (K_min=0) breaks prediction at E=128: with no floor the
penalty drives tokens to zero routed experts and loss lands at 3.8–4.9 (vs fixed-K=1's 3.305),
non-monotonic in λ — the threshold thrashes rather than tracing a clean frontier.

## Result 4: the K-curriculum does not pay off

**Takeaway: train cheap-then-wide does not recover wide-everywhere quality. In the fat
geometry the curriculum (avg ~2.3 active) ends worse than fixed-K=4. The swap mechanism is
sound; the schedule is the problem.**

The curriculum idea: train most of the run at K=1 (cheap), then ramp K up over the last stretch
to recover the quality of a wide-K run on the full expert set. We schedule the routed top-K
width over training and rebuild the model and optimizer at each phase boundary
(`_swap_active_k`, `train.py`): `dataclasses.replace` to the new K, re-`init` the Transformer
and optimizer to the new treedef, and transplant the trained arrays by flatten order.

The mechanism is sound. On a thin run, the k1→k2 swap at step 7630 logged loss 3.20 → 3.20 —
continuous across the swap, so the array transplant preserves trained state. The cost is one
`train_step` recompile per phase (one step at ~64s vs ~1.9s/it steady-state).

We ran the schedule K=1 for 80% of steps, then 5% each at K=2, K=4, K=8, K=16 (average ≈ 2.3
active experts), against the fixed-K frontier in both geometries.

Fat E=32 I=1024, step 9500 (clean):

| arm | avg active K | loss |
|---|---|---|
| fixed K=1 | 1 | 2.9839 |
| fixed K=4 | 4 | 2.9287 |
| curriculum 1→16 | ~2.3 | 2.9594 |
| fixed K=16 | 16 | 2.9103 |

The curriculum lands between fixed-K=1 and fixed-K=4. It beats K=1 but loses to fixed-K=4
(2.959 vs 2.929), and is well short of K=16 (2.910). For its ~2.5× training-FLOP saving over
always-K=16, it gives up 0.049 nats to K=16 and 0.031 to a same-ballpark-compute fixed-K=4. The
5%-of-training tail at the wider K values is too short to recover the wide-K loss. The
curriculum is below the fixed-K frontier exactly in the geometry where saving expert compute
would matter.

Thin E=1024, step 9500: the curriculum ends at 2.9497, between fixed-K=8 (2.9769, frozen
@8911) and fixed-K=16 (2.8574). At the common step 8300 the curriculum reads 3.1202 — still
mid-ramp, tracking a low-K run, because 80% of its training was at K=1. Its loss then drops
3.12→2.95 over the final ramp, a steeper late descent than the fixed arms, so the ramp does
help here. But the thin geometry is where an expert is worth only 1.33×, so training mostly at
K=1 saves almost nothing — the curriculum helps precisely where the savings are negligible, and
fails to recover precisely where the savings (2.9×) would matter.

## What we would try next

- A longer wide-K tail (e.g. last 40–50% at K=16, not 5%) in the fat geometry, to test whether
  the curriculum can reach the fixed-K=16 floor at a still-reduced average K. As run, the tail
  is too short.
- The fat geometry has no adaptive-routing arm yet. On the thin geometry adaptive (λ=0.25)
  was the best sparsity-saving lever; whether it beats fixed-K in the fat geometry, where
  experts are expensive enough for the saving to count, is the natural next comparison.
- A clean TPU A/B for `approx_max_k` (`fast_qb_beta`): throughput delta and loss parity vs the
  exact top-K. The FLOP analysis says it is the ~2× lever; it is implemented but not yet
  benchmarked end-to-end.
- Downstream eval. Every number here is train cross-entropy; whether the shallow loss frontier
  implies a shallow capability frontier is untested (Mixture of Parrots, 2410.19034).

## Related work

- **Abnar et al., Optimal Sparsity (2501.12370)** — at fixed training compute, increasing MoE
  sparsity is loss-neutral to loss-positive up to a point. The flat fixed-K baselines here are
  that regime.
- **AdaMoE (2406.13233)**, **Harder Tasks Need More Experts (2403.07652)** — per-token variable
  expert count via null/threshold experts. The K_min=1 adaptive arms test this against a fixed
  budget; it ties-to-wins at E=1024, loses at E=128.
- **ReMoE (2412.14711)** — ReLU-routed MoE with a dense-to-sparse anneal and an L1 penalty. The
  θ₀ = −4.0 init and λ·E[active_fraction] penalty follow this recipe; the K_min=0 collapse is
  the failure mode when the anneal has no floor and the loss landscape is flat.
- **DeepSeek-V3 (2412.19437)** — the loss-free, bias-based load balancing the grug router uses;
  the router top-K over the biased logits is the sort that dominates device time at D=512.
- **Mixture of Parrots (2410.19034)** — train loss is not reasoning ability. These results are
  train cross-entropy only.

## Reproduction

```bash
# nemotron thin E=1024 fixed-K frontier (K in {1,2,4,8,16}); pin a region with a valid cache
SP_DATA_REGION=us-east5 SP_STEPS=9537 ./sweep.sh frontier
# adaptive variable-k, K_min=1 penalty sweep
SP_DATA_REGION=us-east5 SP_STEPS=9537 ./sweep.sh adapt
# fat-expert geometry (D=1024 E=32 I=1024, batch 128): fixed K + curriculum
SP_INTERMEDIATE=1024 SP_BATCH=128 SP_DATA_REGION=us-east5 ./curriculum.sh
# iso-step quality<->sparsity table from a wandb group
uv run python iso_step.py --group sparsity-frontier-E1024-t10e9 --step 9500
```

The `nemotron_mix` cache is content-addressed and region-replicated, but not byte-identical
across regions: `gs://marin-eu-west4` has divergent/partial `starcoderdata` and `proofpile_2`
replicas that crash cache load with `EOF while parsing` (filed as marin-community/marin #6530).
Of the three v6e zones (europe-west4-a, us-east1-d, us-east5-b), only us-east5 holds a cache
that loads cleanly, so we pin `SP_DATA_REGION=us-east5` for nemotron arms.

## Caveats

- Train cross-entropy on FineWeb-Edu / nemotron_mix only; no held-out or downstream evaluation.
- Two model scales and two token budgets (~1.05B tokens at E=128 step 4000; ~10B at E=1024/E=32
  step 9500). The over-provisioning is budget-relative; the useful-sparsity floor and the
  adaptive/curriculum comparisons could move at a budget where extra experts carry more
  marginal loss.
- The thin nemotron fixed-K streams crashed between steps 8333 and 8911 while the curriculum and
  K=16 ran to 9500, so the thin endpoint comparisons read fixed-K slightly high. The fat
  geometry arms all logged to 9500 and carry no such caveat; the headline curriculum and
  frontier conclusions rest on the fat arms.
- The adaptive keep-mask and the curriculum K-swap are quality-sparse but the keep-mask still
  pays K_max dispatch FLOPs. Realized wall-clock/FLOP savings need a ragged kernel and are not
  measured here.
- The AdamH learning rate is sized once from a reference geometry; the fat arms (batch 128) run
  with an LR slightly high for their batch, but identically across all fat arms, so the
  frontier shape and curriculum comparison hold even though the absolute loss floor shifts.
- `v6e-8` is preemptible; preempted arms resume from checkpoint. The fat-K=16 arm was preempted
  twice and reported numbers are read iso-step after it completed.
