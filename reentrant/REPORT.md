# Re-entrant (Self-Looping) Transformers at d512: A Negative Result

**Marin Research Report · June 2026**

---

## TL;DR

We ran eight experiments (E0–E7) testing whether a weight-tied, self-looping transformer — one that re-applies its own core layers at inference time instead of emitting chain-of-thought tokens — can improve loss by "thinking longer." At the d512 MoE-64 compute-optimal point, no variant beats the dense baseline (E0, paloma 3.818). Test-time depth scaling exists within the training range but is weak (R=2→8 buys ~0.02 paloma) and does not extrapolate: the loop drifts off-manifold past its training range. Every targeted fix — a contraction penalty, depth-conditioned routing, per-depth supervision, and a learned halting head — failed in a characteristic and mutually consistent way. The one transferable positive: anytime-decodability (E4, partially E7) is cheap and works, useful if graceful early-exit were the goal.

---

## 1. Motivation

Standard test-time scaling spends extra compute by generating more tokens: chain-of-thought, reasoning traces, search. A re-entrant (or "looped") model instead spends extra compute by re-applying layers to the same positions, accumulating in latent space. The appeal: it decouples effective depth from parameter count, avoids verbosity, and could enable per-token adaptive compute by halting at different depths for different inputs.

The theoretical case is real. Saunshi et al. (2025, arXiv:2502.17416) show that a k-layer transformer looped L times can simulate L steps of chain-of-thought, and empirically nearly matches a kL-layer model on reasoning tasks despite far fewer parameters. Geiping et al. (2025, arXiv:2502.05171) — "Huginn" — trained a 3.5B looped model that improves on reasoning benchmarks by running more loops at inference. PonderNet (Banino et al., 2021, arXiv:2107.05407) and Adaptive Computation Time (Graves, 2016, arXiv:1603.08983) supply principled learned-halting heads. The Universal Transformer (Dehghani et al., 2018, arXiv:1807.03819) established that weight-tied depth-via-recurrence is trainable.

Our goal was to evaluate whether this holds in the Marin Grug MoE setting at a fixed compute budget (the d512 compute-optimal point), and specifically whether naive randomized-depth training produces a model that benefits from extra test-time loops. The short answer is no, and the structure of the failure is informative.

---

## 2. Setup

All experiments share the following:

- **Architecture baseline:** d512 MoE-64 Grug (the compute-optimal point from the Marin v16 isoflop sweep; README d512 → paloma 3.8104).
- **Re-entrant variants:** a prelude → shared recurrent core → coda structure. Prelude and coda are unique blocks; the core is one block applied R times (strict weight-tying). Effective depth = 1 (prelude) + 1·R (core loops) + 1 (coda) = R+2.
- **Training budget:** 6387 steps × batch 32 × seq 4096 ≈ 8.4×10⁸ tokens, 2.19×10¹⁷ FLOPs.
- **Optimizer:** AdamH heuristic (Marin default).
- **Data:** Nemotron mix with default validation.
- **Platform:** one v5p-8 TPU slice, us-central1 (region-local to the data bucket; marin Iris cluster).
- **Primary metric:** `eval/paloma/macro_loss` (lower is better), consistent with the Grug benchmark series.
- **What "winning" means:** any variant that keeps improving paloma as R increases at inference, especially past the max trained depth.

Each experiment changes exactly one thing vs. its predecessor. Experiments were gated: if a variant was clearly broken or >10% worse at step 1600 (~25% of training), it was killed.

---

## 3. Results

### 3.1 Summary table

| ID | What changes | Paloma | vs. E0 | Verdict |
|----|-------------|--------|--------|---------|
| **E0** | Dense baseline (6 independent blocks, R=1) | **3.818** | — | Reference |
| **E1** | Re-entrant: 3 unique blocks looped to eff. depth 6 (~½ params) | 3.905 | +0.087 | Looping recovers most but not all of independent params |
| **E2** | E1 + FiLM/adaLN conditioning on loop index | 3.908 | +0.090 | Neutral — iteration-conditioning adds nothing at d512 |
| **E3** | E1 + randomized depth training (R∈{2,4,8}/step) | 3.937 @R=8 | +0.119 | Weak in-range scaling, no extrapolation |
| **E5** | E3 + core-consistency/contraction penalty | 4.085 best | +0.267 | Fixes drift by freezing the core — large net loss |
| **E6** | E3 + depth-conditioned MoE routing (f_t ≠ f) | 3.940 @R=4 | +0.122 | Neutral — same U-curve as E3 |
| **E4** | E3 + per-iteration readout supervision (anytime CE) | 3.995 @R=4 | +0.177 | Anytime-decodable, but +0.058 worse than E3 at all depths |
| **E7** | E3 + PonderNet learned halting head | 3.980 @R=4 | +0.162 | Halts at floor (expected depth ~1); model chooses min compute |

No variant beats E0. The gap between E0 and the best re-entrant model (E1, +0.087) is not closed by any subsequent experiment.

### 3.2 The depth-scaling failure: E3 sweep

E3 is the central experiment. It trains with a randomly sampled loop count R∈{2,4,8} per step, following the Huginn recipe, so that the model learns to produce good outputs at multiple depths. One checkpoint (step 6387) was evaluated at nine depths R=1..32 by swapping the static `recurrence_steps` argument (the parameter tree is depth-independent):

| R | paloma | Note |
|---|--------|------|
| 1 | 4.759 | Collapse — R=1 was never in the training distribution |
| 2 | 3.959 | Bottom of trained range |
| 4 | 3.938 | |
| 6 | 3.938 | (interpolation) |
| **8** | **3.937** | **Global min = top of training range** |
| 12 | 3.938 | Extrapolation begins to degrade |
| 16 | 3.940 | |
| 24 | 3.946 | |
| 32 | 3.956 | Back to ≈R=2, whole within-range gain erased |

**Takeaway (E3 sweep).** The curve is U-shaped, with its minimum at exactly R=8 (the max trained depth). Within-range depth scaling is real but small: R=2→8 buys 0.022 paloma. Past R=8, the loss rises monotonically — at R=32 the full within-range gain is gone. The looped map is not contractive toward a useful fixed point; extra iterations drift off-manifold. E3's best depth (R=8, paloma 3.937) still trails E1 (3.905) and E0 (3.818).

### 3.3 E5: the consistency penalty confirms the mechanism

E5 adds a training-only penalty on the normalized squared delta between consecutive core-loop states — a soft contraction constraint — at two strengths (λ=300, λ=3000):

| R | E3 (no penalty) | E5 λ=300 | E5 λ=3000 |
|---|----------------|----------|-----------|
| 1 | 4.759 | 4.099 | 4.114 |
| 2 | 3.959 | 4.097 | 4.116 |
| 4 | 3.938 | 4.092 | 4.114 |
| **8** | **3.937** | 4.087 | 4.114 |
| 16 | 3.940 | **4.085** | 4.114 |
| 32 | 3.956 | 4.096 | **4.114** |
| spread (R≥2) | 0.019 | 0.014 | **0.002** |

The consistency penalty is a clean "drift ↔ base quality" knob. λ=300 extends the usable-depth window; λ=3000 makes the curve nearly flat (spread 0.002 across R=1..32). But the optimizer drives the penalty term to ~0 within ~100 steps by making the core contractive — i.e. by making each loop iteration a near-identity transformation. The entire E5 curve sits 0.15–0.18 above E3 at every depth. The best E5 point (λ=300, R=16, paloma 4.085) is +0.148 above E3's best (3.937). The mechanism the user hypothesized is correct — a contractive map does produce a stable fixed point — but the only way to achieve it without hurting quality would require nontrivial computation at that fixed point, which the residual-core architecture has no way to provide.

### 3.4 E6: depth-conditioned routing (neutral)

E6 adds a learned per-(iteration, layer, expert) additive router-logit bias, making the MoE expert mixture a function of the loop index (f_t ≠ f). This addresses the structural objection that re-applying a single fixed map cannot be expressive. Result: essentially identical to E3.

| R | E3 paloma | E6 paloma | Δ |
|---|----------|----------|---|
| 1 | 4.759 | 4.764 | +0.004 |
| 2 | 3.959 | 3.959 | +0.001 |
| 4 | 3.938 | 3.940 | +0.002 |
| **8** | **3.937** | **3.939** | +0.002 |
| 16 | 3.940 | 3.941 | +0.001 |
| 24 | 3.946 | 3.943 | **−0.003** |
| 32 | 3.956 | 3.947 | **−0.009** |

E6 has a marginally less severe deep-tail drift (R=32: 3.947 vs 3.956), which is the one direction in which it differs from E3. But the minimum remains at R=8 and the base loss at trained depths is unchanged. Making the weight-tied map genuinely depth-varying through its routing did not change the fundamental inability to extrapolate.

### 3.5 E4: anytime supervision (anytime-decodable, not better)

E4 adds a per-iteration CE term: the shared head reads off after each core loop and the loss averages across all R readouts. This is designed to make every depth usable and incentivize refinement. Result: it achieved the first goal but not the second.

| R | E3 paloma | E4 paloma | Δ |
|---|----------|----------|---|
| **1** | 4.759 | **4.086** | **−0.674** |
| 2 | 3.959 | 3.998 | +0.039 |
| 4 | 3.938 | **3.995** | +0.057 |
| 8 | **3.937** | 3.995 | +0.058 |
| 16 | 3.940 | 3.996 | +0.057 |
| 32 | 3.956 | 4.005 | +0.049 |

At R=1, E4 is usable (paloma 4.086) where E3 collapses (4.759). Training every iteration's readout to be decodable buys −0.67 at R=1, which is real and useful if graceful early-exit were the goal. But E4 is +0.058 worse than E3 at every trained depth (R≥2), the in-range scaling slope is flatter (R=2→8 buys 0.003 vs E3's 0.022), and the curve still rises past R=8. Anytime supervision spreads quality uniformly across depths rather than concentrating it at full depth.

### 3.6 E7: PonderNet learned halting (halts at the floor)

E7 adds a learned per-token halting head over the core loop (PonderNet formulation): a scalar head σ(z·w) gives a halting probability per step, inducing a geometric-like distribution over exit depth; training adds a reconstruction term and a KL penalty toward a geometric prior with mean 5 (β=0.01). The base CE on the final readout is unchanged so that paloma stays comparable across variants.

The halting head learned to halt at the floor. At the end of training, `expected_halt_step = 0.999` — approximately 1 core iteration — despite the geometric prior penalizing early halting. Halting was nearly free: the reconstruction CE at the halted depth (3.692) was close to the full-depth CE (3.631), so the halting head paid little penalty for stopping immediately. The halt weight was not a stuck-at-init artifact: `params/halt_head` = 0.633.

The depth sweep confirms that E7 even over-halts relative to its own loss-optimum:

| R | E3 | E4 (anytime) | E6 (route) | **E7 (ponder)** |
|---|----|----|----|----|
| 1 | 4.759 | 4.086 | 4.764 | **4.332** |
| 2 | 3.959 | 3.998 | 3.959 | 3.991 |
| **4** | 3.938 | 3.995 | 3.940 | **3.980** |
| 8 | **3.937** | 3.995 | **3.939** | 3.981 |
| 16 | 3.940 | 3.996 | 3.941 | 3.984 |
| 32 | 3.956 | 4.005 | 3.947 | 3.999 |

E7's loss-optimum from the outside is R≈4 (paloma 3.980), but its own halting head stops at ~1 iteration (paloma 4.332). It trades ~0.05 paloma — over half the total within-range scaling budget — to save ~3 iterations. There is no depth at which the model's own policy chooses to loop more. The global-R sweeps of E3/E4/E6 could only show the average-R curve; E7 asks the model itself, and it answers: minimum compute.

---

## 4. Discussion

### Why the loop cannot both compute and converge

The clean structure of the failure — which holds across four targeted interventions — has a single root cause. A weight-tied residual core applied R times is computing:

```
z_{t+1} = z_t + f(z_t)
```

For this to improve with more iterations, f must be doing useful work each time (the hidden state changes in a way that lowers prediction error). For the map to be stable past the training range, f(z_t) must shrink as z_t approaches a good representation — the loop should converge. But a residual block that converges toward a useful fixed point must simultaneously (a) make z change enough to improve the representation, and (b) make z change less as it approaches the fixed point. These are compatible only if the fixed point itself is a qualitatively better representation than the input — i.e. if the loop is doing genuine computation. At d512, with the training setup used here, that condition is not met.

E5 is the clearest demonstration: the optimizer achieves convergence by making the core a near-identity transformation (each pass changes the residual stream by ~1.7%, and the penalty drives that toward zero). The loop converges, but to a trivially stable fixed point that adds nothing. E6 shows that making the routing depth-dependent does not change this — the map is more expressive per-iteration, but the underlying tension between "compute" and "converge" is the same. E4 shows that supervising every depth trades peak quality for uniformity. E7 shows that when the model itself controls its depth, it resolves the tension by choosing not to loop.

### What the Huginn recipe gets right that we did not

Huginn (Geiping et al., 2025) does work at 3.5B parameters and 800B tokens — a scale roughly two to three orders of magnitude larger in parameters, and roughly three orders of magnitude larger in tokens, than these experiments. Huginn also uses full truncated BPTT through the last k iterations and random-noise initialization of the recurrent state (not just the prelude output), which may help maintain useful gradient signal across depths. At d512 with 8.4×10⁸ tokens and no truncated BPTT (omitted as unnecessary at this scale), the model may simply not have enough capacity or data to learn a fixed point with nontrivial content. That is a plausible and honest caveat — but it is not testable within this budget without substantially scaling up.

### The one transferable finding

Anytime-decodability (E4) is cheap and real. Training with a per-iteration CE readout makes shallow depths usable without external modification: E4 at R=1 (paloma 4.086) is within 0.27 of full-depth performance, compared to E3 at R=1 (4.759, a near-collapse). The cost is +0.058 at full depth. If the goal is graceful early-exit — a model that returns a usable answer at any compute budget — this is a straightforward win. It is not a path to test-time scaling beyond the training depth.

---

## 5. Conclusion and next directions

At d512, none of the looped variants improved on the dense baseline, and the failure is structural rather than a matter of tuning. The four targeted fixes each confirmed the same diagnosis from a different angle: E5 (contraction penalty) froze the core; E6 (depth-conditioned routing) was neutral; E4 (anytime supervision) made performance uniformly mediocre and anytime-decodable; E7 (learned halting) halted at the floor. No variant extrapolates past its training depth, and the model free to choose its own depth chooses the minimum.

The thesis — "more loops = more thinking" — is not supported at this scale and setup. We are not in a position to say whether it would hold at larger scale; the gap in tokens and parameters between this study and Huginn is large enough that the question is open.

**What we would try next,** in order of expected payoff:

1. **Decoupled scratchpad/accumulator.** Give each iteration a dedicated slot in the sequence (not the main residual stream) to write to. The core reads its own previous pass's scratchpad and writes an update; the final readout takes the last scratchpad entry. This separates "doing computation" from "converging the residual," which is the tension that underlies E5/E7's failure.

2. **E4 with self-distillation.** Deep-supervise every depth, but additionally distill late-iteration logits into early iterations (train the shallow exit to predict what the deep exit would say). This gives a genuine refinement signal and a calibrated anytime curve, and may train a more useful fixed point than plain CE averaging.

3. **DEQ-style implicit fixed-point solve.** Rather than stacking iterations and backpropagating through them, solve directly for the fixed point z* = f(z*; x) using Anderson acceleration and train with implicit gradients. This keeps the fixed-point constraint without the identity-collapse failure mode of an additive penalty. It adds solver complexity and is higher-risk, but the implicit-gradient path is the one way to enforce convergence without requiring f(z*) ≈ 0.

The training-free KL-convergence early-exit (which would stop when consecutive loop states stop changing) is deprioritized: E3 shows the convergent direction drifts off-manifold, and E5 shows that the models which do converge are converging toward a degenerate fixed point.

---

## References

| # | Title | Authors / year | arXiv |
|---|-------|---------------|-------|
| 1 | Adaptive Computation Time for Recurrent Neural Networks | Graves 2016 | [1603.08983](https://arxiv.org/abs/1603.08983) |
| 2 | Universal Transformers | Dehghani et al. 2018 (ICLR'19) | [1807.03819](https://arxiv.org/abs/1807.03819) |
| 3 | ALBERT | Lan et al. 2019 (ICLR'20) | [1909.11942](https://arxiv.org/abs/1909.11942) |
| 4 | Deep Equilibrium Models | Bai, Kolter, Koltun 2019 (NeurIPS) | [1909.01377](https://arxiv.org/abs/1909.01377) |
| 5 | PonderNet: Learning to Ponder | Banino, Balaguer, Blundell 2021 | [2107.05407](https://arxiv.org/abs/2107.05407) |
| 6 | CoTFormer | Mohtashami, Pagliardini, Jaggi 2023 | [2310.10845](https://arxiv.org/abs/2310.10845) |
| 7 | Relaxed Recursive Transformers | Bae et al. 2024 (ICLR'25) | [2410.20672](https://arxiv.org/abs/2410.20672) |
| 8 | Scaling up Test-Time Compute with Latent Reasoning (Huginn) | Geiping et al. 2025 | [2502.05171](https://arxiv.org/abs/2502.05171) |
| 9 | Reasoning with Latent Thoughts: On the Power of Looped Transformers | Saunshi et al. 2025 (ICLR'25) | [2502.17416](https://arxiv.org/abs/2502.17416) |
| 10 | Mixture-of-Recursions | Bae, Kim, et al. 2025 (NeurIPS'25) | [2507.10524](https://arxiv.org/abs/2507.10524) |
