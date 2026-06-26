# Throughput-First Sweep — megagpt-speedrun

**Pivot (2026-06-26):** the goal is NOT "best loss on a small token budget." It is **maximize
tokens pushed through a big, deeply-sparse MoE in 24h** (target **100B–1T tokens**), keeping the
model big+capable. Headline metric = **tokens/24h** at a defensible bpb.

## Governing equation
```
tokens/24h ≈ 86400 · MFU · P_peak / (6 · N_active)          P_peak ≈ 8 PFLOPS (8×H100 bf16)
```
Baseline (prod-pre: E64 D1536 seq4096 K8 b16): ~105k tok/s, **~5% MFU, 0.75B active → ~8B tok/24h.**
We need **12×–125×**. That comes from attacking BOTH terms:

| Target/24h | tok/s | N_active | MFU |
|---|---|---|---|
| 100B | 1.16M | ~200M | ~20% |
| 300B | 3.5M | ~150M | ~30% |
| 1T | 11.6M | ~40M | ~40% + near-zero attn |

We are **overhead-bound, not expert-bound** (E64 only 1.33× faster than E256; per-expert matmul
M≈128 ≪ 512 sweet spot; attn+lm_head+router-sort dominate). So the wins are: ↑MFU (batch, kill
attn/router overhead) and ↓N_active (deeper sparsity), while keeping total params high (capacity).

## Levers (all confirmed in code; env / field / file)
| Lever | Env (field) | Sweep values | Hypothesis |
|---|---|---|---|
| seq len | `SP_SEQ` (max_seq_len) | 4096, 2048, 1024 | attn O(seq²)+KV mem; shorter → bigger batch + cheaper attn |
| batch | `SP_BATCH` / `SP_FIT_BATCH` | max-fit per cfg | **#1 MFU lever** (raises per-expert M) |
| top-K | `SP_TOPK` (num_experts_per_token) | 8, 4, 2 | N_active ∝ K → direct tok/s |
| #experts | `SP_EXPERTS` (num_experts) | 64, 256, 512, 1024 | capacity ~free for throughput (not expert-bound) |
| expert hidden | `SP_INTERMEDIATE` (intermediate_dim) | heuristic, 2·D | active ∝ K·I; total ∝ E·I |
| hidden D | `SP_HIDDEN` (hidden_dim) | 1024, 1536, 2048 | attn/dense/active cost ∝ D(²) |
| remat | `SP_REMAT` (remat_mode) | recompute_all, **save_moe** | save_moe skips re-running EP collectives in bwd (~15-20%) |
| fast QB-β | `SP_FAST_QB` (fast_qb_beta) | 0, **1** | QB top-k sort ~35% of device time; approx_max_k ~2× — **verify GPU support** |
| **global:local attn** | `SP_GLOBAL_EVERY` / `SP_LOCAL_WINDOW` (NEW) | every {4,6,8}th global; local {512,1024} | most layers cheap LOCAL window, only every Nth GLOBAL (5:1 = SP_GLOBAL_EVERY=6) — a top attn-throughput lever |
| GQA | num_kv_heads (heuristic gqa_ratio) | 4:1, 8:1 | shrinks KV mem → bigger batch (compute ~same) |
| MoE dispatch | `SP_MOE_IMPL` (moe_implementation) | ring, ragged_all_to_all | EP collective backend |
| optimizer | (hardcoded → make env) | AdamH, **Muon** | quality/token (compounds; NanoGPT-speedrun staple) |
| mesh | `SP_EP` / `SP_TP` / `SP_REPLICA` | EP=8/TP=1 baseline | EP shards experts+batch |
| attn impl | `SP_ATTN` | gpu_fa4_cute | keep FA4 (8-10× vs reference); **MLA = not in code (future)** |

## Design — three tiers (budget: ≤32 H100; spr1 holds 8 as fallback)

**Tier 1 — single-GPU proxy (`H100x1`, ≤24 parallel, SYNTHETIC data).**
Tiny proxy that fits one H100 (D1024, E8, layers8, EP=1) to rank the *per-device* levers cheaply &
massively-parallel: seq, D, head_dim/GQA, remat, fast_qb, batch-memory ceiling, optimizer-step
cost. Gives relative lever ranking + the per-op profile. (Does NOT capture EP-collective overhead.)

**Tier 2 — 8-GPU full-model throughput (`H100x8`, 3 parallel, SYNTHETIC data).**
The real EP config. Sweep the full-model dials; measure steady-state `tokens_per_second`, `mfu`,
max-fit batch (OOM probe), `flops_per_token_analytic`. **This is the tokens/24h number.**

**Tier 3 — convergence (`H100x8`, REAL cwobject proofpile, ~1–2B tok).**
Top 2 Tier-2 configs × {AdamH, Muon}; measure loss/token. Final pick maximizes
**tokens/24h × quality/token**.

## Tier-2 config matrix (8-GPU, from baseline E64/D1536/seq4096/K8; one+combined dials)
| id | D | E | K | seq | I_expert | remat | fast_qb | batch | note |
|----|---|---|---|-----|----------|-------|---------|-------|------|
| T0 baseline | 1536 | 64 | 8 | 4096 | heur | recompute_all | 0 | 16 | reproduce ~8B/24h |
| T1 seq2048 | 1536 | 64 | 8 | 2048 | heur | recompute_all | 0 | max | cheaper attn → batch↑ |
| T2 seq1024 | 1536 | 64 | 8 | 1024 | heur | recompute_all | 0 | max | |
| T3 K4 | 1536 | 256 | 4 | 2048 | heur | recompute_all | 0 | max | half active, more experts |
| T4 K2 | 1536 | 512 | 2 | 2048 | heur | recompute_all | 0 | max | deep sparsity |
| T5 save_moe | 1536 | 256 | 4 | 2048 | heur | save_moe | 0 | max | bwd collective save |
| T6 fastqb | 1536 | 256 | 4 | 2048 | heur | save_moe | 1 | max | + approx QB (if GPU ok) |
| T7 D1024 | 1024 | 512 | 4 | 2048 | heur | save_moe | 1 | max | smaller active, deeper sparse |
| T8 bigbatch | 1024 | 512 | 4 | 1024 | heur | save_moe | 1 | max·2 | seq↓ → push batch hard |
| T9 E1024K2 | 1024 | 1024 | 2 | 2048 | heur | save_moe | 1 | max | max sparsity, ~30B+ total |
| T10 fatI | 1024 | 512 | 4 | 2048 | 2·D | save_moe | 1 | max | capacity via fatter experts |
| T11 D2048 | 2048 | 256 | 4 | 2048 | heur | save_moe | 1 | max | capacity via width |

## Measurement & readout
- Each benchmark: `SP_STEPS≈80`, synthetic data (no cold-start). Read steady-state (steps 40–80)
  `throughput/tokens_per_second`, `throughput/mfu` (p50), `throughput/flops_per_token_analytic`.
- Max-fit batch: launch with increasing `SP_BATCH` until OOM (RESOURCE_EXHAUSTED); record largest.
- `SP_PROFILE=1` on T0 + best config → per-op time split (attn / lm_head / ragged_dot / router-sort
  / expert-gmm / optimizer) to confirm where the budget goes.
- Compute **tokens/24h = tok/s · 86400 · 0.85** (ckpt/eval/restart overhead) + **total params**.

## Decision rule
Pick the config maximizing `tokens/24h` subject to: total params ≥ ~10B (capacity), seq ≥ 1024
(usable context), realized active-frac stable, not data-bound. Then Tier-3 convergence (Muon vs
AdamH) on that geometry → launch the **24h throughput run** (WSD cosine→0) + **SFT cooldown**.

## Status
- [x] synthetic-data benchmark mode (`SP_SYNTH_DATA=1`, `synth_data.py`) — verified locally
- [x] global/local attention lever (`SP_GLOBAL_EVERY`/`SP_LOCAL_WINDOW`, model.py) — verified
- [x] benchmark launcher `launch_bench.sh` (H100x1/x8, short synthetic, exits)
- [~] Tier 2 wave 1 LAUNCHED: bench-t0 (baseline control), bench-dsa (E512/K2/seq2048 +
      global6/local1024 + save_moe), bench-fqb (baseline + fast_qb GPU test)
- [ ] Tier 1 proxy sweep · [ ] Tier 2 full sweep · [ ] Tier 3 convergence (Muon vs AdamH)
- [ ] pick config → 24h run

Budget: ≤32 H100. spr1 (8) = fallback. 24 free → 3× 8-GPU benchmarks/round.
Muon: available in levanter.optim (muon/grug_muon/muonH); currently AdamH hardcoded in launch.py
→ Tier-3 needs a small launch.py change (optimizer select by env) before the convergence runs.
