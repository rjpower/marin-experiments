# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Single-node 8xH100 throughput model: async/sync PIPELINE parallelism vs the production
EP+FSDP baseline, for the megagpt E128/K8 MoE.

WHY a new model (vs delayed-gradient-pp/pp_throughput_model.py): that one assumes the baseline is
synchronous DATA parallelism whose cross-slice DCN gradient all-reduce (volume ~ N_total) is the thing
PP removes -- and concludes "single slice -> PP has nothing to beat." We have NO DCN (one NVLink node),
so by that model PP is moot. BUT our baseline is EP+FSDP, whose bottleneck is the **EP all-to-all
dispatch (29% of step time)** + the FSDP param all-gather -- neither of which that model captures. PP
eliminates BOTH (experts become layer-local; params are resident per stage), replacing them with thin
P2P activation sends at the 7 stage boundaries. So our single-node case is genuinely different from the
PR#11 finding ("sync PP lost 0.78x to FSDP single-host") -- because PR#11's baseline was FSDP, not EP+FSDP.

This model is PROFILE-ANCHORED (uses our measured xprof device-time fractions, not guessed NVLink GB/s)
and sweeps the genuinely-uncertain PP knobs (microbatch MFU penalty, 1F1B recompute, P2P cost, the Muon
staleness token-tax). It decides whether to BUILD real PP, it does not replace measuring it.

Run: uv run python h100_pp_model.py
"""
from dataclasses import dataclass

# --- Measured EP+FSDP step decomposition (normalized to 100), from the xprof profile ---
# profile: EP collectives 29%, optimizer/scatter 25%, attention 22%, matmul <9%, expert ~0.2%, bubble ~25%
# (those overlap/approx >100%); normalize to a clean, additive split that sums to 100:
EP = {
    "compute":   31.0,  # attention(22) + dense matmul(9) + expert(0.2)  -- the real arithmetic
    "ep_a2a":    29.0,  # EP all-to-all dispatch+combine (ring-dispatch all-gather), PER LAYER
    "fsdp_comm": 15.0,  # FSDP param all-gather + grad reduce-scatter (~half of the 25% "optimizer/scatter")
    "opt_comp":  10.0,  # AdamH/scatter compute (the other half)
    "bubble":    15.0,  # exposed/idle residual
}
EP_TOTAL = sum(EP.values())  # 100


# KEY DISTINCTION (corrected): async no-flush PP does NOT microbatch. The pipeline-depth parallelism
# comes from P FULL batches in flight at once (stage s works on batch t-(P-1-s)), so each stage runs a
# FULL-batch GEMM -> eta_mb ~= 1 (no microbatch penalty) and there is NO global flush/barrier. The cost
# is the per-stage staleness token-tax (Muon-tamed). SYNC 1F1B/ZB is the opposite: it chops the global
# batch into M microbatches to fill+drain around a per-step barrier -> pays the eta_mb microbatch penalty
# AND a (P-1)/M bubble, but is gradient-exact (no tax). The eta_mb penalty -- PR#11's "residual MFU gap"
# that sank sync single-host PP -- is therefore an ARTIFACT OF SYNC; async is built precisely to dodge it.
@dataclass(frozen=True)
class PPKnobs:
    eta_async: float   # async per-stage MFU efficiency: full-batch GEMMs -> ~1.0 (only steady-state stage imbalance)
    eta_sync: float    # sync MFU efficiency: small per-microbatch GEMMs -> <1 (the PR#11 penalty)
    recompute: float   # activation-recompute FLOP multiplier (async 2BW stores -> ~1.0; sync 1F1B may recompute)
    p2p_frac: float    # PP P2P activation-send cost as a fraction of the ORIGINAL EP step (of 100)
    bubble_sync: float # steady-state bubble fraction for SYNC PP (of compute); async no-flush = 0
    token_tax: float   # Muon per-stage(pp8) staleness token-overhead (async only); sync is tax-free


def pp_step(k: PPKnobs, async_: bool):
    """PP per-step wall-time on the EP=100 scale. PP removes ep_a2a (experts local) and fsdp_comm
    (params resident per stage); keeps opt_comp; pays P2P sends. ASYNC: full-batch GEMMs (eta_async),
    no bubble, no microbatch. SYNC: microbatch GEMMs (eta_sync) + (P-1)/M bubble, gradient-exact."""
    eta = k.eta_async if async_ else k.eta_sync
    rc = 1.0 if async_ else k.recompute        # async 2BW stores in-flight activations; sync 1F1B recomputes
    compute_pp = EP["compute"] / eta * rc
    bubble = 0.0 if async_ else k.bubble_sync
    return compute_pp * (1.0 + bubble) + k.p2p_frac + EP["opt_comp"]


def verdict(label, k: PPKnobs):
    t_async = pp_step(k, async_=True)
    t_sync = pp_step(k, async_=False)
    raw_async = EP_TOTAL / t_async
    raw_sync = EP_TOTAL / t_sync
    net_async = raw_async / k.token_tax          # async pays the staleness token-tax, NO microbatch penalty
    net_sync = raw_sync                          # sync PP is gradient-exact (no tax) but eta_sync-penalized
    def tag(x): return "WIN " if x > 1.02 else ("wash" if x > 0.98 else "LOSS")
    print(f"  {label:<28} async-PP raw {raw_async:.2f}x /tax {k.token_tax:.2f} = NET {net_async:.2f}x [{tag(net_async)}]"
          f"   | sync-PP {net_sync:.2f}x [{tag(net_sync)}]")
    return net_async, net_sync


print("=" * 100)
print("Single-node 8xH100: PIPELINE parallel vs EP+FSDP baseline (megagpt E128/K8, 7.62B/0.82B)")
print("Baseline EP+FSDP step decomposed (=100):", "  ".join(f"{k}={v:.0f}" for k, v in EP.items()))
print("PP removes ep_a2a(29) + fsdp_comm(15) = 44 of 100; cost = microbatch-MFU + recompute + P2P + bubble/tax")
print("=" * 100)

# NOMINAL: async runs full-batch GEMMs (eta_async~0.95, no microbatch, no recompute, no bubble); sync
# microbatches (eta_sync~0.80) + recompute(1.15) + bubble((P-1)/M, M=32 -> 0.22). Muon tax 1.16.
print("\nNOMINAL (async eta=0.95 / sync eta=0.80, recompute=1.15, p2p=10, sync bubble=0.22, Muon tax=1.16):")
verdict("nominal", PPKnobs(0.95, 0.80, 1.15, 10.0, 0.22, 1.16))

print("\nASYNC SENSITIVITY -- steady-state stage-imbalance MFU (full-batch GEMMs, so near 1.0):")
for eta in (1.00, 0.95, 0.90, 0.85):
    verdict(f"eta_async={eta:.2f}", PPKnobs(eta, 0.80, 1.15, 10.0, 0.22, 1.16))

print("\nASYNC SENSITIVITY -- Muon staleness token-tax (the async cost; sync pays 0):")
for tax in (1.10, 1.16, 1.23, 1.33, 1.50):
    verdict(f"token_tax={tax:.2f}", PPKnobs(0.95, 0.80, 1.15, 10.0, 0.22, tax))

print("\nASYNC SENSITIVITY -- P2P activation-send cost (fraction of EP=100):")
for p in (5.0, 10.0, 20.0, 30.0):
    verdict(f"p2p={p:.0f}", PPKnobs(0.95, 0.80, 1.15, p, 0.22, 1.16))

print("\nSYNC-PP rows (the PR#11 regime): microbatch eta + bubble, gradient-exact (no tax):")
for esync, M in ((0.90, 64), (0.80, 32), (0.70, 16), (0.60, 8)):
    verdict(f"eta_sync={esync:.2f},M={M}", PPKnobs(0.95, esync, 1.15, 10.0, (8 - 1) / M, 1.16))

print("\nPESSIMISTIC ASYNC (eta=0.85, p2p=25, tax=1.33):")
verdict("pessimistic-async", PPKnobs(0.85, 0.80, 1.15, 25.0, 0.22, 1.33))

# Sanity anchor: reproduce PR#11's MEASURED single-host sync-PP result (0.78x vs FSDP). NB their
# baseline was FSDP (no EP all-to-all); to compare like-for-like we zero the ep_a2a removal credit.
print("\nANCHOR vs PR#11 (sync-PP single-host was measured 0.78x vs an FSDP baseline):")
_save = EP["ep_a2a"]
EP["ep_a2a"] = 0.0  # FSDP baseline has no EP all-to-all for PP to remove
verdict("FSDP-baseline sync (M=8)", PPKnobs(0.95, 0.60, 1.33, 10.0, (8 - 1) / 8, 1.16))
EP["ep_a2a"] = _save

print("\n" + "=" * 100)
print("All numbers above are arithmetic on ASSUMED inputs (the EP=... decomposition + the PPKnobs).")
print("They show the regime/break-even, NOT a measurement. Real numbers require measuring the Muon")
print("staleness tax at our geometry and a real pipeline schedule's per-stage MFU + P2P cost.")
print("=" * 100)
