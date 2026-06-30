# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Next-run decision table from MEASURED smoke data (sc1/sc2/sc3, new-CE, 8xH100).

Unlike design_scan.py (analytic floor), this uses the real tok/s + MFU + fit verdict from the
scale smoke sweeps and computes tokens x params over a real 24h budget (with an SFT-cooldown
holdout). The headline question -- "what model, 4x tokens*params over spr1, best quality" -- is
answered by ranking on tokens*active (useful compute, the quality driver) subject to FITTING at
full-throughput b16 and FEEDING experts >= spr1's 0.675B tok/expert.

Run: uv run python decision_table.py
"""
from heuristic import MoeAdamHHeuristic
from design_scan import params  # reuse the validated param counter

h = MoeAdamHHeuristic()
D_E = 512

# 24h budget, holding back HOURS_SFT for the SFT cooldown.
HOURS_TOTAL = 24.0
HOURS_SFT = 4.0
PRE_SECS = (HOURS_TOTAL - HOURS_SFT) * 3600

# spr1 production baseline (the quality comparison point): old slow CE, 15.2h.
SPR1_TOKENS, SPR1_TOTAL, SPR1_ACTIVE = 5.36e9, 3.97e9, 0.807e9
SPR1_TXT, SPR1_TXA = SPR1_TOKENS * SPR1_TOTAL, SPR1_TOKENS * SPR1_ACTIVE
SPR1_TPE = SPR1_TOKENS * 8 / 64  # 0.67B healthy tok/expert

# MEASURED smoke results: (name, D, E, K, I_mult, batch, tok/s, mfu%, fit_verdict)
# fit: CLEAN=ran clean to step>=58 at b16; MARGINAL=ran then late-OOM; OOM=died; B8=only fits at b8.
M = [
    ("spr1 (run#1 geom)", 1536, 64,  8,  1, 16, 192_000, 15.5, "CLEAN(anchor)"),
    ("E128/K8",           1536, 128, 8,  1, 16, 187_000, 15.1, "CLEAN"),
    ("E64/I1536 chonky",  1536, 64,  8,  2, 16, 148_000, 17.0, "MARGINAL(OOM@40)"),
    ("E256/K8 @b8",       1536, 256, 8,  1,  8, 145_000, 11.7, "B8 only"),
    ("E192/K8",           1536, 192, 8,  1, 16, None,    None, "OOM"),
    ("E128/I1024",        1536, 128, 8,  1, 16, None,    None, "OOM"),
    ("E256/K8 @b16",      1536, 256, 8,  1, 16, None,    None, "OOM"),
    ("E256/K16 @b16",     1536, 256, 16, 1, 16, None,    None, "OOM"),
    ("D2048/E64",         2048, 64,  8,  1, 16, None,    None, "OOM"),
    ("D2048/E128",        2048, 128, 8,  1, 16, None,    None, "OOM"),
    # round 3 ceiling rows get appended below once measured.
]
# Round 3 (thin-expert b16 ceiling) -- MEASURED. NONDETERMINISTIC near the wall: E144 & E176 ran
# clean but E160 (between them) OOM'd at startup -> the E144-E176 zone is a fragmentation-dependent
# edge, NOT a safe production ceiling. E128 is the reliable full-throughput pick.
M += [
    ("E144/K8 (edge)",    1536, 144, 8,  1, 16, 185_000, 14.9, "CLEAN but near edge"),
    ("E160/K8",           1536, 160, 8,  1, 16, None,    None, "OOM (E176 fit!)"),
    ("E176/K8 (edge)",    1536, 176, 8,  1, 16, 181_000, 14.6, "CLEAN but E160 OOM'd"),
]


def row(name, D, E, K, Imult, b, tps, mfu, fit):
    cfg = h.build_model_config(D, num_experts=E, num_experts_per_token=K, embed_dim=D_E)
    if Imult != 1:
        import dataclasses
        cfg = dataclasses.replace(cfg, intermediate_dim=cfg.intermediate_dim * Imult)
    total, active = params(cfg, K)
    if isinstance(tps, (int, float)) and tps:
        tok = tps * PRE_SECS
        txt, txa, tpe = tok * total, tok * active, tok * K / E
        return dict(name=name, total=total, active=active, b=b, tps=tps, mfu=mfu, fit=fit,
                    tok=tok, txt=txt, txa=txa, tpe=tpe)
    return dict(name=name, total=total, active=active, b=b, tps=tps, mfu=mfu, fit=fit,
                tok=None, txt=None, txa=None, tpe=None)


rows = [row(*m) for m in M]
print("=" * 132)
print(f"NEXT-RUN decision table (MEASURED, new CE, 8xH100)   pretrain budget = {HOURS_TOTAL-HOURS_SFT:.0f}h "
      f"(holding {HOURS_SFT:.0f}h for SFT)   d_e={D_E} seq=4096")
print(f"  spr1 baseline: {SPR1_TOTAL/1e9:.2f}B/{SPR1_ACTIVE/1e9:.2f}B, tokens*total={SPR1_TXT:.2e}, "
      f"tokens*active={SPR1_TXA:.2e}, tok/expert={SPR1_TPE/1e9:.2f}B (healthy)")
print("=" * 132)
print(f"{'config':<20}{'tot(B)':>8}{'act(B)':>8}{'b':>4}{'tok/s':>9}{'MFU%':>6}{'tok(B)':>9}"
      f"{'xTOTAL':>9}{'xACTIVE':>9}{'t/exp(B)':>9}  {'fit':<16}")
print("-" * 132)
for r in rows:
    if r["tok"]:
        fed = "" if r["tpe"] >= SPR1_TPE else " STARVE"
        print(f"{r['name']:<20}{r['total']/1e9:>8.2f}{r['active']/1e9:>8.2f}{r['b']:>4}"
              f"{r['tps']/1e3:>8.0f}K{r['mfu']:>6.1f}{r['tok']/1e9:>9.1f}"
              f"{r['txt']/SPR1_TXT:>8.2f}x{r['txa']/SPR1_TXA:>8.2f}x{r['tpe']/1e9:>9.2f}  {r['fit']:<12}{fed}")
    else:
        print(f"{r['name']:<20}{r['total']/1e9:>8.2f}{r['active']/1e9:>8.2f}{r['b']:>4}"
              f"{str(r['tps']):>9}{str(r['mfu']):>6}{'-':>9}{'-':>9}{'-':>9}{'-':>9}  {r['fit']:<16}")
print("-" * 132)
print("xTOTAL/xACTIVE = multiple of spr1's tokens*total / tokens*active. STARVE = tok/expert < spr1's 0.67B.")
print("KEY: full-throughput b16 caps total params ~7.6-? B (MoE fwd ring-dispatch transient, unfreeable);")
print("     bigger models only run at b8 = fewer tokens AND lower MFU AND starved experts.")
