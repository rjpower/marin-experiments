# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Design-space scan for the NEXT run: maximize tokens x params in a fixed 24h on 8xH100.

Uses the heuristic to derive geometry from (D, E, K, I_mult) and an exact param/FLOP
count (cross-checked against spr1: 3.97B total / 0.80B active). Projects throughput
from the measured new-CE anchor and reports the two products the user cares about
(tokens x total, tokens x active) plus the EXPERT-FEEDING check (the thing that killed
naive E256 in the A/B): tokens-per-expert over the run vs spr1's healthy 0.675B.

Run: uv run python design_scan.py
"""
import math
from heuristic import MoeAdamHHeuristic
from perf_flop_breakdown import fwd_components

h = MoeAdamHHeuristic()
D_E = 512          # factorized CE dim (fixed -- the user said reduce embed dim, keep vocab)
V = 128_256
SEQ = 4096
GN_RANK = 128      # gated-norm rank

# --- hardware / wall-clock ---
PEAK = 8 * 989.5e12      # 8x H100 SXM bf16 dense peak FLOP/s
HOURS = 24.0
SECS = HOURS * 3600

# --- anchor: spr1 thin E64/K8 on the NEW CE measured 192K tok/s @ active 0.81B ---
# achieved matmul-FLOP/s = 6 * active * tok/s ; bigger configs only RAISE MFU, so this
# is a conservative (same-FLOP/s) floor for projecting tok/s of bigger models.
ANCHOR_TOKS = 192_000
ANCHOR_ACTIVE = 0.807e9
ANCHOR_FLOPS = 6 * ANCHOR_ACTIVE * ANCHOR_TOKS    # ~9.3e14 FLOP/s

# --- spr1 PRODUCTION baseline (old slow CE, 15.2h) -- the quality comparison point ---
SPR1_TOKS = 98_000
SPR1_TIME = 15.2 * 3600
SPR1_TOKENS = SPR1_TOKS * SPR1_TIME           # ~5.36e9
SPR1_TOTAL = 3.97e9
SPR1_ACTIVE = 0.807e9
SPR1_TXTOTAL = SPR1_TOKENS * SPR1_TOTAL
SPR1_TXACTIVE = SPR1_TOKENS * SPR1_ACTIVE
SPR1_TOK_PER_EXPERT = SPR1_TOKENS * 8 / 64    # 0.675B -- the healthy-feeding reference


def params(cfg, K):
    """(total, active) params. Validated vs spr1 = 3.97B/0.81B."""
    D, E, I, L = cfg.hidden_dim, cfg.num_experts, cfg.intermediate_dim, cfg.num_layers
    Ish = cfg.shared_expert_intermediate_dim
    H, KV, hd = cfg.num_heads, cfg.num_kv_heads, cfg.inferred_head_dim
    attn = D * (H * hd + 2 * KV * hd) + D * D          # QKV + O
    router = D * E
    shared = 3 * D * Ish                               # GLU shared expert
    norms = 2 * (2 * (2 * D * GN_RANK))                # 2 gated-norms/layer, down+up
    per_layer_fixed = attn + router + shared + norms
    routed_total = E * 3 * D * I
    routed_active = K * 3 * D * I
    embed_head = 2 * V * D_E + 2 * D_E * D             # factorized embed table+head + projs
    total = L * (per_layer_fixed + routed_total) + embed_head
    active = L * (per_layer_fixed + routed_active) + embed_head
    return total, active


def fwd_flops_active(cfg, K):
    """Per-token forward matmul+attn FLOPs at active K (exact, from perf_flop_breakdown)."""
    return fwd_components(cfg, K)["_total"]


def scan_row(name, D, E, K, I_mult=1):
    cfg = h.build_model_config(D, num_experts=E, num_experts_per_token=K, embed_dim=D_E)
    if I_mult != 1:
        import dataclasses
        cfg = dataclasses.replace(cfg, intermediate_dim=cfg.intermediate_dim * I_mult)
    total, active = params(cfg, K)
    fwd = fwd_flops_active(cfg, K)
    # tok/s floor: assume same achieved matmul-FLOP/s as the anchor (bigger -> MFU rises -> faster).
    # FLOP/token (fwd+bwd) ~= 6*active; ANCHOR_FLOPS = 6*active_anchor*tok_anchor.
    toks_per_s = ANCHOR_FLOPS / (6 * active)
    tokens24 = toks_per_s * SECS
    txtotal = tokens24 * total
    txactive = tokens24 * active
    tok_per_expert = tokens24 * K / E
    hbm_gb = total * 12 / 8 / 1e9                       # AdamH ~12 B/param, sharded /8
    return {
        "name": name, "D": D, "E": E, "K": K, "I": cfg.intermediate_dim, "L": cfg.num_layers,
        "total": total, "active": active, "afrac": active / total,
        "KE": K / E, "toks": toks_per_s, "tokens24": tokens24,
        "txtotal": txtotal, "txactive": txactive,
        "tok_per_expert": tok_per_expert, "hbm_gb": hbm_gb,
    }


CANDS = [
    ("spr1-base",   1536, 64,  8,  1),
    # --- more experts (free total capacity); same K ---
    ("E128/K8",     1536, 128, 8,  1),
    ("E256/K8",     1536, 256, 8,  1),
    # --- "more experts with LESS sparsity" (user hint): raise K with E ---
    ("E128/K16",    1536, 128, 16, 1),
    ("E256/K16",    1536, 256, 16, 1),
    ("E256/K32",    1536, 256, 32, 1),
    # --- bigger ACTIVE via chonky experts ---
    ("E64/I1536",   1536, 64,  8,  2),
    ("E64/I3072",   1536, 64,  8,  4),
    ("E128/I1536",  1536, 128, 8,  2),
    ("E128k16/I1536",1536,128, 16, 2),
    # --- bigger D (more active + auto-deeper via heuristic) ---
    ("D2048/E64",   2048, 64,  8,  1),
    ("D2048/E128",  2048, 128, 8,  1),
    ("D2048/E128k16",2048,128, 16, 1),
    ("D2560/E64",   2560, 64,  8,  1),
    ("D2560/E128",  2560, 128, 8,  1),
]

rows = [scan_row(*c) for c in CANDS]

print("=" * 140)
print(f"NEXT-RUN design scan  (24h, 8xH100, d_e={D_E}, seq={SEQ})   "
      f"baseline spr1: {SPR1_TOTAL/1e9:.2f}B tot / {SPR1_ACTIVE/1e9:.2f}B act, "
      f"tokens24~{SPR1_TOKENS/1e9:.1f}B")
print(f"  spr1 products: tokens*total={SPR1_TXTOTAL:.2e}  tokens*active={SPR1_TXACTIVE:.2e}  "
      f"tok/expert={SPR1_TOK_PER_EXPERT/1e9:.3f}B (HEALTHY ref)")
print("=" * 140)
hdr = (f"{'name':<16}{'D':>5}{'E':>5}{'K':>4}{'I':>6}{'L':>4}"
       f"{'tot(B)':>8}{'act(B)':>8}{'afrac':>7}{'K/E':>7}"
       f"{'tok/s':>9}{'tok24(B)':>10}{'tok*tot':>10}{'tok*act':>10}{'t/exp(B)':>9}{'HBM/dev':>9}")
print(hdr)
print("-" * 140)
for r in sorted(rows, key=lambda x: -x["txtotal"]):
    fed = "" if r["tok_per_expert"] >= SPR1_TOK_PER_EXPERT else "  <-STARVE"
    print(f"{r['name']:<16}{r['D']:>5}{r['E']:>5}{r['K']:>4}{r['I']:>6}{r['L']:>4}"
          f"{r['total']/1e9:>8.2f}{r['active']/1e9:>8.2f}{r['afrac']*100:>6.1f}%{r['KE']*100:>6.1f}%"
          f"{r['toks']/1e3:>8.0f}K{r['tokens24']/1e9:>10.1f}"
          f"{r['txtotal']/SPR1_TXTOTAL:>9.2f}x{r['txactive']/SPR1_TXACTIVE:>9.2f}x"
          f"{r['tok_per_expert']/1e9:>9.3f}{r['hbm_gb']:>8.1f}G{fed}")
print("-" * 140)
print("NOTES: tok/s is the SAME-FLOP/s floor (bigger experts RAISE MFU -> real tok/s higher);")
print("       tok*tot / tok*act are MULTIPLES of spr1's products; <-STARVE = each expert sees")
print("       fewer tokens than spr1's healthy 0.675B (the failure mode that killed naive E256).")
print("       HBM/dev is static (params+AdamH /8); ~40G headroom remains for the activation transient.")
