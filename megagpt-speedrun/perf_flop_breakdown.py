# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Analytic per-token FLOP breakdown for the strong-sparsity grug MoE.

Mirrors levanter.utils.flop_utils.lm_flops_per_token component-by-component so we
can attribute per-token FLOPs to: attention (QKVO projections + scores/AV),
router projection (D x E), routed experts (K * 3 * 2 * D * I_expert), shared
expert (3 * 2 * D * I_shared), embeddings/lm_head, and the gated-norms (which the
levanter util OMITS -- we add them explicitly to size that overhead).

Run: uv run python perf_flop_breakdown.py
"""

from heuristic import MoeAdamHHeuristic

# Production strong-sparsity geometry (see launch.py defaults / experiment brief).
HIDDEN = 512
NUM_EXPERTS = 1024
SEQ_LEN = 4096
VOCAB = 128_256
GATED_NORM_RANK = 128  # _GATED_NORM_RANK in model.py


def fwd_components(cfg, k):
    """Forward per-token FLOPs by component (2*MACs). Multiply by 3 for fwd+bwd."""
    D = cfg.hidden_dim
    I_exp = cfg.intermediate_dim
    I_shared = cfg.shared_expert_intermediate_dim
    E = cfg.num_experts
    L = cfg.num_layers
    H = cfg.num_heads
    KV = cfg.num_kv_heads
    hd = cfg.inferred_head_dim
    S = cfg.max_seq_len
    V = cfg.vocab_size

    # --- attention ---
    qkv_proj = 2 * D * (H * hd + 2 * KV * hd)
    o_proj = 2 * D * D
    # scores + softmax-mask + AV, amortized per token (matches levanter util)
    key_query = 2 * S**2 * H * hd
    mask = 3 * S * S * H
    mask_value = 2 * S * S * hd * H
    attn_scores = (key_query + mask + mask_value) / S
    attn = qkv_proj + o_proj + attn_scores

    # --- router projection D x E (dense, does NOT shrink with K) ---
    router = 2 * D * E

    # --- routed experts: GLU = 3 matmuls (gate, up, down), K active ---
    routed = 2 * 3 * D * I_exp * k

    # --- shared (always-on dense) expert: GLU = 3 matmuls ---
    shared = 2 * 3 * D * I_shared

    # --- gated-norms: levanter util omits these. model.py applies a GatedNorm
    # (down D->r, up r->D) at: attn input, mlp input, embed, final = 2 per block
    # + 2 global. Each GatedNorm = 2 matmuls of D*r.
    gated_norm_per = 2 * (2 * D * GATED_NORM_RANK)  # down + up
    gated_norms_per_layer = 2 * gated_norm_per  # attn_gated_norm + mlp_gated_norm

    attn_proj = qkv_proj + o_proj  # QKVO projections (∝ D², independent of seq_len)
    attn_score = attn_scores  # QK^T + softmax + AV (∝ seq_len)
    per_layer = attn + router + routed + shared + gated_norms_per_layer

    # --- embeddings / lm_head (once, not per layer) ---
    lm_head = 2 * D * V
    embed_gated_norms = 2 * gated_norm_per  # embed_gated_norm + final_gated_norm

    total = L * per_layer + lm_head + embed_gated_norms

    return {
        "attention": L * attn,
        "attn_proj_QKVO": L * attn_proj,
        "attn_scores_AV": L * attn_score,
        "router_proj": L * router,
        "routed_experts": L * routed,
        "shared_expert": L * shared,
        "gated_norms": L * gated_norms_per_layer + embed_gated_norms,
        "lm_head": lm_head,
        "_total": total,
    }


def main():
    h = MoeAdamHHeuristic()
    cfg = h.build_model_config(HIDDEN, num_experts=NUM_EXPERTS, num_experts_per_token=1)
    D, E, I_exp, I_shared, L = (
        cfg.hidden_dim,
        cfg.num_experts,
        cfg.intermediate_dim,
        cfg.shared_expert_intermediate_dim,
        cfg.num_layers,
    )
    print("=" * 78)
    print("Strong-sparsity grug MoE geometry")
    print("=" * 78)
    print(
        f"  hidden D={D}  experts E={E}  I_expert={I_exp}  I_shared={I_shared}\n"
        f"  num_layers={L}  num_heads={cfg.num_heads}  num_kv_heads={cfg.num_kv_heads}  "
        f"head_dim={cfg.inferred_head_dim}\n"
        f"  seq_len={cfg.max_seq_len}  vocab={cfg.vocab_size}"
    )
    print()

    Ks = [1, 2, 4, 8, 16]
    comp_order = ["attention", "router_proj", "routed_experts", "shared_expert", "gated_norms", "lm_head"]
    sub_order = ["attn_proj_QKVO", "attn_scores_AV"]

    # Per-K forward-FLOP breakdown (fwd only; fwd+bwd is 3x, ratios identical).
    rows = {}
    for k in Ks:
        rows[k] = fwd_components(cfg, k)

    # Component table (GFLOPs/token fwd)
    print("Per-token FORWARD FLOPs by component (GFLOP = 1e9), by top-K:")
    hdr = f"  {'component':<16}" + "".join(f"{('K=' + str(k)):>12}" for k in Ks)
    print(hdr)
    print("  " + "-" * (16 + 12 * len(Ks)))
    for c in comp_order:
        line = f"  {c:<16}"
        for k in Ks:
            line += f"{rows[k][c] / 1e9:>12.4f}"
        print(line)
    line = f"  {'TOTAL':<16}"
    for k in Ks:
        line += f"{rows[k]['_total'] / 1e9:>12.4f}"
    print(line)
    print("  (attention split: QKVO-proj vs scores/AV)")
    for c in sub_order:
        line = f"  {'  ' + c:<16}"
        for k in Ks:
            line += f"{rows[k][c] / 1e9:>12.4f}"
        print(line)
    print()

    # Fraction table
    print("Per-token FLOP FRACTION of total, by top-K:")
    print(hdr)
    print("  " + "-" * (16 + 12 * len(Ks)))
    for c in comp_order:
        line = f"  {c:<16}"
        for k in Ks:
            line += f"{100 * rows[k][c] / rows[k]['_total']:>11.2f}%"
        print(line)
    print()

    # Routed-expert fraction headline
    print("ROUTED-EXPERT fraction of total per-token FLOPs:")
    for k in Ks:
        print(f"  K={k:<3}: {100 * rows[k]['routed_experts'] / rows[k]['_total']:6.2f}%")
    print()

    # Speedup ceiling
    t1 = rows[1]["_total"]
    t16 = rows[16]["_total"]
    print("=" * 78)
    print("SPEEDUP CEILING (total per-token FLOPs; assumes perfect HW utilization)")
    print("=" * 78)
    print(f"  K=1 total : {t1 / 1e9:.4f} GFLOP/tok")
    print(f"  K=16 total: {t16 / 1e9:.4f} GFLOP/tok")
    print(f"  K=1 vs K=16 max total-FLOP speedup: {t16 / t1:.3f}x")

    # Hypothetical dense K=E=1024
    dense = fwd_components(cfg, E)
    tdense = dense["_total"]
    print(f"  dense K=E={E} total: {tdense / 1e9:.4f} GFLOP/tok")
    print(f"  K=1 vs dense(K={E}) max total-FLOP speedup: {tdense / t1:.3f}x")
    print()
    print("  Naive hope was ~1000x ('0.1% activation'). Amdahl ceiling at this geometry:")
    print(f"    routed-expert FLOPs alone shrink {E}x going K=E->K=1,")
    print(f"    but the FIXED cost (attn + router_proj + shared + norms + lm_head) caps")
    print(f"    the realized total-FLOP win at {tdense / t1:.1f}x (dense->K=1).")
    print()
    # Decompose the fixed floor at K=1
    fixed = t1 - rows[1]["routed_experts"]
    print(f"  At K=1 the NON-routed 'fixed floor' is {fixed / 1e9:.4f} GFLOP/tok "
          f"({100 * fixed / t1:.1f}% of K=1 total):")
    for c in comp_order:
        if c == "routed_experts":
            continue
        print(f"      {c:<16}{100 * rows[1][c] / t1:6.2f}% of K=1 total")
    print()
    # Confirm against the levanter util used by the heuristic (no lm_head).
    from heuristic import compute_flops_per_token

    cfg_k1 = h.build_model_config(HIDDEN, num_experts=NUM_EXPERTS, num_experts_per_token=1)
    util_no_lmhead = compute_flops_per_token(cfg_k1)
    ours_no_lmhead = t1 - rows[1]["lm_head"] - rows[1]["gated_norms"]
    print("Cross-check vs levanter compute_flops_per_token (excludes lm_head AND gated_norms):")
    print(f"  levanter util (no lm_head)         : {util_no_lmhead / 1e9:.4f} GFLOP/tok")
    print(f"  ours (no lm_head, no gated_norms)  : {ours_no_lmhead / 1e9:.4f} GFLOP/tok")
    print(f"  gated_norms (util omits these)     : {rows[1]['gated_norms'] / 1e9:.4f} GFLOP/tok")
    print()
    # MFU reference: v6e-8 bf16 peak.
    V6E_PEAK = 8 * 9.1e14  # 8 chips * ~910 TFLOP/s bf16 per v6e chip
    tokens_per_batch_256 = 256 * SEQ_LEN
    train_flop_per_tok_k1 = 3 * t1  # fwd+bwd
    print(f"v6e-8 bf16 peak ~= {V6E_PEAK / 1e12:.0f} TFLOP/s (8 x ~910 TFLOP/s)")
    print(f"Train FLOP/token (fwd+bwd, 3x) at K=1: {train_flop_per_tok_k1 / 1e9:.4f} GFLOP/tok")
    for tps in (0.55e6, 1.0e6, 2.0e6):
        mfu = train_flop_per_tok_k1 * tps / V6E_PEAK
        print(f"  at {tps / 1e6:.2f}M tok/s  =>  MFU = {100 * mfu:5.2f}%")


if __name__ == "__main__":
    main()
