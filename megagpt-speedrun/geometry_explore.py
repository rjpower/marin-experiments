# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Analytic geometry explorer for megagpt-speedrun.

Given candidate (D, layers, E, K, I_expert, vocab, seq), compute total params,
active params/token, the lm_head FLOP fraction (the thing we shrink by cutting
vocab), train FLOP/token, and a rough 8xH100 HBM estimate. Used to pick the
"largest deeply-sparse MoE that fits 8x80GB" before paying a worker fit-smoke.

Run: uv run python geometry_explore.py
"""

from dataclasses import dataclass

HEAD_DIM = 128
GQA = 4
GATED_NORM_RANK = 128
N_GPU = 8
HBM_PER_GPU = 80e9  # bytes
# AdamH master fp32 (4) + 2 fp32 moments (8) = 12 B/param, sharded across GPUs.
BYTES_PER_PARAM = 12


@dataclass
class Geo:
    name: str
    D: int
    layers: int
    E: int
    K: int
    I_expert: int
    vocab: int = 128256  # kept at llama3 size; we factorize the embed dim, not the vocab
    seq: int = 4096
    I_shared: int | None = None  # default = D
    embed_dim: int | None = None  # factorized CE dim d_e; None => d_e = D (no factorization)

    @property
    def heads(self):
        return self.D // HEAD_DIM

    @property
    def kv_heads(self):
        return max(1, self.heads // GQA)

    @property
    def i_shared(self):
        return self.I_shared if self.I_shared is not None else self.D

    @property
    def d_e(self):
        return self.embed_dim if self.embed_dim is not None else self.D


def attn_params(g: Geo) -> int:
    D, H, KV, hd = g.D, g.heads, g.kv_heads, HEAD_DIM
    wq = D * (H * hd)
    wk = D * (KV * hd)
    wv = D * (KV * hd)
    wo = (H * hd) * D
    gate = D * H
    return wq + wk + wv + wo + gate


def expert_params_per_layer(g: Geo) -> int:
    # GLU expert: w_gate_up [D, 2I] + w_down [I, D] = 3*D*I per expert.
    return g.E * 3 * g.D * g.I_expert


def shared_params_per_layer(g: Geo) -> int:
    return 3 * g.D * g.i_shared


def gated_norm_params_per_layer(g: Geo) -> int:
    # 2 GatedNorms per block (attn + mlp), each = down + up = 2*D*rank.
    return 2 * (2 * g.D * GATED_NORM_RANK)


def router_params_per_layer(g: Geo) -> int:
    return g.D * g.E


def total_params(g: Geo) -> dict:
    # Factorized embedding: table + head at d_e, plus tiny up/down projections to D.
    embed = g.vocab * g.d_e
    head = g.d_e * g.vocab
    proj = (g.d_e * g.D) * 2 if g.embed_dim is not None else 0
    per_layer = (
        attn_params(g)
        + router_params_per_layer(g)
        + expert_params_per_layer(g)
        + shared_params_per_layer(g)
        + gated_norm_params_per_layer(g)
        + 2 * g.D  # two RMSNorms
    )
    body = g.layers * per_layer
    total = embed + head + proj + body
    return {"embed": embed, "head": head, "proj": proj, "body": body, "total": total}


def _head_active_params(g: Geo) -> int:
    """Per-token params in the output path: LM head (d_e*V) + factorized down-proj (D*d_e)."""
    proj = g.D * g.d_e if g.embed_dim is not None else 0
    return g.d_e * g.vocab + proj


def active_params(g: Geo) -> int:
    # Params actually used per token (drives FLOP): attn + router + K experts +
    # shared + norms, plus the (factorized) output head. Embedding lookup ~0; the
    # embed up-projection (d_e*D) is added to the body via the head accounting symmetry.
    per_layer = (
        attn_params(g)
        + router_params_per_layer(g)
        + g.K * 3 * g.D * g.I_expert
        + shared_params_per_layer(g)
        + gated_norm_params_per_layer(g)
    )
    return g.layers * per_layer + _head_active_params(g)


def fwd_flops_per_token(g: Geo) -> dict:
    # 2 * active matmul params + attention scores (∝ seq).
    D, H, hd, S = g.D, g.heads, HEAD_DIM, g.seq
    attn_score = (2 * S * H * hd) + (2 * S * hd * H)  # QK^T + AV, amortized /S already folded
    head = _head_active_params(g)
    body_mm = active_params(g) - head  # body active params
    flops = {
        "body_mm": 2 * body_mm,
        "attn_scores": g.layers * attn_score,
        "lm_head": 2 * head,
    }
    flops["total"] = sum(flops.values())
    return flops


def report(geos: list[Geo]):
    print(f"{'name':<16}{'D':>5}{'L':>4}{'E':>6}{'K':>4}{'Ie':>6}{'tot(B)':>9}{'act(B)':>9}"
          f"{'head%FLOP':>10}{'GF/tok':>8}{'HBM/GPU':>9}")
    print("-" * 98)
    for g in geos:
        tp = total_params(g)
        ap = active_params(g)
        fl = fwd_flops_per_token(g)
        head_frac = 100 * fl["lm_head"] / fl["total"]
        gf = fl["total"] / 1e9  # fwd GFLOP/token
        # HBM: params+opt sharded across 8 GPUs. (activations extra, remat keeps modest.)
        hbm_per_gpu = tp["total"] * BYTES_PER_PARAM / N_GPU
        print(f"{g.name:<16}{g.D:>5}{g.layers:>4}{g.E:>6}{g.K:>4}{g.I_expert:>6}"
              f"{tp['total']/1e9:>9.2f}{ap/1e9:>9.3f}{head_frac:>9.1f}%{gf:>8.2f}"
              f"{hbm_per_gpu/1e9:>7.1f}G")


if __name__ == "__main__":
    print("=== Candidate geometries (vocab 128256, seq 4096) ===")
    print("Factorized embedding (d_e<D) cuts the head matmul from 2*D*V to 2*d_e*V + 2*D*d_e,")
    print("moving wall-clock FLOP out of the head and into the (sparse) expert layers.")
    print("tot=total params, act=active params/token, head%FLOP=output-head share of fwd FLOP,")
    print("GF/tok=fwd GFLOP/token, HBM/GPU=params+AdamH(12B/param)/8 GPUs (activations extra)\n")
    cands = [
        # name, D, layers, E, K, I_expert  [+ embed_dim]
        Geo("ref-thin-llama", 512, 6, 1024, 4, 256),  # the prior REPORT geometry (no factorize)
        Geo("A-deepseek", 1536, 16, 128, 8, 512),
        Geo("A-deepseek-de512", 1536, 16, 128, 8, 512, embed_dim=512),
        Geo("D-bigE", 1536, 16, 256, 6, 384),
        Geo("D-bigE-de512", 1536, 16, 256, 6, 384, embed_dim=512),
        Geo("C-fatD-de768", 2048, 18, 64, 6, 768, embed_dim=768),
        Geo("E-deep-de512", 1024, 24, 256, 6, 384, embed_dim=512),
        Geo("F-bigE2-de512", 1536, 16, 512, 8, 384, embed_dim=512),
    ]
    report(cands)
    print("\n(HBM headroom: 80G/GPU; leave ~30-40G/GPU for activations+grads under remat.)")
