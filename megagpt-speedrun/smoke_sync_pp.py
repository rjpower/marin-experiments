# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""CPU gradient-exactness gate for the SYNC microbatched pipeline (sync_pipeline.py).

Two checks, on a tiny model over simulated CPU devices:

  1. PIPELINE vs ORACLE (tight): the pipeline's per-stage accumulated grads must equal
     a non-pipelined autodiff (`jax.grad`) of the IDENTICAL composed-stage loss on one
     device. Validates the backward seeds, the cotangent chaining across stage
     boundaries, the cross-device transport, and the accumulation.

  2. M-INVARIANCE (looser, bf16 reassociation): grads with M=1 vs M=4 microbatches must
     agree. Validates the 1/M loss/grad scaling -- the thing a microbatched sync pipeline
     most easily gets wrong.

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \\
      uv run python smoke_sync_pp.py
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("RAGGED_DOT_IMPL", "xla")  # CPU: jax.lax.ragged_dot_general (no Triton/Pallas)

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh
from haliax.partitioning import set_mesh

from model import GrugModelConfig, Transformer
from async_pipeline import _make_stage_mesh, _put_on
from sync_pipeline import build_sync_pipeline


def _tiny_cfg() -> GrugModelConfig:
    return GrugModelConfig(
        vocab_size=64,
        hidden_dim=32,
        embed_dim=16,          # exercise the factorized embed_up/head_down path (d_e<D)
        intermediate_dim=64,
        shared_expert_intermediate_dim=0,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=4,
        num_heads=4,
        num_kv_heads=4,
        max_seq_len=16,
        sliding_window=16,
        global_attn_every=2,
        local_window=8,
        router_z_loss_coef=0.001,
        attention_implementation=None,   # reference attention (CPU-friendly)
        moe_implementation=None,
    )


def _reldiff(a, b) -> float:
    """Relative L2 difference between two matching pytrees (max over leaves)."""
    la = jax.tree_util.tree_leaves(a)
    lb = jax.tree_util.tree_leaves(b)
    worst = 0.0
    for x, y in zip(la, lb):
        x = np.asarray(jax.device_get(x), np.float64)
        y = np.asarray(jax.device_get(y), np.float64)
        denom = np.linalg.norm(x) + 1e-12
        worst = max(worst, float(np.linalg.norm(x - y) / denom))
    return worst


def main() -> int:
    P = 4
    cfg = _tiny_cfg()
    devices = jax.devices()
    print(f"[SYNC_SMOKE] {len(devices)} {devices[0].platform} devices; P={P} model={cfg.num_layers}L "
          f"D={cfg.hidden_dim} d_e={cfg.embed_dim} E={cfg.num_experts}/K{cfg.num_experts_per_token}", flush=True)
    if len(devices) < P:
        raise SystemExit(f"need >={P} devices (set XLA_FLAGS device count)")

    dev0 = devices[0]
    init_mesh = Mesh(np.array([[[[dev0]]]]), ("replica_dcn", "data", "expert", "model"),
                     axis_types=(AxisType.Explicit,) * 4)
    with set_mesh(init_mesh):
        model = Transformer.init(cfg, key=jax.random.PRNGKey(0))

    B, S = 8, cfg.max_seq_len
    rng = np.random.default_rng(0)
    tok = rng.integers(1, cfg.vocab_size, (B, S)).astype(np.int32)
    lw = np.ones((B, S), np.float32)

    # Build the pipeline once (M=4). NOTE: build deletes the transformer's array leaves
    # (placed per-stage), so the oracle params are pulled back from the placed arrays.
    sa4, _opt4, _step4, submeshes, grads4 = build_sync_pipeline(
        model, num_stages=P, num_microbatches=4, muon=False, remat=False
    )
    g_pipe4, loss4 = grads4(sa4, tok, lw)
    jax.block_until_ready((g_pipe4, loss4))

    # ----- ORACLE: non-pipelined autodiff of the identical composed-stage loss on dev0 -----
    mesh0 = submeshes[0]
    # Reuse the SAME compiled stage fns + masks/static from a fresh splitter so the math is
    # identical; pull current (placed) params back to dev0 to differentiate there.
    from async_pipeline import _StageFns, _build_stage_masks, _split_transformer  # noqa
    # Rebuild stage fns bound to mesh-agnostic jit (same as pipeline's fns).
    # We re-derive statics from a throwaway init (structure only) since model arrays were deleted.
    with set_mesh(init_mesh):
        model_struct = Transformer.init(cfg, key=jax.random.PRNGKey(1))
    _sa_host, stage_statics = _split_transformer(model_struct, P)
    stage_masks = _build_stage_masks(cfg, P)
    fns = [_StageFns(s, P, stage_statics[s], cfg, stage_masks[s], remat=False) for s in range(P)]

    params0 = [_put_on(jax.device_get(sa4[s]), mesh0) for s in range(P)]
    z_coef = cfg.router_z_loss_coef / cfg.num_layers

    def oracle_total(params, M):
        mb = B // M
        labels = np.concatenate([tok[:, 1:], np.zeros((B, 1), np.int32)], axis=1).astype(np.int32)
        from sync_pipeline import _put_batch
        t = 0.0
        for m in range(M):
            sl = slice(m * mb, (m + 1) * mb)
            h, z = fns[0].forward(params[0], _put_batch(tok[sl], mesh0))
            zsum = z
            for s in range(1, P - 1):
                h, z = fns[s].forward(params[s], h)
                zsum = zsum + z
            ce, z = fns[P - 1].forward(
                params[P - 1], h, _put_batch(labels[sl], mesh0), _put_batch(lw[sl], mesh0)
            )
            zsum = zsum + z
            t = t + ce + z_coef * zsum
        return t / M

    with set_mesh(mesh0):  # mesh context must be OUTSIDE jax.grad tracing
        g_oracle = jax.grad(lambda p: oracle_total(p, 4))(params0)
        oracle_loss = float(oracle_total(params0, 4))

    # Compare pipeline grads (per stage) to oracle grads (per stage).
    rel_po = max(_reldiff(g_pipe4[s], g_oracle[s]) for s in range(P))
    print(f"[SYNC_SMOKE] loss: pipeline={float(loss4):.6f} oracle={oracle_loss:.6f} "
          f"reldiff={abs(float(loss4)-oracle_loss)/(abs(oracle_loss)+1e-9):.2e}", flush=True)
    print(f"[SYNC_SMOKE] CHECK1 pipeline-vs-oracle grads: worst per-stage relL2={rel_po:.2e} "
          f"({'PASS' if rel_po < 2e-2 else 'FAIL'})", flush=True)

    # ----- M-INVARIANCE: M=1 grads vs M=4 grads (sum over batch must be M-independent) -----
    sa1, _o1, _s1, _sm1, grads1 = build_sync_pipeline(
        model_struct, num_stages=P, num_microbatches=1, muon=False, remat=False
    )
    # place identical param VALUES as the M=4 pipeline so grads are comparable
    sa1 = [_put_on(jax.device_get(sa4[s]), submeshes[s]) for s in range(P)]
    g_pipe1, loss1 = grads1(sa1, tok, lw)
    jax.block_until_ready((g_pipe1, loss1))
    rel_m = max(_reldiff(g_pipe1[s], g_pipe4[s]) for s in range(P))
    print(f"[SYNC_SMOKE] loss M=1 vs M=4: {float(loss1):.6f} vs {float(loss4):.6f}", flush=True)
    print(f"[SYNC_SMOKE] CHECK2 M-invariance (M=1 vs M=4) grads: worst relL2={rel_m:.2e} "
          f"({'PASS' if rel_m < 3e-2 else 'FAIL'})", flush=True)

    ok = rel_po < 2e-2 and rel_m < 3e-2
    print(f"[SYNC_SMOKE] {'PASS' if ok else 'FAIL'}: gradient-exactness gate", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
