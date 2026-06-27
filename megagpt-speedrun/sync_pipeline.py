# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""SYNC (gradient-exact) microbatched pipeline parallelism for megagpt MoE (8xH100).

This is the CORRECT pipeline that the earlier async_pipeline.py got wrong. The
overlap probe (pp_overlap_probe.py, PROBE4/PROBE5) proved that eager per-stage
``jit`` dispatch + cross-device ``device_put`` + ``set_mesh`` OVERLAP the 8 GPUs at
~100% efficiency WHEN driven microbatch-major. async_pipeline rippled ONE batch with
a per-tick per-stage weight update -- a serial dependent chain (PROBE3 = 7.85x, i.e.
no overlap). Here we split the global batch into M microbatches and stream them
through the P stages, GPipe-style, with a SINGLE synchronous optimizer step per stage
at the end -- gradient-exact vs the non-pipelined oracle, and overlapping.

Schedule (GPipe):
  * FORWARD sweep, microbatch-major: for each microbatch, ripple the activation
    stage0->...->stage P-1 (a per-microbatch chain), dispatched back-to-back with NO
    host sync so microbatch m+1's stage 0 overlaps microbatch m's stage 1, etc.
  * BACKWARD sweep, microbatch-major: same, cotangents flowing P-1->0, each stage's
    weight-grad accumulated into a per-stage running sum.
  * OPTIMIZER: one Muon+AdamW update per stage from the accumulated grad.

Gradient-exactness: total loss = (1/M) sum_m [ CE_m + z_coef * sum_s Z_{m,s} ]. The
last stage's CE backward is seeded with 1/M and every stage's router-z output with
z_coef/M, so the per-stage accumulated grads equal d(total_loss)/d(params) computed
non-pipelined over the whole batch (to float reassociation tolerance).

Stages own disjoint single-GPU submeshes (EP=1 experts-local, FSDP=1 params-resident);
the EP all-to-all (29% of EP+FSDP step) and FSDP all-gather (15%) are eliminated,
replaced by cheap NVLink P2P activation hops at the 7 stage boundaries.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from async_pipeline import (
    _StageFns,
    _build_stage_masks,
    _make_stage_mesh,
    _put_on,
    _split_transformer,
    _transport,
    orthogonalize_tree,
    set_mesh,
)
from model import Transformer, _batch_spec

_REPL = P()

# Diagnostic: when _SERIALIZE[0] is True, the pipeline blocks on every stage op so the 8
# GPUs run strictly serially -- the reference for the overlap-factor sanity check
# (serial_step_ms / overlapped_step_ms). Off by default (it would kill cross-stage overlap).
_SERIALIZE = [False]


def _maybe_block(x):
    if _SERIALIZE[0]:
        jax.block_until_ready(x)
    return x


def _put_batch(arr: np.ndarray, mesh):
    return jax.device_put(arr, NamedSharding(mesh, _batch_spec()))


def _to_repl(x, mesh):
    """Place a (scalar) array replicated on ``mesh`` so cross-stage scalars can be summed."""
    return jax.device_put(x, NamedSharding(mesh, _REPL))


def _accum(prev, g):
    return g if prev is None else jax.tree_util.tree_map(jnp.add, prev, g)


def build_sync_pipeline(
    transformer: Transformer,
    *,
    num_stages: int = 8,
    num_microbatches: int = 8,
    lr: float = 3e-4,
    muon: bool = True,
    remat: bool = True,
):
    """Build the sync microbatched GPipe pipeline.

    Returns ``(stage_arrays, stage_opt_st, step, submeshes, grads_fn)`` where
    ``step(stage_arrays, stage_opt_st, token_ids_np, lw_np) -> (loss_dev, stage_arrays, stage_opt_st)``
    runs one pipelined forward+backward over the global batch and applies one optimizer
    step per stage, and ``grads_fn(stage_arrays, token_ids_np, lw_np) -> (g_per_stage, loss_dev)``
    returns the raw accumulated per-stage grads (for the gradient-exactness check).
    """
    cfg = transformer.config
    devices = jax.devices()
    if len(devices) < num_stages:
        raise ValueError(f"need >={num_stages} devices, got {len(devices)}")
    Pn = num_stages
    last = Pn - 1

    submeshes = [_make_stage_mesh(devices[s]) for s in range(Pn)]
    stage_arrays_host, stage_statics = _split_transformer(transformer, Pn)
    stage_masks = _build_stage_masks(cfg, Pn)
    fns = [_StageFns(s, Pn, stage_statics[s], cfg, stage_masks[s], remat) for s in range(Pn)]

    # Place params per stage; free the device-0 source copy (see async_pipeline note).
    stage_arrays = []
    for s in range(Pn):
        host_s = jax.tree_util.tree_map(
            lambda x: np.asarray(x) if eqx.is_array(x) else x, stage_arrays_host[s]
        )
        for leaf in jax.tree_util.tree_leaves(stage_arrays_host[s]):
            try:
                leaf.delete()
            except Exception:
                pass
        stage_arrays.append(_put_on(host_s, submeshes[s]))
        del host_s
    del stage_arrays_host

    per_opt = [optax.adamw(learning_rate=lr, b1=0.9, b2=0.95) for _ in range(Pn)]
    stage_opt_st = [per_opt[s].init(stage_arrays[s]) for s in range(Pn)]

    z_coef = cfg.router_z_loss_coef / cfg.num_layers
    inv_m = 1.0 / num_microbatches
    dloss_m = jnp.asarray(inv_m, jnp.float32)           # CE backward seed (mean over M)
    dz = jnp.asarray(inv_m * z_coef, jnp.float32)       # router-z backward seed per stage
    muon_fn = jax.jit(orthogonalize_tree) if muon else None
    mesh0, meshL = submeshes[0], submeshes[last]

    def _apply_opt(s, dparams, stage_arrays, stage_opt_st):
        with set_mesh(submeshes[s]):
            if muon_fn is not None:
                dparams = muon_fn(dparams)
            updates, new_opt = per_opt[s].update(dparams, stage_opt_st[s], stage_arrays[s])
            stage_arrays[s] = optax.apply_updates(stage_arrays[s], updates)
        stage_opt_st[s] = new_opt

    def _grads(stage_arrays, token_ids_np, lw_np):
        """Pipelined forward+backward; returns (g_per_stage, loss_dev). No optimizer step."""
        B, S = token_ids_np.shape
        if B % num_microbatches != 0:
            raise ValueError(f"B={B} must divide num_microbatches={num_microbatches}")
        mb = B // num_microbatches
        labels_np = np.concatenate([token_ids_np[:, 1:], np.zeros((B, 1), np.int32)], axis=1).astype(np.int32)

        saved_x = [[None] * Pn for _ in range(num_microbatches)]   # input to stage s, microbatch m
        lbl_dev = [None] * num_microbatches
        lw_dev = [None] * num_microbatches
        ce = [None] * num_microbatches
        z = [[None] * Pn for _ in range(num_microbatches)]

        # ---- FORWARD sweep (microbatch-major, no host sync -> overlaps across microbatches) ----
        for m in range(num_microbatches):
            sl = slice(m * mb, (m + 1) * mb)
            tok = _put_batch(token_ids_np[sl], mesh0)
            saved_x[m][0] = tok
            with set_mesh(mesh0):
                h, z[m][0] = fns[0].forward(stage_arrays[0], tok)
            _maybe_block(h)
            for s in range(1, Pn):
                h_in = _transport(h, submeshes[s])
                saved_x[m][s] = h_in
                if s < last:
                    with set_mesh(submeshes[s]):
                        h, z[m][s] = fns[s].forward(stage_arrays[s], h_in)
                    _maybe_block(h)
                else:
                    lbl = _put_batch(labels_np[sl], meshL)
                    w = _put_batch(lw_np[sl], meshL)
                    lbl_dev[m], lw_dev[m] = lbl, w
                    with set_mesh(meshL):
                        ce[m], z[m][last] = fns[last].forward(stage_arrays[last], h_in, lbl, w)
                    _maybe_block(ce[m])

        # ---- BACKWARD sweep (microbatch-major) ----
        g = [None] * Pn
        for m in range(num_microbatches):
            with set_mesh(meshL):
                dparams, dx = fns[last].backward(
                    stage_arrays[last], saved_x[m][last], lbl_dev[m], lw_dev[m], dloss_m, dz
                )
            g[last] = _accum(g[last], dparams)
            _maybe_block(dx)
            for s in range(last - 1, -1, -1):
                dx_in = _transport(dx, submeshes[s])
                if s == 0:
                    with set_mesh(mesh0):
                        dparams = fns[0].backward(stage_arrays[0], saved_x[m][0], dx_in, dz)
                    g[0] = _accum(g[0], dparams)
                    _maybe_block(dparams)
                else:
                    with set_mesh(submeshes[s]):
                        dparams, dx = fns[s].backward(stage_arrays[s], saved_x[m][s], dx_in, dz)
                    g[s] = _accum(g[s], dparams)
                    _maybe_block(dx)

        # ---- loss value (ce[m] already on meshL; bring z scalars to meshL and sum) ----
        ce_total = jnp.sum(jnp.stack(list(ce)))
        z_scalars = [_to_repl(z[m][s], meshL) for m in range(num_microbatches) for s in range(Pn)]
        z_total = jnp.sum(jnp.stack(z_scalars))
        loss_dev = (ce_total + z_coef * z_total) * inv_m
        return g, loss_dev

    def step(stage_arrays, stage_opt_st, token_ids_np, lw_np):
        g, loss_dev = _grads(stage_arrays, token_ids_np, lw_np)
        stage_arrays = list(stage_arrays)
        stage_opt_st = list(stage_opt_st)
        for s in range(Pn):
            _apply_opt(s, g[s], stage_arrays, stage_opt_st)
        return loss_dev, stage_arrays, stage_opt_st

    return stage_arrays, stage_opt_st, step, submeshes, _grads
