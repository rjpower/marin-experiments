# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Async no-flush pipeline parallelism for megagpt MoE (8×H100, 8 stages).

Architecture
============
P pipeline stages, 1 H100 per stage. Each stage owns ``L/P`` transformer layers
(16 layers → 2 layers per stage at P=8). All 128 experts are **local** to each
stage (EP=1 per stage), and parameters are **resident** per stage (no FSDP).
The EP all-to-all dispatch (29% of EP+FSDP step time) and the FSDP param
all-gather (15%) are completely eliminated; replaced by cheap NVLink P2P
activation transfers at the 7 stage boundaries (~10% of step time per model).

Schedule (steady-state async)
==============================
Each "tick" advances ONE original batch through the full pipeline::

    --- WARMUP (P-1 ticks, pipeline filling, no backward/optimizer) ---
    tick 0:  S0 fwd batch[0] → inter[0]
    tick 1:  S0 fwd batch[1]; S1 fwd inter[0] → inter[1]
    ...
    tick P-2: S0..S_{P-2} fwd their respective batches

    --- STEADY STATE (every tick from tick P-1 onwards) ---
    tick T:
      1. FORWARD SWEEP (stages 0..P-1, can be dispatched concurrently):
            S0   fwd token_ids[T]         → inter_acts[0]
            S1   fwd inter_acts_prev[0]   → inter_acts[1]
            ...
            S_{P-2} fwd inter_acts_prev[P-3] → inter_acts[P-2]
            S_{P-1} buffer inter_acts_prev[P-2] in bwd_buf[P-1] (no fwd here)
      2. BACKWARD SWEEP (stages P-1 → 0, sequential cotangent chain):
            S_{P-1} bwd (fwd_input from P-1 ticks ago, labels from batch[T-(P-1)])
            → seeds cotangent dx_{P-1}
            S_{P-2} bwd (fwd_input from P-2 ticks ago, cotangent dx_{P-1})
            → dx_{P-2}
            ...
            S0 bwd (token_ids from P-1 ticks ago, cotangent dx_1)
      3. OPTIMIZER STEP (all stages, concurrent, per-stage Muon):
            each stage updates its local weights

Buffer accounting:
    bwd_bufs[s]: deque of fwd_inputs at stage s.  depth[s] = P-1-s.
      Backward fires when len(bwd_bufs[s]) > depth[s].
    inter_acts: list of P activations.  inter_acts[s] is the output of stage s
      from the PREVIOUS tick (used by stage s+1's CURRENT-tick forward).
    last_label_buf: deque of (labels_np, lw_np) pushed every tick.  Popped
      at each stage-P-1 backward to get labels for the correct batch.

Staleness profile
=================
Stage s applies its gradient (P-1-s) ticks late (matches delay_optim.grug_stage_tau).
Stage 0 is stalest (delay = P-1 = 7 for P=8); stage P-1 is fresh (delay = 0).
Characterised in delayed-gradient-pp/REPORT.md: converged token cost ~1.16×
(falling), ~1.23× with weight-prediction corrector.

Optimizer
=========
Per stage: Muon (Newton-Schulz orthogonalization) for rank≥2 weight tensors;
AdamW for rank<2 (norms, biases, router).  No cross-stage gradient communication.
"""
from __future__ import annotations

import collections
import time
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

try:
    from haliax.partitioning import set_mesh
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def set_mesh(m):
        yield


from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss

from model import GrugModelConfig, Transformer, _batch_spec, _layer_attention_masks
from levanter.grug.attention import AttentionMask

# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

_STAGE_MESH_AXIS_NAMES = ("replica_dcn", "data", "expert", "model")
_REPLICATED = P()


def _make_stage_mesh(device) -> Mesh:
    """Single-device Explicit mesh for one pipeline stage (EP=1, FSDP=1)."""
    arr = np.array([[[[device]]]], dtype=object)  # shape (1,1,1,1)
    return Mesh(
        arr,
        _STAGE_MESH_AXIS_NAMES,
        axis_types=tuple(AxisType.Explicit for _ in _STAGE_MESH_AXIS_NAMES),
    )


def _put_on(tree, mesh: Mesh):
    """Replicate a param pytree onto ``mesh`` (fully replicated within stage)."""
    return jax.device_put(tree, NamedSharding(mesh, _REPLICATED))


def _transport(x: jax.Array, dst: Mesh) -> jax.Array:
    """Move an activation array from one stage device to ``dst`` stage device."""
    return jax.device_put(x, NamedSharding(dst, _batch_spec()))


def _put_batch(arr: np.ndarray, mesh: Mesh) -> jax.Array:
    """Place a host numpy array as a batch-sharded array on ``mesh``.

    Uses ``_batch_spec()`` = ``P(_BATCH_AXES)`` which shards the first
    (batch) axis across replica_dcn×data×expert collectively.  Works for
    any array shape [B, ...] since only the first axis spec is set.
    """
    return jax.device_put(arr, NamedSharding(mesh, _batch_spec()))


# ---------------------------------------------------------------------------
# Model splitting helpers
# ---------------------------------------------------------------------------


def _split_transformer(transformer: Transformer, num_stages: int):
    """Split transformer into per-stage (array, static) pairs.

    Returns:
        stage_arrays:  list[pytree]  per-stage array leaves (num_stages)
        stage_statics: list          per-stage static info (num_stages)
    """
    cfg = transformer.config
    num_layers = cfg.num_layers
    if num_layers % num_stages != 0:
        raise ValueError(f"num_layers={num_layers} must divide num_stages={num_stages}")
    lps = num_layers // num_stages

    # Shared block static (all blocks have identical structure)
    block_static = eqx.partition(transformer.blocks[0], eqx.is_array)[1]
    block_arrays_all = [eqx.partition(b, eqx.is_array)[0] for b in transformer.blocks]

    # Embed prefix fields (stage 0)
    te_a, te_s = eqx.partition(transformer.token_embed, eqx.is_array)
    en_a, en_s = eqx.partition(transformer.embed_norm, eqx.is_array)
    egn_a, egn_s = eqx.partition(transformer.embed_gated_norm, eqx.is_array)
    eu_a = transformer.embed_up   # None or jax.Array (no static needed)
    embed_arrays = (te_a, eu_a, en_a, egn_a)
    embed_static = (te_s, None, en_s, egn_s)

    # Head suffix fields (stage P-1)
    fn_a, fn_s = eqx.partition(transformer.final_norm, eqx.is_array)
    fgn_a, fgn_s = eqx.partition(transformer.final_gated_norm, eqx.is_array)
    hd_a = transformer.head_down  # None or jax.Array
    op_a = transformer.output_proj
    head_arrays = (hd_a, fn_a, fgn_a, op_a)
    head_static = (None, fn_s, fgn_s, None)

    stage_arrays = []
    stage_statics = []
    for s in range(num_stages):
        blk_slice = tuple(block_arrays_all[s * lps : (s + 1) * lps])
        if s == 0:
            stage_arrays.append((embed_arrays, blk_slice))
            stage_statics.append(("first", embed_static, block_static))
        elif s == num_stages - 1:
            stage_arrays.append((head_arrays, blk_slice))
            stage_statics.append(("last", head_static, block_static))
        else:
            stage_arrays.append((blk_slice,))
            stage_statics.append(("mid", block_static))

    return stage_arrays, stage_statics


# ---------------------------------------------------------------------------
# Per-stage attention masks
# ---------------------------------------------------------------------------


def _build_stage_masks(cfg: GrugModelConfig, num_stages: int, *, doc_len: int = 1024) -> list:
    """Per-stage list of (per-block) AttentionMask tuples.

    The FA4 CuTe attention backend REQUIRES packed segment_ids on the mask
    (gpu_fa4_cute raises NotImplementedError when mask.segment_ids is None).
    We attach segment_ids matching the synthetic-data benchmark loader
    (synth_data.py): ~doc_len-token documents -> several segments per sequence
    (block-diagonal attention), so the async-PP attention FLOPs match the
    EP+FSDP baseline's exactly. with_sliding_window preserves segment_ids, so
    the local/global window variants inherit them.
    """
    seq_len = cfg.max_seq_len
    chunk = max(1, min(int(doc_len), seq_len))
    seg = jnp.asarray((np.arange(seq_len) // chunk).astype(np.int32))
    max_segments = (seq_len + chunk - 1) // chunk

    num_layers = cfg.num_layers
    lps = num_layers // num_stages
    base = AttentionMask.causal().with_segment_ids(seg, max_segments=max_segments)
    short_mask, long_mask = _layer_attention_masks(
        base, sliding_window=cfg.sliding_window, local_window=cfg.local_window
    )
    g = max(1, getattr(cfg, "global_attn_every", 4))
    per_layer = [long_mask if (i % g == g - 1) else short_mask for i in range(num_layers)]
    return [tuple(per_layer[s * lps : (s + 1) * lps]) for s in range(num_stages)]


# ---------------------------------------------------------------------------
# Mixed precision (matches baseline mp policy params=fp32, compute=bf16)
# ---------------------------------------------------------------------------

# The FA4 CuTe attention kernel ONLY accepts bf16/fp16 inputs (see train.py:636-639:
# "params are stored in param dtype (fp32) but the FA4 attention kernel only accepts
# bf16/fp16, and training casts via mp.cast_to_compute inside the train step").
# We store f32 master weights per stage (for the optimizer / Muon) and cast to bf16
# inside each stage's forward; the vjp upcasts grads back to f32 automatically.
_COMPUTE_DTYPE = jnp.bfloat16


def _cast_compute(tree):
    """Cast floating-point leaves to the compute dtype (bf16); leave ints/None alone."""
    return jax.tree_util.tree_map(
        lambda x: x.astype(_COMPUTE_DTYPE)
        if (eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating))
        else x,
        tree,
    )


# ---------------------------------------------------------------------------
# Stage-local forward functions
# ---------------------------------------------------------------------------


def _apply_blocks(block_arrays_slice, block_static, hidden: jax.Array, masks: tuple, remat: bool):
    """Run a stage's blocks; return (hidden, z_sum)."""
    z = jnp.zeros((), jnp.float32)
    for block_arrays, mask in zip(block_arrays_slice, masks):
        def _fwd(ba, h, _m=mask, _bs=block_static):
            return eqx.combine(ba, _bs)(h, _m)
        apply = jax.checkpoint(_fwd) if remat else _fwd
        hidden, rs = apply(block_arrays, hidden)
        z = z + rs["router_z_loss"].astype(jnp.float32)
    return hidden, z


def _embed_prefix(embed_arrays, embed_static, token_ids: jax.Array, spec) -> jax.Array:
    """Stage-0 embed prefix: lookup + optional up-proj + embed norms."""
    te_a, eu_a, en_a, egn_a = embed_arrays
    te_s, _eu_s, en_s, egn_s = embed_static
    token_embed = eqx.combine(te_a, te_s)
    embed_norm = eqx.combine(en_a, en_s)
    embed_gated_norm = eqx.combine(egn_a, egn_s)
    hidden = token_embed.at[token_ids].get(out_sharding=spec)
    if eu_a is not None:
        hidden = jnp.einsum("bse,ed->bsd", hidden, eu_a, out_sharding=spec)
    return embed_gated_norm(embed_norm(hidden))


def _head_suffix(head_arrays, head_static, hidden: jax.Array, labels: jax.Array, lw: jax.Array, spec) -> jax.Array:
    """Stage-P-1 head suffix: optional down-proj + final norms + fused CE → scalar loss."""
    hd_a, fn_a, fgn_a, op_a = head_arrays
    _hd_s, fn_s, fgn_s, _op_s = head_static
    final_norm = eqx.combine(fn_a, fn_s)
    final_gated_norm = eqx.combine(fgn_a, fgn_s)
    if hd_a is not None:
        hidden = jnp.einsum("bsd,de->bse", hidden, hd_a, out_sharding=spec)
    hidden = final_gated_norm(final_norm(hidden))
    return fused_linear_softmax_cross_entropy_loss(
        hidden, op_a, labels, weight=lw, reduction="mean",
        logsumexp_weight=None, dtype=jnp.float32,
    )


# ---------------------------------------------------------------------------
# Per-stage compiled forward + backward
# ---------------------------------------------------------------------------


class _StageFns:
    """Holds jitted fwd + vjp for one stage, keyed by stage type."""

    def __init__(self, s: int, num_stages: int, static, cfg: GrugModelConfig, masks: tuple, remat: bool):
        self.s = s
        self.is_first = (s == 0)
        self.is_last = (s == num_stages - 1)
        spec = _batch_spec()

        if self.is_first:
            embed_static, block_static = static[1], static[2]

            def _fwd(arrays, token_ids):
                # Cast f32 master weights -> bf16 compute (FA4 needs bf16); vjp upcasts
                # the grad back to f32 so the optimizer/Muon operate on f32 master weights.
                arrays = _cast_compute(arrays)
                embed_arr, blk_arr = arrays
                h = _embed_prefix(embed_arr, embed_static, token_ids, spec)
                return _apply_blocks(blk_arr, block_static, h, masks, remat)

            @jax.jit
            def forward(arrays, token_ids):
                return _fwd(arrays, token_ids)

            @jax.jit
            def backward(arrays, token_ids, dy, dz):
                # token_ids not diff'd; only arrays
                _, vjp = jax.vjp(lambda a: _fwd(a, token_ids), arrays)
                (dparams,) = vjp((dy, dz))
                return dparams

        elif self.is_last:
            head_static, block_static = static[1], static[2]

            def _fwd(arrays, hidden_in, labels, lw):
                arrays = _cast_compute(arrays)  # bf16 compute (see _cast_compute)
                head_arr, blk_arr = arrays
                h, z = _apply_blocks(blk_arr, block_static, hidden_in, masks, remat)
                ce = _head_suffix(head_arr, head_static, h, labels, lw, spec)
                return ce, z

            @jax.jit
            def forward(arrays, hidden_in, labels, lw):
                return _fwd(arrays, hidden_in, labels, lw)

            @jax.jit
            def backward(arrays, hidden_in, labels, lw, dloss, dz):
                _, vjp = jax.vjp(lambda a, h: _fwd(a, h, labels, lw), arrays, hidden_in)
                dparams, dx = vjp((dloss, dz))
                return dparams, dx

        else:
            (block_static,) = (static[1],)

            def _fwd(arrays, hidden_in):
                arrays = _cast_compute(arrays)  # bf16 compute (see _cast_compute)
                (blk_arr,) = arrays
                return _apply_blocks(blk_arr, block_static, hidden_in, masks, remat)

            @jax.jit
            def forward(arrays, hidden_in):
                return _fwd(arrays, hidden_in)

            @jax.jit
            def backward(arrays, hidden_in, dy, dz):
                _, vjp = jax.vjp(lambda a, h: _fwd(a, h), arrays, hidden_in)
                dparams, dx = vjp((dy, dz))
                return dparams, dx

        self.forward = forward
        self.backward = backward


# ---------------------------------------------------------------------------
# Muon (Newton-Schulz orthogonalization, per-stage local)
# ---------------------------------------------------------------------------

try:
    from levanter.optim.util import NEWTON_SCHULZ_COEFFICIENTS
    _NS_COEFFS = NEWTON_SCHULZ_COEFFICIENTS["quintic"]
except Exception:
    # Fallback hardcoded quintic coefficients (matches grug-moe-pp/pipeline_zb.py)
    _NS_COEFFS = [(1.5, -0.5, 0.0), (1.5, -0.5, 0.0), (1.5, -0.5, 0.0)]


def _newton_schulz(x: jax.Array, eps: float = 1e-7) -> jax.Array:
    """Newton-Schulz iteration toward an orthogonal matrix."""
    x = x / (jnp.linalg.norm(x) + eps)
    transpose = x.shape[0] > x.shape[1]
    if transpose:
        x = x.T
    for a, b, c in _NS_COEFFS:
        gram = jnp.matmul(x, x.T)
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.T if transpose else x


def _orth_leaf(g: jax.Array) -> jax.Array:
    if g.ndim < 2:
        return g
    if g.ndim == 2:
        return _newton_schulz(g)
    flat = g.reshape((-1, *g.shape[-2:]))
    return jax.vmap(_newton_schulz)(flat).reshape(g.shape)


@jax.jit
def orthogonalize_tree(tree):
    """Muon-orthogonalize all rank≥2 leaves in a pytree."""
    return jax.tree_util.tree_map(_orth_leaf, tree)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AsyncPipelineState(NamedTuple):
    """Full mutable state for the async pipeline.

    stage_arrays:   list[pytree]    per-stage weight arrays (on respective devices)
    stage_opt_st:   list            per-stage optax optimizer states
    inter_acts:     list[arr|None]  pending activation between adjacent stages
                                    inter_acts[s] = last output of stage s
                                    (input to stage s+1 in the NEXT tick)
    bwd_bufs:       list[deque]     per-stage FIFO of fwd_inputs for backward
    last_label_buf: deque           FIFO of (labels_np, lw_np) pushed every tick
    tick:           int             global tick counter (0-indexed)
    """
    stage_arrays: list
    stage_opt_st: list
    inter_acts: list
    bwd_bufs: list
    last_label_buf: object
    tick: int


# ---------------------------------------------------------------------------
# Build + step
# ---------------------------------------------------------------------------


def build_async_pipeline(
    transformer: Transformer,
    *,
    num_stages: int = 8,
    lr: float = 3e-4,
    muon: bool = True,
    remat: bool = True,
):
    """Build the async pipeline.

    Parameters
    ----------
    transformer:  Initialized Transformer (weights are split across stages).
    num_stages:   Pipeline depth (= number of H100s on the node).
    lr:           Per-stage AdamW learning rate.
    muon:         Apply Newton-Schulz orthogonalization to rank≥2 grad leaves.
    remat:        jax.checkpoint per block (saves ~50% activation memory).

    Returns
    -------
    state:      Initial AsyncPipelineState (warmup phase; backward fires after P-1 ticks).
    step_fn:    ``step(state, batch_tokens_np, loss_weight_np) -> (state, loss | None)``
                ``batch_tokens_np`` and ``loss_weight_np`` are host numpy arrays.
                Returns loss=None during warmup (first P-1 ticks).
    submeshes:  list of per-stage single-device Mesh objects.
    """
    cfg = transformer.config
    devices = jax.devices()
    if len(devices) < num_stages:
        raise ValueError(f"Need ≥{num_stages} devices, got {len(devices)}")

    # Per-stage single-device meshes
    submeshes = [_make_stage_mesh(devices[s]) for s in range(num_stages)]

    # Split params
    stage_arrays_host, stage_statics = _split_transformer(transformer, num_stages)
    stage_masks = _build_stage_masks(cfg, num_stages)

    # Compile per-stage fns
    fns = [_StageFns(s, num_stages, stage_statics[s], cfg, stage_masks[s], remat) for s in range(num_stages)]

    # Put params on respective devices
    stage_arrays = [_put_on(stage_arrays_host[s], submeshes[s]) for s in range(num_stages)]

    # Per-stage AdamW optimizers
    per_opt = [optax.adamw(learning_rate=lr, b1=0.9, b2=0.95) for _ in range(num_stages)]
    stage_opt_st = [per_opt[s].init(stage_arrays[s]) for s in range(num_stages)]

    z_coef = cfg.router_z_loss_coef / cfg.num_layers
    dz_scalar = jnp.asarray(z_coef, jnp.float32)
    dloss_one = jnp.ones((), jnp.float32)

    muon_fn = jax.jit(orthogonalize_tree) if muon else None

    initial_state = AsyncPipelineState(
        stage_arrays=stage_arrays,
        stage_opt_st=stage_opt_st,
        inter_acts=[None] * num_stages,
        bwd_bufs=[collections.deque() for _ in range(num_stages)],
        last_label_buf=collections.deque(),
        tick=0,
    )

    def step(state: AsyncPipelineState, batch_tokens_np: np.ndarray, loss_weight_np: np.ndarray):
        """One async pipeline tick.

        Accepts HOST numpy arrays for batch_tokens and loss_weight to avoid
        JAX tracing overhead on host-side buffer management.  The per-stage
        forwards are dispatched as JAX async computations (no global barrier
        between stages during the forward sweep).

        Returns (new_state, loss) where loss is None during warmup (first P-1 ticks).
        """
        # Unpack mutable state (these are Python lists/deques, not JAX traced values)
        stage_arrays = list(state.stage_arrays)
        stage_opt_st = list(state.stage_opt_st)
        inter_acts = list(state.inter_acts)  # snapshot: stage s+1 uses inter_acts[s] from PREV tick
        bwd_bufs = state.bwd_bufs
        last_label_buf = state.last_label_buf
        tick = state.tick

        B, S = batch_tokens_np.shape
        labels_np = np.concatenate(
            [batch_tokens_np[:, 1:], np.zeros((B, 1), np.int32)], axis=1
        ).astype(np.int32)

        # Push labels for this batch (popped during stage P-1 backward)
        last_label_buf.append((labels_np, loss_weight_np))

        # ======================================================================
        # 1. FORWARD SWEEP
        #    Stage 0: always runs (embeds new batch)
        #    Stage s>0: runs iff inter_acts[s-1] was set in previous tick
        #    Stage P-1: only buffers its incoming activation; no loss fwd here
        #               (backward re-runs fwd internally via vjp for correct labels)
        # ======================================================================

        new_inter_acts = [None] * num_stages

        # Stage 0 always runs
        tok_s0 = _put_batch(batch_tokens_np, submeshes[0])
        with set_mesh(submeshes[0]):
            h0, _z0 = fns[0].forward(stage_arrays[0], tok_s0)
        new_inter_acts[0] = h0
        bwd_bufs[0].append(tok_s0)   # fwd_input for stage 0's future backward

        for s in range(1, num_stages):
            prev_act = inter_acts[s - 1]  # from END of PREVIOUS tick
            if prev_act is None:
                # Still warming up: stage s hasn't received its first activation
                continue
            act_in = _transport(prev_act, submeshes[s])
            bwd_bufs[s].append(act_in)  # save for backward

            if s < num_stages - 1:
                with set_mesh(submeshes[s]):
                    hs, _zs = fns[s].forward(stage_arrays[s], act_in)
                new_inter_acts[s] = hs
            # else: s == num_stages - 1 — skip fwd (backward uses correct labels)

        # ======================================================================
        # 2. BACKWARD SWEEP (stages P-1 → 0, sequential cotangent chain)
        #    All stages fire simultaneously once the pipeline is full (tick >= P-1).
        #    During warmup (tick < P-1), bwd_bufs don't have enough items yet.
        # ======================================================================

        loss = None
        cotangent = None  # dx propagated upstream; None = downstream not ready

        for s in reversed(range(num_stages)):
            depth = num_stages - 1 - s   # how many items to age before backward fires
            if len(bwd_bufs[s]) <= depth:
                # Warmup: not enough buffered fwd_inputs yet
                cotangent = None  # can't use upstream cotangent if downstream didn't fire
                continue

            old_fwd_in = bwd_bufs[s].popleft()  # oldest fwd_input at stage s
            mesh = submeshes[s]

            if s == num_stages - 1:
                # Seed backward from head loss
                old_lbl_np, old_lw_np = last_label_buf.popleft()
                old_lbl = _put_batch(old_lbl_np, mesh)
                old_lw = _put_batch(old_lw_np, mesh)
                old_fwd_in_m = _transport(old_fwd_in, mesh)
                with set_mesh(mesh):
                    dparams, dx = fns[s].backward(
                        stage_arrays[s], old_fwd_in_m, old_lbl, old_lw, dloss_one, dz_scalar
                    )
                    # Re-run fwd for loss logging (same jit cache, no extra compile)
                    loss_v, z_v = fns[s].forward(stage_arrays[s], old_fwd_in_m, old_lbl, old_lw)
                loss = float(jax.device_get(loss_v + z_v * z_coef))
                cotangent = dx

            elif s == 0:
                if cotangent is None:
                    # Shouldn't happen in steady state; skip if somehow None
                    bwd_bufs[s].appendleft(old_fwd_in)  # put back
                    continue
                dy = _transport(cotangent, mesh)
                old_fwd_in_m = _transport(old_fwd_in, mesh)
                with set_mesh(mesh):
                    dparams = fns[s].backward(stage_arrays[s], old_fwd_in_m, dy, dz_scalar)
                cotangent = None  # stage 0 has no upstream

            else:
                if cotangent is None:
                    bwd_bufs[s].appendleft(old_fwd_in)  # put back
                    continue
                dy = _transport(cotangent, mesh)
                old_fwd_in_m = _transport(old_fwd_in, mesh)
                with set_mesh(mesh):
                    dparams, dx = fns[s].backward(stage_arrays[s], old_fwd_in_m, dy, dz_scalar)
                cotangent = dx

            # ================================================================
            # 3. OPTIMIZER STEP (per-stage, local Muon)
            # ================================================================
            if muon_fn is not None:
                dparams = muon_fn(dparams)
            updates, new_opt_st = per_opt[s].update(dparams, stage_opt_st[s], stage_arrays[s])
            stage_arrays[s] = optax.apply_updates(stage_arrays[s], updates)
            stage_opt_st[s] = new_opt_st

        new_state = AsyncPipelineState(
            stage_arrays=stage_arrays,
            stage_opt_st=stage_opt_st,
            inter_acts=new_inter_acts,
            bwd_bufs=bwd_bufs,
            last_label_buf=last_label_buf,
            tick=tick + 1,
        )
        return new_state, loss

    return initial_state, step, submeshes


# ---------------------------------------------------------------------------
# Throughput measurement (synthetic data)
# ---------------------------------------------------------------------------


def measure_throughput(
    transformer: Transformer,
    *,
    num_stages: int = 8,
    num_ticks: int = 60,
    warmup_ticks: int = 12,
    lr: float = 3e-4,
    muon: bool = True,
    remat: bool = True,
) -> dict:
    """Run the async pipeline for ``num_ticks`` and report tok/s + MFU.

    Uses synthetic random token data (SP_SYNTH_DATA=1 equivalent).
    Prints a ``[PP_THRUPUT]`` line per measurement tick.

    Returns dict with keys: tok_s, mfu_pct, step_ms, num_ticks, losses.
    """
    cfg = transformer.config
    B = 16
    S = cfg.max_seq_len

    rng = np.random.default_rng(42)
    batches = [rng.integers(0, cfg.vocab_size, (B, S)).astype(np.int32) for _ in range(num_ticks)]
    lweights = [np.ones((B, S), np.float32) for _ in range(num_ticks)]

    state, step_fn, submeshes = build_async_pipeline(
        transformer, num_stages=num_stages, lr=lr, muon=muon, remat=remat
    )

    print(f"[PP_THRUPUT] warming up {warmup_ticks} ticks (pipeline fill = {num_stages - 1} ticks)...", flush=True)
    for i in range(warmup_ticks):
        state, loss = step_fn(state, batches[i], lweights[i])
        if loss is not None:
            print(f"[PP_THRUPUT] warmup tick {i}: loss={loss:.4f}", flush=True)

    # Timed measurement
    jax.block_until_ready(state.stage_arrays)
    t0 = time.perf_counter()
    losses = []
    for i in range(warmup_ticks, num_ticks):
        state, loss = step_fn(state, batches[i], lweights[i])
        if loss is not None:
            losses.append(loss)

    jax.block_until_ready(state.stage_arrays)
    elapsed = time.perf_counter() - t0
    meas = num_ticks - warmup_ticks
    tok_s = B * S * meas / elapsed
    step_ms = elapsed / meas * 1e3

    # MFU estimation (best effort)
    try:
        from levanter.utils.flop_utils import lm_flops_per_token

        fpt = lm_flops_per_token(
            hidden_dim=cfg.hidden_dim,
            intermediate_dim=cfg.intermediate_dim,
            shared_intermediate_dim=getattr(cfg, "shared_expert_intermediate_dim", 0),
            num_layers=cfg.num_layers,
            num_kv_heads=cfg.num_kv_heads,
            num_heads=cfg.num_heads,
            seq_len=S,
            vocab_size=cfg.vocab_size,
            glu=True,
            num_experts=cfg.num_experts,
            num_shared_experts=1 if getattr(cfg, "shared_expert_intermediate_dim", 0) > 0 else 0,
            num_experts_per_tok=cfg.num_experts_per_token,
        )
        peak_tflops = num_stages * 989e12  # 989 TFLOPS/H100 BF16
        mfu = fpt * B * S / elapsed * meas / peak_tflops * 100
    except Exception:
        mfu = float("nan")

    avg_loss = sum(losses) / len(losses) if losses else float("nan")
    print(
        f"[PP_THRUPUT] ticks={meas} elapsed={elapsed:.2f}s "
        f"tok/s={tok_s:.0f} mfu={mfu:.2f}% step_ms={step_ms:.1f}ms avg_loss={avg_loss:.4f}",
        flush=True,
    )
    return {"tok_s": tok_s, "mfu_pct": mfu, "step_ms": step_ms, "num_ticks": meas, "losses": losses}


__all__ = [
    "AsyncPipelineState",
    "build_async_pipeline",
    "measure_throughput",
    "orthogonalize_tree",
]
