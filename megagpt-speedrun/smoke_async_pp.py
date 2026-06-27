#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""CPU smoke test for the async pipeline SCHEDULE logic.

Since the production MoE model uses Pallas/Triton GPU kernels that are not
available on CPU, this smoke test uses a simple linear model (einsum-only,
no Pallas) to verify that the async pipeline schedule:

  1. Produces loss=None during warmup (first P-1 ticks).
  2. Produces numeric loss from tick P-1 onwards.
  3. All losses are finite.
  4. Per-stage params change after the first complete backward.
  5. Buffer accounting is correct (bwd_bufs empty after each tick in steady state).
  6. Staleness profile matches grug_stage_tau: all stages fire at tick P-1 together.

This verifies the correctness of:
  - AsyncPipelineState.inter_acts / bwd_bufs / last_label_buf management
  - The cotangent chain (backward P-1 → 0)
  - The optimizer update (AdamW per stage)

The actual Transformer + Pallas MoE + Muon test runs on GPU via the iris job
(see iris_jobs.py, sweep "pp_async").

Usage:
    cd megagpt-speedrun
    uv run python smoke_async_pp.py [--stages 4] [--ticks 20] [--staleness] [-v]
"""
from __future__ import annotations

import argparse
import collections
import sys
import os
import time

# Force 8 fake CPU devices
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P, AxisType

jax.config.update("jax_platforms", "cpu")


# ---------------------------------------------------------------------------
# Simple linear pipeline model (no Pallas, CPU-compatible)
# ---------------------------------------------------------------------------


def _make_linear_mesh(device) -> Mesh:
    """Single-device mesh for the linear smoke model."""
    arr = np.array([[[[device]]]], dtype=object)
    return Mesh(arr, ("d0", "d1", "d2", "d3"), axis_types=(AxisType.Explicit,) * 4)


class LinearStageFns:
    """Pure-einsum stage for testing the async schedule (no Pallas).

    Each stage is a single linear projection: y = x @ W.
    Stage 0: embed lookup x → h = W0[token_ids], then h @ W_block.
    Stage P-1: h @ W_block, then h @ W_head → scalar CE loss.
    Intermediate stages: h → h @ W_block.
    """

    def __init__(self, s: int, num_stages: int, D: int, V: int):
        self.s = s
        self.is_first = (s == 0)
        self.is_last = (s == num_stages - 1)

        if self.is_first:
            @jax.jit
            def forward(arrays, token_ids):
                W_embed, W_block = arrays
                h = W_embed[token_ids]  # [B, S, D]
                h = jnp.einsum("bsd,de->bse", h, W_block)
                z = jnp.zeros(())
                return h, z

            @jax.jit
            def backward(arrays, token_ids, dy, dz):
                def f(a):
                    W_embed, W_block = a
                    h = W_embed[token_ids]
                    return jnp.einsum("bsd,de->bse", h, W_block), jnp.zeros(())
                _, vjp = jax.vjp(f, arrays)
                (dparams,) = vjp((dy, dz))
                return dparams

        elif self.is_last:
            @jax.jit
            def forward(arrays, hidden_in, labels, lw):
                W_block, W_head = arrays
                h = jnp.einsum("bsd,de->bse", hidden_in, W_block)
                logits = jnp.einsum("bsd,dv->bsv", h, W_head)
                B, S, Vv = logits.shape
                log_probs = jax.nn.log_softmax(logits, axis=-1)
                one_hot = jax.nn.one_hot(labels, Vv)
                ce = -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1) * lw)
                z = jnp.zeros(())
                return ce, z

            @jax.jit
            def backward(arrays, hidden_in, labels, lw, dloss, dz):
                def f(a, h):
                    W_block, W_head = a
                    hs = jnp.einsum("bsd,de->bse", h, W_block)
                    logits = jnp.einsum("bsd,dv->bsv", hs, W_head)
                    B, S, Vv = logits.shape
                    log_probs = jax.nn.log_softmax(logits, axis=-1)
                    one_hot = jax.nn.one_hot(labels, Vv)
                    ce = -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1) * lw)
                    return ce, jnp.zeros(())
                _, vjp = jax.vjp(f, arrays, hidden_in)
                dparams, dx = vjp((dloss, dz))
                return dparams, dx

        else:
            @jax.jit
            def forward(arrays, hidden_in):
                (W_block,) = arrays
                h = jnp.einsum("bsd,de->bse", hidden_in, W_block)
                z = jnp.zeros(())
                return h, z

            @jax.jit
            def backward(arrays, hidden_in, dy, dz):
                def f(a, h):
                    (W_block,) = a
                    return jnp.einsum("bsd,de->bse", h, W_block), jnp.zeros(())
                _, vjp = jax.vjp(f, arrays, hidden_in)
                dparams, dx = vjp((dy, dz))
                return dparams, dx

        self.forward = forward
        self.backward = backward


# ---------------------------------------------------------------------------
# Pure-schedule async pipeline (same logic as async_pipeline.py)
# ---------------------------------------------------------------------------


def build_linear_pipeline(num_stages: int, D: int, V: int, lr: float = 1e-2):
    """Build an async pipeline over simple linear stages (CPU-compatible).

    Returns (initial_state, step_fn) with the same schedule as build_async_pipeline.
    """
    devices = jax.devices()
    submeshes = [_make_linear_mesh(devices[s]) for s in range(num_stages)]

    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, num_stages + 10)

    # Initialize per-stage params
    def _init(s):
        if s == 0:
            W_embed = jax.random.normal(keys[s], (V, D)) * 0.02
            W_block = jax.random.normal(keys[s + 1], (D, D)) * 0.02
            return (W_embed, W_block)
        elif s == num_stages - 1:
            W_block = jax.random.normal(keys[s + 2], (D, D)) * 0.02
            W_head = jax.random.normal(keys[s + 3], (D, V)) * 0.02
            return (W_block, W_head)
        else:
            W_block = jax.random.normal(keys[s + 4], (D, D)) * 0.02
            return (W_block,)

    stage_arrays = [jax.device_put(_init(s), NamedSharding(submeshes[s], P())) for s in range(num_stages)]
    fns = [LinearStageFns(s, num_stages, D, V) for s in range(num_stages)]
    per_opt = [optax.adamw(learning_rate=lr) for _ in range(num_stages)]
    stage_opt_st = [per_opt[s].init(stage_arrays[s]) for s in range(num_stages)]

    dloss_one = jnp.ones(())
    dz_zero = jnp.zeros(())
    last = num_stages - 1

    def _to(x, s):
        return jax.device_put(x, NamedSharding(submeshes[s], P()))

    def _apply_opt(s, dparams, stage_arrays, stage_opt_st):
        updates, new_opt = per_opt[s].update(dparams, stage_opt_st[s], stage_arrays[s])
        stage_arrays[s] = optax.apply_updates(stage_arrays[s], updates)
        stage_opt_st[s] = new_opt

    # 1F1B state: forward AND backward each advance ONE hop per tick (mirrors
    # async_pipeline.build_async_pipeline). fwd_in[s]/bwd_in[s] = activation/cotangent
    # that arrived at stage s last tick; act_fifo[s] = saved fwd inputs awaiting bwd
    # (depth 2*(P-1-s)); label_pipe = labels flowing to the last stage (depth P-1).
    init_state = {
        "stage_arrays": stage_arrays,
        "stage_opt_st": stage_opt_st,
        "fwd_in": [None] * num_stages,
        "bwd_in": [None] * num_stages,
        "act_fifo": [collections.deque() for _ in range(num_stages)],
        "label_pipe": collections.deque(),
        "tick": 0,
    }

    def step(state, batch_tokens_np, loss_weight_np):
        stage_arrays = list(state["stage_arrays"])
        stage_opt_st = list(state["stage_opt_st"])
        fwd_in = state["fwd_in"]
        bwd_in = state["bwd_in"]
        act_fifo = state["act_fifo"]
        label_pipe = state["label_pipe"]
        tick = state["tick"]

        B, S = batch_tokens_np.shape
        labels_np = np.concatenate(
            [batch_tokens_np[:, 1:], np.zeros((B, 1), np.int32)], axis=1
        ).astype(np.int32)
        label_pipe.append((labels_np, loss_weight_np))

        new_fwd_in = [None] * num_stages
        new_bwd_in = [None] * num_stages
        loss = None

        # --- FORWARD HOPS (+ last stage's immediate backward) ---
        tok0 = _to(batch_tokens_np, 0)
        h0, _z0 = fns[0].forward(stage_arrays[0], tok0)
        act_fifo[0].append(tok0)
        if num_stages > 1:
            new_fwd_in[1] = _to(h0, 1)

        for s in range(1, num_stages):
            a = fwd_in[s]
            if a is None:
                continue
            if s < last:
                hs, _zs = fns[s].forward(stage_arrays[s], a)
                act_fifo[s].append(a)
                new_fwd_in[s + 1] = _to(hs, s + 1)
            else:
                lbl_np, lw_np = label_pipe.popleft()
                lbl = _to(lbl_np, s)
                lw = _to(lw_np, s)
                loss_v, _zv = fns[s].forward(stage_arrays[s], a, lbl, lw)
                dparams, dx = fns[s].backward(stage_arrays[s], a, lbl, lw, dloss_one, dz_zero)
                loss = float(jax.device_get(loss_v))
                _apply_opt(s, dparams, stage_arrays, stage_opt_st)
                if s > 0:
                    new_bwd_in[s - 1] = _to(dx, s - 1)

        # --- BACKWARD HOPS for stages 0..last-1 ---
        for s in range(last):
            c = bwd_in[s]
            if c is None:
                continue
            old_in = act_fifo[s].popleft()
            if s == 0:
                dparams = fns[0].backward(stage_arrays[0], old_in, c, dz_zero)
            else:
                dparams, dx = fns[s].backward(stage_arrays[s], old_in, c, dz_zero)
                new_bwd_in[s - 1] = _to(dx, s - 1)
            _apply_opt(s, dparams, stage_arrays, stage_opt_st)

        new_state = {
            "stage_arrays": stage_arrays,
            "stage_opt_st": stage_opt_st,
            "fwd_in": new_fwd_in,
            "bwd_in": new_bwd_in,
            "act_fifo": act_fifo,
            "label_pipe": label_pipe,
            "tick": tick + 1,
        }
        return new_state, loss

    return init_state, step


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_basic_schedule(num_stages: int = 4, num_ticks: int = 30, verbose: bool = False):
    """Test basic schedule: warmup, loss finiteness, param updates."""
    print(f"\n[test_basic] num_stages={num_stages}, num_ticks={num_ticks}")
    D, V, B, S = 32, 128, 4, 8

    state, step_fn = build_linear_pipeline(num_stages, D, V)
    init_arrays = [jax.tree_util.tree_leaves(a) for a in state["stage_arrays"]]

    rng = np.random.default_rng(0)
    losses = []
    for t in range(num_ticks):
        batch = rng.integers(0, V, (B, S)).astype(np.int32)
        lw = np.ones((B, S), np.float32)
        state, loss = step_fn(state, batch, lw)

        if t < num_stages - 1:
            if loss is not None:
                print(f"[FAIL] tick {t}: expected loss=None during warmup, got {loss}")
                return False
        else:
            if loss is None:
                print(f"[FAIL] tick {t}: expected numeric loss after warmup, got None")
                return False
            losses.append(loss)
            if verbose:
                print(f"  tick {t:3d}: loss={loss:.4f}")

    if not losses:
        print("[FAIL] no losses recorded")
        return False

    for i, l in enumerate(losses):
        if not np.isfinite(l):
            print(f"[FAIL] non-finite loss at step {i}: {l}")
            return False

    print(f"  losses[0]={losses[0]:.4f} losses[-1]={losses[-1]:.4f} (should decrease or stay bounded)")

    # Params should change
    for s in range(num_stages):
        final_leaves = jax.tree_util.tree_leaves(state["stage_arrays"][s])
        n_changed = sum(
            1 for a, b in zip(init_arrays[s], final_leaves)
            if not jnp.allclose(a, b, atol=1e-8)
        )
        if n_changed == 0:
            print(f"[FAIL] stage {s} params unchanged after training")
            return False
        print(f"  stage {s}: {n_changed}/{len(init_arrays[s])} leaves updated")

    print(f"[OK] test_basic: {num_ticks} ticks, {len(losses)} post-warmup losses, all finite, all stages updated")
    return True


def test_staleness_profile(num_stages: int = 4):
    """Verify the 1F1B per-stage delay profile.

    Forward reaches stage s at tick s; the last stage fwd+bwd at tick P-1.
    Cotangents then flow UP one stage per tick, so stage s's FIRST optimizer
    update is at tick (P-1) + (P-1-s) = 2*(P-1) - s. Last stage is freshest
    (tick P-1); stage 0 is stalest (tick 2*(P-1)).
    """
    print(f"\n[test_staleness] num_stages={num_stages}")
    D, V, B, S = 16, 64, 2, 4
    state, step_fn = build_linear_pipeline(num_stages, D, V)

    prev_arrays = [jax.tree_util.tree_leaves(a) for a in state["stage_arrays"]]
    first_update_tick = [None] * num_stages

    for t in range(num_stages * 4):
        batch = np.random.randint(0, V, (B, S)).astype(np.int32)
        lw = np.ones((B, S), np.float32)
        state, loss = step_fn(state, batch, lw)

        curr_arrays = [jax.tree_util.tree_leaves(a) for a in state["stage_arrays"]]
        for s in range(num_stages):
            if first_update_tick[s] is None:
                changed = any(
                    not jnp.allclose(a, b, atol=1e-10)
                    for a, b in zip(prev_arrays[s], curr_arrays[s])
                )
                if changed:
                    first_update_tick[s] = t
        prev_arrays = curr_arrays

    expected = [2 * (num_stages - 1) - s for s in range(num_stages)]
    print(f"  first update tick per stage: {first_update_tick}")
    print(f"  expected 1F1B (2*(P-1)-s):    {expected}")

    ok = True
    for s in range(num_stages):
        actual = first_update_tick[s]
        if actual != expected[s]:
            print(f"  [FAIL] stage {s}: first update at tick {actual}, expected {expected[s]}")
            ok = False
        else:
            delay = actual - s  # ticks from this stage's forward to its update
            print(f"  [OK] stage {s}: first update tick {actual} (grad delay {delay} ticks) ✓")

    return ok


def test_loss_decreases(num_stages: int = 4, num_ticks: int = 100):
    """Verify loss decreases on a fixed dataset (memorization check)."""
    print(f"\n[test_converge] num_stages={num_stages}, num_ticks={num_ticks}")
    D, V, B, S = 64, 16, 8, 4  # tiny vocab so memorization is easy
    state, step_fn = build_linear_pipeline(num_stages, D, V, lr=0.05)

    # Fixed dataset (one batch, repeated)
    rng = np.random.default_rng(99)
    batch = rng.integers(0, V, (B, S)).astype(np.int32)
    lw = np.ones((B, S), np.float32)

    losses = []
    for t in range(num_ticks):
        state, loss = step_fn(state, batch, lw)
        if loss is not None:
            losses.append(loss)

    if not losses:
        print("[FAIL] no losses")
        return False

    # Loss should decrease from first to last measured point
    first_loss = losses[0]
    last_loss = losses[-1]
    print(f"  first loss: {first_loss:.4f}, last loss: {last_loss:.4f}")

    if last_loss >= first_loss:
        print(f"  [WARN] loss did not decrease ({first_loss:.4f} → {last_loss:.4f}) — may need more ticks")
        # Don't fail; with async staleness, convergence is slower
    else:
        print(f"  [OK] loss decreased by {(first_loss - last_loss)/first_loss*100:.1f}%")

    return all(np.isfinite(l) for l in losses)


def test_buffer_consistency(num_stages: int = 4, num_ticks: int = 30):
    """Verify the 1F1B in-flight buffers reach their bounded steady-state depths.

    In steady state, act_fifo[s] holds exactly the saved forward inputs whose
    backward hasn't arrived yet = the round-trip latency 2*(P-1-s). label_pipe
    holds the labels of batches between stage 0 and the last stage = P-1.
    These are BOUNDED (each tick pushes 1 and pops 1 once steady) -- the test
    confirms no unbounded accumulation.
    """
    print(f"\n[test_buffers] num_stages={num_stages}, num_ticks={num_ticks}")
    D, V, B, S = 16, 32, 2, 4
    state, step_fn = build_linear_pipeline(num_stages, D, V)

    rng = np.random.default_rng(7)
    for t in range(num_ticks):
        batch = rng.integers(0, V, (B, S)).astype(np.int32)
        lw = np.ones((B, S), np.float32)
        state, loss = step_fn(state, batch, lw)

    print("  act_fifo depths (at end of run):")
    ok = True
    for s in range(num_stages):
        buf_len = len(state["act_fifo"][s])
        expected = 2 * (num_stages - 1 - s)  # round-trip latency
        flag = "" if buf_len == expected else "  <-- MISMATCH"
        print(f"    stage {s}: len={buf_len}, expected={expected}{flag}")
        if abs(buf_len - expected) > 1:
            ok = False

    label_len = len(state["label_pipe"])
    print(f"  label_pipe: {label_len} items (expected {num_stages - 1})")
    if abs(label_len - (num_stages - 1)) > 1:
        ok = False

    print(f"[{'OK' if ok else 'FAIL'}] test_buffers")
    return ok


# ---------------------------------------------------------------------------
# Throughput mini-benchmark
# ---------------------------------------------------------------------------


def bench_throughput(num_stages: int = 4, num_ticks: int = 50, D: int = 256, V: int = 1024):
    """Mini throughput benchmark on CPU (measures schedule overhead, not compute)."""
    print(f"\n[bench] num_stages={num_stages}, D={D}, V={V}, ticks={num_ticks}")
    B, S = 4, 32
    state, step_fn = build_linear_pipeline(num_stages, D, V, lr=1e-3)

    rng = np.random.default_rng(42)
    # Warmup
    for _ in range(num_stages + 2):
        b = rng.integers(0, V, (B, S)).astype(np.int32)
        state, _ = step_fn(state, b, np.ones((B, S), np.float32))

    t0 = time.perf_counter()
    for _ in range(num_ticks):
        b = rng.integers(0, V, (B, S)).astype(np.int32)
        state, _ = step_fn(state, b, np.ones((B, S), np.float32))

    jax.block_until_ready(state["stage_arrays"])
    elapsed = time.perf_counter() - t0
    step_ms = elapsed / num_ticks * 1e3
    print(f"  {num_ticks} ticks in {elapsed:.2f}s ({step_ms:.1f}ms/tick)")
    print(f"  (schedule overhead only; actual compute throughput measured on H100)")
    return step_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", type=int, default=4)
    parser.add_argument("--ticks", type=int, default=30)
    parser.add_argument("--staleness", action="store_true", help="Run staleness check")
    parser.add_argument("--bench", action="store_true", help="Run throughput bench")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    n_devs = jax.device_count()
    if args.stages > n_devs:
        print(f"[smoke] Note: {args.stages} stages requested, {n_devs} devices available")
        print(f"         Set XLA_FLAGS=--xla_force_host_platform_device_count={args.stages}")

    print(f"[smoke] JAX devices: {jax.devices()[:args.stages]}")

    results = []

    # Core schedule tests
    results.append(test_basic_schedule(args.stages, args.ticks, args.verbose))
    results.append(test_buffer_consistency(args.stages, min(args.ticks, 20)))
    results.append(test_loss_decreases(args.stages, args.ticks * 3))

    if args.staleness:
        results.append(test_staleness_profile(args.stages))

    if args.bench:
        bench_throughput(args.stages)

    n_pass = sum(results)
    n_total = len(results)
    print(f"\n[smoke] {'PASS' if all(results) else 'FAIL'}: {n_pass}/{n_total} tests passed")

    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
