# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Decisive GPU overlap probe for pipeline parallelism (standalone, megagpt stack).

Adapted from grug-moe-pp/overlap_probe.py to run on the megagpt 8xH100 stack with
NO marin/grug source deps (bare matmuls + this repo's `_make_stage_mesh`).

The question this answers: when the pipeline is dispatched the CORRECT way --
M microbatches x P stages, microbatch-major, no per-op host sync -- do the 8 GPUs
actually OVERLAP, or does single-thread eager dispatch serialize them?  My earlier
async_pipeline used one batch rippling with a per-tick per-stage weight update; this
probe isolates whether the *execution model* (eager per-stage jit + cross-device
device_put + set_mesh) can overlap at all on GPU, independent of any schedule.

Probes (1-3 bare matmuls; 4-5 the real pipeline dispatch shape):
  1. 2-GPU overlap            -- does eager dispatch overlap two GPUs?
  2. device_put blocking      -- does a cross-GPU transfer block the host thread?
  3. 8-hop chain vs 8 fanout  -- do transported dependencies pipeline?
  4. M mb x P stage mb-major  -- THE pipeline pattern: PIPELINES or SERIALIZES?
  5. same on real submesh+set_mesh -- is set_mesh/explicit-mesh the culprit?

Run via launch.py with SP_PP_MODE=probe (see iris_jobs.py `pp_probe`).
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

try:
    from haliax.partitioning import set_mesh
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def set_mesh(m):
        yield


from async_pipeline import _make_stage_mesh

N = 8192
DEPTH = 40


def _heavy(x):
    for _ in range(DEPTH):
        x = jnp.tanh(x @ x) * 1e-4 + 1.0
    return x


def main() -> int:
    devs = jax.devices()
    print(f"[PROBE] overlap_probe on {len(devs)} {devs[0].platform} device(s)", flush=True)
    heavy = jax.jit(_heavy)

    a = [jax.device_put(jnp.ones((N, N), jnp.float32), d) for d in devs]
    for d in range(min(8, len(devs))):
        jax.block_until_ready(heavy(a[d]))  # warmup-compile per device

    # --- probe 1: 2-GPU overlap ---
    iters = 10
    jax.block_until_ready((heavy(a[0]), heavy(a[1])))
    t = time.perf_counter()
    for _ in range(iters):
        jax.block_until_ready(heavy(a[0]))
        jax.block_until_ready(heavy(a[1]))
    serial2 = (time.perf_counter() - t) / iters
    t = time.perf_counter()
    for _ in range(iters):
        r0, r1 = heavy(a[0]), heavy(a[1])
        jax.block_until_ready((r0, r1))
    par2 = (time.perf_counter() - t) / iters
    print(
        f"[PROBE1] 2-GPU: serial={serial2*1e3:.1f}ms parallel={par2*1e3:.1f}ms "
        f"speedup={serial2/par2:.2f}x (2.0=perfect overlap, 1.0=serialized)",
        flush=True,
    )

    # --- probe 2: is cross-GPU device_put blocking the host thread? ---
    x = heavy(a[0])  # async, still computing on dev0
    t = time.perf_counter()
    y = jax.device_put(x, devs[1])  # dispatch only
    dispatch = time.perf_counter() - t
    jax.block_until_ready(y)
    one_kernel = serial2 / 2
    print(
        f"[PROBE2] device_put dispatch={dispatch*1e3:.1f}ms (one kernel={one_kernel*1e3:.1f}ms). "
        + ("BLOCKS host (serializes pipeline)" if dispatch > 0.4 * one_kernel else "async (does not block)"),
        flush=True,
    )

    # --- probe 3: transported 8-stage chain vs 8 independent kernels ---
    p = min(8, len(devs))
    t = time.perf_counter()
    for _ in range(iters):
        h = a[0]
        for s in range(p):
            h = jax.device_put(h, devs[s])
            h = heavy(h)
        jax.block_until_ready(h)
    chain = (time.perf_counter() - t) / iters
    t = time.perf_counter()
    for _ in range(iters):
        outs = [heavy(a[s]) for s in range(p)]
        jax.block_until_ready(outs)
    fanout = (time.perf_counter() - t) / iters
    print(
        f"[PROBE3] {p}-hop transported chain={chain*1e3:.1f}ms | {p} independent kernels={fanout*1e3:.1f}ms "
        f"| chain/fanout={chain/fanout:.2f}x",
        flush=True,
    )

    # --- probe 4: M microbatches x P stages, mb-major dispatch (EXACTLY the pipeline pattern) ---
    m_count = 8
    t = time.perf_counter()
    for _ in range(iters):
        finals = []
        for _m in range(m_count):
            h = a[0]
            for s in range(p):
                h = jax.device_put(h, devs[s])
                h = heavy(h)
            finals.append(h)
        jax.block_until_ready(finals)
    pipe = (time.perf_counter() - t) / iters
    ideal = (m_count + p - 1) * one_kernel
    serial = m_count * p * one_kernel
    print(
        f"[PROBE4] {m_count}mb x {p}stage mb-major={pipe*1e3:.0f}ms | pipelined-ideal={ideal*1e3:.0f}ms "
        f"serial={serial*1e3:.0f}ms | "
        + ("PIPELINES" if pipe < 0.6 * serial else "SERIALIZES")
        + f" (overlap eff={100.0*(serial-pipe)/(serial-ideal):.0f}%)",
        flush=True,
    )

    # --- probe 5: same pipeline pattern but with REAL per-stage submesh + set_mesh ---
    submeshes = [_make_stage_mesh(devs[s]) for s in range(p)]
    heavy_j = jax.jit(_heavy)
    repl = [NamedSharding(submeshes[s], P()) for s in range(p)]
    x0 = jax.device_put(jnp.ones((N, N), jnp.float32), repl[0])
    for s in range(p):
        with set_mesh(submeshes[s]):
            jax.block_until_ready(heavy_j(jax.device_put(jnp.ones((N, N), jnp.float32), repl[s])))
    t = time.perf_counter()
    for _ in range(iters):
        finals = []
        for _m in range(m_count):
            h = x0
            for s in range(p):
                if s > 0:
                    h = jax.device_put(h, repl[s])
                with set_mesh(submeshes[s]):
                    h = heavy_j(h)
            finals.append(h)
        jax.block_until_ready(finals)
    pipe5 = (time.perf_counter() - t) / iters
    print(
        f"[PROBE5] {m_count}mb x {p}stage submesh+set_mesh={pipe5*1e3:.0f}ms | pipelined-ideal={ideal*1e3:.0f}ms "
        f"serial={serial*1e3:.0f}ms | "
        + ("PIPELINES" if pipe5 < 0.6 * serial else "SERIALIZES (set_mesh/explicit-mesh transport is the culprit)"),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
