# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Worker-side diagnostic for the MoE grouped-matmul (ragged_dot) backend.

The 15B MoE OOMs at ~278 GiB on 8xH100. Root cause (confirmed locally): the
``haliax.nn.ragged_dot`` *XLA* fallback (``jax.lax.ragged_dot_general``) lowers the
backward pass to a dense per-expert ``[G, M, N]`` fp32 tensor. For a deeply-sparse
MoE (G=E/EP experts/shard, M=local_capacity ~ 1.25*tokens*K/EP) that is hundreds of
GiB. The streaming Pallas-Triton kernel avoids it entirely. jaxlib *bundles* triton,
so the kernel *should* be available -- yet the run OOMed, meaning auto-selection fell
back to XLA. This script answers, on the actual H100 worker:

  * is ``_has_pallas_triton`` true, and what does ``_preferred_implementations('auto')``
    return on this backend?
  * does the Pallas-Triton ragged_dot fwd+bwd actually COMPILE and RUN here?
  * if auto silently falls back to XLA, what exception triggers it?

Gated behind ``SP_DIAG=ragged`` in launch.py so it reuses the normal job bundle.
"""
from __future__ import annotations

import os
import traceback


def ring() -> None:
    """Replicate the EXACT EP ring MoE call (real mesh + shard_map + grad).

    In isolation (plain jit) ``ragged_dot`` auto-selects triton and uses ~0.6GB.
    Training OOMs on the XLA dense ``[G,M,N]`` fallback. The only difference is the
    calling context: training runs ragged_dot inside ``moe_mlp(implementation='ring')``
    which wraps it in a ``shard_map`` over the expert axis. This probes whether that
    shard_map context forces the XLA fallback (peak scales with G*M*N) vs triton
    (tiny peak), at a small M so the XLA path can't OOM the probe itself.
    """
    import jax
    import jax.numpy as jnp
    from haliax.partitioning import set_mesh

    from levanter.grug.grug_moe import moe_mlp
    from levanter.grug.sharding import compact_grug_mesh

    EP = 8
    mesh = compact_grug_mesh(expert_axis_size=EP, replica_axis_size=1, model_axis_size=1)
    print("[diag-ring] mesh:", mesh.shape, flush=True)

    E, D, I = 256, 1536, 768
    # dense XLA [G=E/EP, M=1.25*T, N] fp32 ≈ 245760 bytes * T. T=65536 -> ~16GB dense
    # (fits, but clearly separable from triton's ~3GB); the real run uses T≈1.05M -> ~258GB.
    T, K = 65536, 8
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    x = jax.random.normal(k1, (T, D), jnp.bfloat16)
    w13 = jax.random.normal(k2, (E, D, 2 * I), jnp.bfloat16) * 0.02
    w2 = jax.random.normal(k3, (E, I, D), jnp.bfloat16) * 0.02
    sel = jax.random.randint(k4, (T, K), 0, E, jnp.int32)
    cw = jnp.ones((T, K), jnp.bfloat16)

    def loss(x, w13, w2):
        out = moe_mlp(x, sel, cw, w13, w2, implementation="ring", mesh=mesh)
        if isinstance(out, tuple):
            out = out[0]
        return out.astype(jnp.float32).sum()

    with set_mesh(mesh):
        for impl_env in (None, "triton", "xla"):
            import os as _os

            if impl_env is None:
                _os.environ.pop("RAGGED_DOT_IMPL", None)
                tag = "auto"
            else:
                _os.environ["RAGGED_DOT_IMPL"] = impl_env
                tag = impl_env
            print(f"[diag-ring] === ring grad, RAGGED_DOT_IMPL={tag}, T={T} ===", flush=True)
            try:
                for d in jax.devices():
                    d.memory_stats()
                f = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))
                gs = f(x, w13, w2)
                jax.block_until_ready(gs)
                peak = max((d.memory_stats() or {}).get("peak_bytes_in_use", 0) for d in jax.devices())
                # XLA dense [G=E/EP, M~capacity, N] would be hundreds of MB even at T=8192;
                # triton stays tiny. Print so we can tell which path ran.
                print(f"[diag-ring] OK  peak={peak/1e9:.2f}GB  (triton≈small, XLA-dense≈large)", flush=True)
            except Exception as e:  # noqa: BLE001
                import traceback as _tb

                print(f"[diag-ring] FAILED RAGGED_DOT_IMPL={tag}: {type(e).__name__}: {e}", flush=True)
                _tb.print_exc()
        _os = __import__("os")
        _os.environ.pop("RAGGED_DOT_IMPL", None)
    print("[diag-ring] done", flush=True)


def fa4() -> None:
    """Validate the FA4 (CuTe + flash-attn-4) stack on the actual H100.

    Confirms (a) the optional deps import (cutlass.cute/.jax, cuda.bindings.driver,
    flash_attn.cute.flash_bwd_*), and (b) ``gpu_fa4_cute_attention`` fwd+bwd compiles
    and runs -- before we wire FA4 into the model as the attention backend (replacing
    the O(S^2) reference path that OOMs/slows the big-seq MoE).
    """
    import traceback

    import jax
    import jax.numpy as jnp

    print("[diag-fa4] backend=", jax.default_backend(), "devices=", len(jax.devices()), flush=True)
    for mod in ("cutlass.cute", "cutlass.jax", "cuda.bindings.driver",
                "flash_attn.cute.flash_bwd_preprocess", "flash_attn.cute.flash_bwd_postprocess"):
        try:
            __import__(mod)
            print(f"[diag-fa4] import OK: {mod}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[diag-fa4] import FAILED: {mod}: {type(e).__name__}: {e}", flush=True)

    from levanter.grug.attention._core import AttentionMask
    from levanter.grug.attention._fa4_cute import gpu_fa4_cute_attention
    from levanter.grug.attention._fa4_cute_backend import cutlass_cute_available

    print("[diag-fa4] cutlass_cute_available() =", cutlass_cute_available(), flush=True)

    B, S, H, D = 2, 1024, 12, 128  # real head_dim=128
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    q = jax.random.normal(k1, (B, S, H, D), jnp.bfloat16)
    k = jax.random.normal(k2, (B, S, H, D), jnp.bfloat16)
    v = jax.random.normal(k3, (B, S, H, D), jnp.bfloat16)
    mask = AttentionMask.causal()

    def loss(q, k, v):
        out = gpu_fa4_cute_attention(q, k, v, mask)
        return out.astype(jnp.float32).sum()

    for label, fn in (("fwd", jax.jit(loss)), ("fwd+bwd", jax.jit(jax.grad(loss, argnums=(0, 1, 2))))):
        try:
            r = fn(q, k, v)
            jax.block_until_ready(r)
            shp = r.shape if hasattr(r, "shape") else [x.shape for x in r]
            print(f"[diag-fa4] {label} OK -> {shp}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[diag-fa4] {label} FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    print("[diag-fa4] done", flush=True)


def fit() -> None:
    """Compile-only probe of the real model train step to find the 278 GiB op.

    The OOM fires at *runtime* (compile succeeds), so we can lower+compile the exact
    fwd+bwd graph WITHOUT executing it -- no allocation, no OOM -- and read the largest
    buffers from the compiled HLO + ``memory_analysis()``. Reuses launch.py's model
    builders so the geometry matches training exactly. Env: SP_HIDDEN/SP_EMBED/
    SP_EXPERTS/SP_TOPK/SP_SEQ/SP_EP/SP_BATCH (same names as the real launch).
    """
    import os
    import re

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from haliax.partitioning import set_mesh
    from jax.sharding import NamedSharding, PartitionSpec

    from heuristic import MoeAdamHHeuristic
    from levanter.grug.sharding import compact_grug_mesh
    from model import AttentionMask, Transformer

    H = int(os.environ.get("SP_HIDDEN", "1536"))
    DE = int(os.environ.get("SP_EMBED", "512"))
    E = int(os.environ.get("SP_EXPERTS", "256"))
    K = int(os.environ.get("SP_TOPK", "8"))
    S = int(os.environ.get("SP_SEQ", "2048"))
    EP = int(os.environ.get("SP_EP", "8"))
    B = int(os.environ.get("SP_BATCH", "512"))
    remat = os.environ.get("SP_REMAT", "recompute_all")
    print(f"[diag-fit] D={H} d_e={DE} E={E} K={K} seq={S} EP={EP} batch={B} remat={remat}", flush=True)

    mesh = compact_grug_mesh(expert_axis_size=EP, replica_axis_size=1, model_axis_size=1)
    cfg = MoeAdamHHeuristic().build_model_config(
        H, seq_len=S, num_experts=E, num_experts_per_token=K, embed_dim=DE
    )
    import dataclasses

    cfg = dataclasses.replace(cfg, remat_mode=remat)
    print(f"[diag-fit] layers={cfg.num_layers} heads={cfg.num_heads} head_dim={cfg.inferred_head_dim} "
          f"I_expert={cfg.intermediate_dim} vocab={cfg.vocab_size}", flush=True)

    with set_mesh(mesh):
        model = Transformer.init(cfg, key=jax.random.PRNGKey(0))
        bspec = NamedSharding(mesh, PartitionSpec(("replica_dcn", "data", "expert"), None))
        tokens = jax.device_put(jnp.ones((B, S), jnp.int32), bspec)
        lw = jax.device_put(jnp.ones((B, S), jnp.float32), bspec)

        # Use a RAW jax.jit (not eqx.filter_jit) so the Compiled exposes .as_text()/
        # .cost_analysis() (eqx's wrapper hides them, and jax 0.10 dropped memory_analysis).
        # Partition the eqx model into (arrays, static) and grad only the arrays.
        arr, static = eqx.partition(model, eqx.is_array)

        def loss_fn(arr_part, tok, w):
            m = eqx.combine(arr_part, static)
            return m.next_token_loss(tok, w, mask=AttentionMask.causal(), reduction="mean")

        grad_fn = jax.jit(jax.grad(loss_fn))
        print("[diag-fit] lowering+compiling fwd+bwd (no execution)...", flush=True)
        try:
            compiled = grad_fn.lower(arr, tokens, lw).compile()
        except Exception as e:  # noqa: BLE001
            import traceback

            print(f"[diag-fit] COMPILE FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            return
        try:
            ca = compiled.cost_analysis()
            peak = ca.get("bytes accessed") if isinstance(ca, dict) else None
            print(f"[diag-fit] cost_analysis flops={ca.get('flops') if isinstance(ca, dict) else '?'} "
                  f"bytes_accessed={peak}", flush=True)
        except Exception as e:  # noqa: BLE001
            print("[diag-fit] cost_analysis unavailable:", repr(e), flush=True)
        txt = compiled.as_text()
        # Tag the biggest single buffers with the HLO op line that defines them, so we can
        # see WHICH op (not just which shape) is the memory hog.
        big_lines = []
        for ln in txt.splitlines():
            mm = re.search(r"(f32|bf16|f16|s32)\[([\d,]+)\]", ln)
            if not mm:
                continue
            dims = [int(x) for x in mm.group(2).split(",") if x]
            prod = 1
            for d in dims:
                prod *= d
            nbytes = prod * (4 if mm.group(1) in ("f32", "s32") else 2)
            if nbytes >= 20e9:
                big_lines.append((nbytes, ln.strip()[:240]))
        big_lines.sort(reverse=True)
        print(f"[diag-fit] HLO lines with a buffer >=20GB ({len(big_lines)}):", flush=True)
        for nbytes, ln in big_lines[:12]:
            print(f"    >> {nbytes/1e9:8.2f}GB  {ln}", flush=True)
        sizes = {}
        for m in re.finditer(r"(f32|bf16|f16|s32)\[([\d,]+)\]", txt):
            dims = [int(x) for x in m.group(2).split(",") if x]
            if not dims:
                continue
            prod = 1
            for d in dims:
                prod *= d
            itemsize = 4 if m.group(1) in ("f32", "s32") else 2
            key = (m.group(1), tuple(dims))
            sizes[key] = max(sizes.get(key, 0), prod * itemsize)
        top = sorted(sizes.items(), key=lambda kv: kv[1], reverse=True)[:12]
        print("[diag-fit] top 12 distinct buffer shapes (GB, dtype, shape):", flush=True)
        for (dt, dims), nbytes in top:
            print(f"    {nbytes/1e9:8.2f}  {dt}  {dims}", flush=True)
    print("[diag-fit] done", flush=True)


def main() -> None:
    import jax
    import jax.numpy as jnp

    print("=" * 72, flush=True)
    print("[diag] jax", jax.__version__, "backend=", jax.default_backend(), flush=True)
    devs = jax.devices()
    print("[diag] devices:", len(devs), devs[0].device_kind if devs else "none", flush=True)

    # NB: ``haliax.nn.ragged_dot`` is re-exported as a *function* in haliax.nn, which
    # shadows the submodule under ``import ... as``; grab the real module explicitly.
    import importlib

    rd = importlib.import_module("haliax.nn.ragged_dot")

    print("[diag] _has_pallas_triton =", rd._has_pallas_triton, flush=True)
    print("[diag] _gmm_megablox is None =", rd._gmm_megablox is None, flush=True)
    try:
        print("[diag] preferred(auto) =", rd._preferred_implementations("auto"), flush=True)
    except Exception as e:  # noqa: BLE001
        print("[diag] preferred(auto) raised:", repr(e), flush=True)

    # Per-shard ring shapes for the headline geometry, but a modest M so the XLA
    # path won't itself OOM the diagnostic. The point is which path COMPILES/RUNS,
    # not throughput. K=D=1536, G=E/EP=256/8=32, N=I2=2*768=1536.
    G, M, K, N = 32, 8192, 1536, 1536
    key = jax.random.PRNGKey(0)
    lhs = jax.random.normal(key, (M, K), jnp.bfloat16)
    rhs = jax.random.normal(key, (G, K, N), jnp.bfloat16)
    gs = jnp.full((G,), M // G, jnp.int32)

    def loss(lhs, rhs, gs):
        return rd.ragged_dot(lhs, rhs, gs).astype(jnp.float32).sum()

    for impl in ("auto", "triton", "xla"):
        if impl == "auto":
            os.environ.pop("RAGGED_DOT_IMPL", None)
        else:
            os.environ["RAGGED_DOT_IMPL"] = impl
        print("-" * 72, flush=True)
        print(f"[diag] === RAGGED_DOT_IMPL={impl} (fwd+bwd, M={M}) ===", flush=True)
        try:
            for d in devs:
                d.memory_stats() and None
            f = jax.jit(jax.grad(loss, argnums=(0, 1)))
            dl, dr = f(lhs, rhs, gs)
            dl.block_until_ready()
            dr.block_until_ready()
            try:
                peak = max((d.memory_stats() or {}).get("peak_bytes_in_use", 0) for d in devs)
                print(f"[diag] OK  dlhs={dl.shape} drhs={dr.shape} peak={peak/1e9:.2f}GB", flush=True)
            except Exception:  # noqa: BLE001
                print(f"[diag] OK  dlhs={dl.shape} drhs={dr.shape}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[diag] FAILED impl={impl}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    os.environ.pop("RAGGED_DOT_IMPL", None)
    print("=" * 72, flush=True)
    print("[diag] done", flush=True)
