#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Async pipeline-parallel (PP) training entry point for megagpt MoE.

Activated when ``SP_PP_MODE=async`` is set in the environment.
Reads the same SP_* config vars as train.py, builds the model, and runs
the async no-flush pipeline (see async_pipeline.py) for SP_STEPS steps.

Key differences from the standard FSDP+EP training (train.py):
  - No EP (SP_EP is ignored; all experts are local to each stage)
  - No FSDP all-gather (params resident per stage)
  - No cross-stage gradient communication
  - Per-stage Muon (Newton-Schulz on-device, no all-reduce on ortho)
  - Staleness: stage 0 gradient is (P-1) steps stale; stage P-1 is fresh

Environment variables (subset of standard SP_* + new PP_* vars):
  SP_PP_STAGES    number of pipeline stages (default 8; should = GPU count)
  SP_PP_LR        per-stage AdamW learning rate (default 3e-4)
  SP_PP_MUON      1 (default) to use Muon; 0 for plain AdamW
  SP_PP_REMAT     1 (default) to use jax.checkpoint per block; 0 off
  SP_STEPS        number of training steps (default 60)
  SP_SYNTH_DATA   1 (default for bench) for synthetic data; 0 for real
  SP_EXPERTS      number of experts (default 128)
  SP_TOPK         experts per token (default 8)
  SP_HIDDEN       hidden dim (default 1536)
  SP_EMBED        embed dim for factorized embedding (default 512)
  SP_SEQ          sequence length (default 4096)
  SP_BATCH        batch size in sequences (default 16)

Prints [PP_THRUPUT] lines per step, then exits.  These are parsed by
monitor.py and emitted to wandb (same as [THRUPUT] lines from train.py).

Launch via iris_jobs.py sweep "pp_async".
"""
from __future__ import annotations

import os
import sys
import dataclasses
import time
import logging

import jax
import jax.numpy as jnp
import jax.random  # explicit submodule import avoids UnboundLocalError from lazy imports
import numpy as np
import optax
import equinox as eqx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P, AxisType

# Module-level imports of local files (uploaded with workspace to worker)
from levanter.grug.sharding import compact_grug_mesh
from haliax.partitioning import set_mesh
from model import Transformer, GrugModelConfig, _batch_spec
from async_pipeline import build_async_pipeline

logger = logging.getLogger(__name__)


def _parse_env():
    """Parse SP_* environment variables into a config dict."""
    e = os.environ.get
    return {
        "num_stages":   int(e("SP_PP_STAGES", "8")),
        "lr":           float(e("SP_PP_LR", "3e-4")),
        "muon":         e("SP_PP_MUON", "1") == "1",
        "remat":        e("SP_PP_REMAT", "1") == "1",
        "steps":        int(e("SP_STEPS", "60")),
        "synth_data":   e("SP_SYNTH_DATA", "1") == "1",
        "experts":      int(e("SP_EXPERTS", "128")),
        "topk":         int(e("SP_TOPK", "8")),
        "hidden_dim":   int(e("SP_HIDDEN", "1536")),
        "embed_dim":    int(e("SP_EMBED", "512") or "0") or None,  # None = no factorization
        "seq_len":      int(e("SP_SEQ", "4096")),
        "batch_size":   int(e("SP_BATCH", "16")),
        "num_layers":   int(e("SP_LAYERS", "16")),
        "vocab_size":   int(e("SP_VOCAB", "128256")),
        "intermediate": int(e("SP_INTERMEDIATE", "768")),
        "tag":          e("SP_TAG", "pp_async"),
        "group":        e("SP_GROUP", "megagpt-pp"),
    }


def _build_model_config(cfg: dict):
    """Build GrugModelConfig matching the EP+FSDP baseline geometry EXACTLY.

    Uses the same MoeAdamHHeuristic + override path as launch.py so the async-PP
    throughput is measured at the identical model geometry as the run2-e128k8
    anchor (num_layers/num_heads/num_kv_heads/intermediate are heuristic-derived;
    hardcoding them previously gave 16 heads -> the wrong B*H and a geometry
    mismatch). The critical fix vs the OOM'd jobs:

      attention_implementation="gpu_fa4_cute"

    Without it the model defaults to None -> reference_attention, which
    materializes the full f32 [B*H, S, S] score matrix (16 GiB at b16/seq4096)
    and OOMs at stage 0. The baseline never does this (BASE sets SP_ATTN=gpu_fa4_cute).
    """
    from heuristic import MoeAdamHHeuristic

    model = MoeAdamHHeuristic().build_model_config(
        cfg["hidden_dim"],
        seq_len=cfg["seq_len"],
        num_experts=cfg["experts"],
        num_experts_per_token=cfg["topk"],
        embed_dim=cfg["embed_dim"],
    )
    overrides = dict(
        remat_mode="recompute_all" if cfg["remat"] else "save_moe",
        # The fix: FlashAttention-4 CuTe backend (never materializes [B*H,S,S]).
        attention_implementation="gpu_fa4_cute",
        moe_implementation=None,  # auto-resolve -> triton ragged_dot on GPU (matches baseline)
    )
    if cfg["intermediate"] > 0:
        overrides["intermediate_dim"] = cfg["intermediate"]
    model = dataclasses.replace(model, **overrides)
    return model


def _build_wandb(cfg: dict):
    """Initialize wandb if WANDB_API_KEY is set; else return None."""
    if not os.environ.get("WANDB_API_KEY"):
        return None
    try:
        import wandb
        run = wandb.init(
            project="megagpt-speedrun",
            group=cfg["group"],
            name=f"{cfg['tag']}-pp",
            config={k: v for k, v in cfg.items()},
            tags=["pp_async", cfg["tag"]],
        )
        return run
    except Exception as e:
        logger.warning(f"wandb init failed: {e}")
        return None


def _mfu_estimate(model_cfg, steps_per_sec: float, B: int, S: int, num_devices: int) -> float:
    """Estimate model FLOPs utilization."""
    try:
        from levanter.utils.flop_utils import lm_flops_per_token
        fpt = lm_flops_per_token(
            hidden_dim=model_cfg.hidden_dim,
            intermediate_dim=model_cfg.intermediate_dim,
            shared_intermediate_dim=0,
            num_layers=model_cfg.num_layers,
            num_kv_heads=model_cfg.num_kv_heads,
            num_heads=model_cfg.num_heads,
            seq_len=S,
            vocab_size=model_cfg.vocab_size,
            glu=True,
            num_experts=model_cfg.num_experts,
            num_shared_experts=0,
            num_experts_per_tok=model_cfg.num_experts_per_token,
        )
        # 989 TFLOPS/H100 BF16
        peak = num_devices * 989e12
        return fpt * B * S * steps_per_sec / peak * 100
    except Exception:
        return float("nan")


def gpu_smoke():
    """Single-GPU smoke: verify the per-stage workload fits + runs (no 8-GPU burn).

    Triggered by SP_PP_SMOKE=1. Builds the production-geometry model, splits it,
    and runs forward+backward for the FIRST (embed+attention), a MID, and the LAST
    (head+CE) stage type on ONE device at the real per-stage shape (b16, seq4096,
    2 layers, 128 experts, gpu_fa4_cute, bf16). This directly reproduces the
    per-device memory the 8-GPU run will use, so it confirms the attention-OOM fix
    (gpu_fa4_cute + bf16 + segment_ids) WITHOUT burning an 8-GPU node.

    Prints [PP_SMOKE] lines; exits non-zero on OOM / non-finite grads.
    """
    import equinox as eqx
    from async_pipeline import (
        _StageFns, _split_transformer, _build_stage_masks,
        _make_stage_mesh, _put_on, _put_batch, orthogonalize_tree,
    )

    cfg = _parse_env()
    B, S = cfg["batch_size"], cfg["seq_len"]
    num_stages = cfg["num_stages"]
    model_cfg = _build_model_config(cfg)
    print(f"[PP_SMOKE] geometry: {model_cfg.num_layers}L hidden={model_cfg.hidden_dim} "
          f"heads={model_cfg.num_heads}/{model_cfg.num_kv_heads} I={model_cfg.intermediate_dim} "
          f"E={model_cfg.num_experts}/K{model_cfg.num_experts_per_token} attn={model_cfg.attention_implementation}",
          flush=True)

    dev = jax.devices()[0]
    init_mesh = Mesh(np.array([[[[dev]]]]), ("replica_dcn", "data", "expert", "model"),
                     axis_types=(AxisType.Explicit,) * 4)
    with set_mesh(init_mesh):
        model = Transformer.init(model_cfg, key=jax.random.PRNGKey(0))
    print("[PP_SMOKE] model initialized", flush=True)

    stage_arrays_host, stage_statics = _split_transformer(model, num_stages)
    stage_masks = _build_stage_masks(model_cfg, num_stages)
    mesh = _make_stage_mesh(dev)

    rng = np.random.default_rng(0)
    tok = rng.integers(1, model_cfg.vocab_size, (B, S)).astype(np.int32)
    labels = np.concatenate([tok[:, 1:], np.zeros((B, 1), np.int32)], axis=1).astype(np.int32)
    lw = np.ones((B, S), np.float32)
    dz = jnp.asarray(model_cfg.router_z_loss_coef / model_cfg.num_layers, jnp.float32)

    # Test the 3 distinct stage types: first (0), mid (1), last (P-1).
    test_stages = sorted(set([0, 1, num_stages - 1]))
    ok = True
    with set_mesh(mesh):
        tok_d = _put_batch(tok, mesh)
        lbl_d = _put_batch(labels, mesh)
        lw_d = _put_batch(lw, mesh)

        for s in test_stages:
            fns = _StageFns(s, num_stages, stage_statics[s], model_cfg, stage_masks[s], remat=cfg["remat"])
            arrays = _put_on(stage_arrays_host[s], mesh)
            try:
                if s == 0:
                    h, z = fns.forward(arrays, tok_d)
                    jax.block_until_ready((h, z))
                    print(f"[PP_SMOKE] stage{s}(first) fwd OK: hidden={h.shape}/{h.dtype} z={float(z):.4f}", flush=True)
                    dparams = fns.backward(arrays, tok_d, jnp.ones_like(h), dz)
                    dparams = orthogonalize_tree(dparams)
                    jax.block_until_ready(dparams)
                    g0 = jax.tree_util.tree_leaves(dparams)[0]
                    finite = bool(jnp.all(jnp.isfinite(g0)))
                    print(f"[PP_SMOKE] stage{s}(first) bwd OK: grad0={g0.shape} finite={finite}", flush=True)
                    ok = ok and finite
                elif s == num_stages - 1:
                    # Need a hidden input; reuse stage-0 output shape (bf16 [B,S,hidden]).
                    hin = jax.device_put(
                        jnp.asarray(rng.standard_normal((B, S, model_cfg.hidden_dim)) * 0.1, jnp.bfloat16),
                        NamedSharding(mesh, _batch_spec()))
                    loss, z = fns.forward(arrays, hin, lbl_d, lw_d)
                    jax.block_until_ready((loss, z))
                    print(f"[PP_SMOKE] stage{s}(last) fwd OK: loss={float(loss):.4f} z={float(z):.4f}", flush=True)
                    dparams, dx = fns.backward(arrays, hin, lbl_d, lw_d, jnp.ones((), jnp.float32), dz)
                    jax.block_until_ready((dparams, dx))
                    g0 = jax.tree_util.tree_leaves(dparams)[0]
                    finite = bool(jnp.all(jnp.isfinite(g0))) and bool(jnp.all(jnp.isfinite(dx)))
                    print(f"[PP_SMOKE] stage{s}(last) bwd OK: dx={dx.shape}/{dx.dtype} finite={finite}", flush=True)
                    ok = ok and finite
                else:
                    hin = jax.device_put(
                        jnp.asarray(rng.standard_normal((B, S, model_cfg.hidden_dim)) * 0.1, jnp.bfloat16),
                        NamedSharding(mesh, _batch_spec()))
                    h, z = fns.forward(arrays, hin)
                    jax.block_until_ready((h, z))
                    print(f"[PP_SMOKE] stage{s}(mid) fwd OK: hidden={h.shape}/{h.dtype}", flush=True)
                    dparams, dx = fns.backward(arrays, hin, jnp.ones_like(h), dz)
                    jax.block_until_ready((dparams, dx))
                    g0 = jax.tree_util.tree_leaves(dparams)[0]
                    finite = bool(jnp.all(jnp.isfinite(g0)))
                    print(f"[PP_SMOKE] stage{s}(mid) bwd OK: dx={dx.shape}/{dx.dtype} finite={finite}", flush=True)
                    ok = ok and finite
            except Exception as e:
                print(f"[PP_SMOKE] stage{s} FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
                ok = False
            del arrays

    print(f"[PP_SMOKE] {'PASS' if ok else 'FAIL'}: per-stage fwd+bwd at production shape", flush=True)
    return ok


def run_pp_async():
    """Main entry point for async PP training.

    Called from launch.py when SP_PP_MODE=async.
    Initializes the model, builds the async pipeline, runs training steps,
    logs throughput/loss, and exits.
    """
    if os.environ.get("SP_PP_SMOKE", "0") == "1":
        ok = gpu_smoke()
        import sys
        sys.exit(0 if ok else 1)

    cfg = _parse_env()
    num_stages = cfg["num_stages"]
    B = cfg["batch_size"]
    S = cfg["seq_len"]
    steps = cfg["steps"]

    logger.info(f"[PP] async pipeline: {num_stages} stages, B={B}, S={S}, steps={steps}")
    print(f"[PP_CONFIG] stages={num_stages} B={B} S={S} steps={steps} "
          f"experts={cfg['experts']} topk={cfg['topk']} hidden={cfg['hidden_dim']}", flush=True)

    n_devs = jax.device_count()
    if n_devs < num_stages:
        print(f"[PP_WARN] {n_devs} devices < {num_stages} stages; using {n_devs} stages", flush=True)
        num_stages = n_devs

    # Build model config
    model_cfg = _build_model_config(cfg)

    # Validate num_layers divisible by num_stages
    if model_cfg.num_layers % num_stages != 0:
        # Adjust num_layers
        adj = (model_cfg.num_layers // num_stages) * num_stages
        logger.warning(f"num_layers={model_cfg.num_layers} not divisible by {num_stages}; using {adj}")
        model_cfg = dataclasses.replace(model_cfg, num_layers=adj)

    # Initialize model on a single-device mesh for speed.
    # Params are split to per-stage submeshes inside build_async_pipeline.
    key = jax.random.PRNGKey(int(os.environ.get("SP_SEED", "0")))
    init_mesh = Mesh(
        np.array([[[[jax.devices()[0]]]]]),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    with set_mesh(init_mesh):
        print("[PP] initializing model...", flush=True)
        model = Transformer.init(model_cfg, key=key)
        print(f"[PP] model initialized: {model_cfg.num_layers}L × {model_cfg.hidden_dim}D × "
              f"{model_cfg.num_experts}E/{model_cfg.num_experts_per_token}K", flush=True)

    print("[PP] building async pipeline...", flush=True)
    t_build = time.perf_counter()
    state, step_fn, submeshes = build_async_pipeline(
        model,
        num_stages=num_stages,
        lr=cfg["lr"],
        muon=cfg["muon"],
        remat=cfg["remat"],
    )
    print(f"[PP] pipeline built in {time.perf_counter() - t_build:.1f}s", flush=True)

    # Warmup
    warmup_steps = num_stages + 2
    print(f"[PP] warming up {warmup_steps} steps (JIT compile + pipeline fill)...", flush=True)
    rng = np.random.default_rng(42)
    for _ in range(warmup_steps):
        b = rng.integers(0, model_cfg.vocab_size, (B, S)).astype(np.int32)
        lw = np.ones((B, S), np.float32)
        state, _ = step_fn(state, b, lw)

    # Wait for warmup JIT to complete
    jax.block_until_ready(state.stage_arrays)
    print("[PP] warmup complete; starting timed training...", flush=True)

    # Initialize wandb
    wb_run = _build_wandb(cfg)

    # Timed training
    t0 = time.perf_counter()
    losses = []
    step_times = []
    t_step = t0

    for step in range(steps):
        if cfg["synth_data"]:
            b = rng.integers(0, model_cfg.vocab_size, (B, S)).astype(np.int32)
        else:
            # Real data not yet wired; fall back to synthetic with a warning
            logger.warning("SP_SYNTH_DATA=0 requested but real data loader not yet implemented for PP; using synthetic")
            b = rng.integers(0, model_cfg.vocab_size, (B, S)).astype(np.int32)
        lw = np.ones((B, S), np.float32)

        state, loss = step_fn(state, b, lw)

        now = time.perf_counter()
        dt = now - t_step
        t_step = now

        if loss is not None:
            losses.append((step, loss))

        elapsed = now - t0
        # In steady state, each tick produces 1 batch
        toks = B * S
        tok_s = toks / dt if dt > 0 else 0
        step_ms = dt * 1e3
        step_times.append(dt)

        # MFU estimate (averaged over last 10 steps)
        if len(step_times) > 1:
            avg_dt = sum(step_times[-10:]) / min(len(step_times), 10)
            mfu = _mfu_estimate(model_cfg, 1.0 / avg_dt, B, S, n_devs)
        else:
            mfu = float("nan")

        loss_str = f" loss={loss:.4f}" if loss is not None else ""
        print(
            f"[PP_THRUPUT] step={step:4d} tok/s={tok_s:.0f} mfu={mfu:.2f}%"
            f" step_ms={step_ms:.1f}ms{loss_str} elapsed={elapsed:.0f}s",
            flush=True,
        )

        if wb_run is not None and loss is not None:
            wb_run.log({
                "train/loss": loss,
                "throughput/tok_s": tok_s,
                "throughput/mfu_pct": mfu,
                "throughput/step_ms": step_ms,
            }, step=step)

    # Final summary
    jax.block_until_ready(state.stage_arrays)
    total_elapsed = time.perf_counter() - t0
    avg_step_ms = sum(step_times) / len(step_times) * 1e3 if step_times else float("nan")
    avg_tok_s = B * S / (sum(step_times) / len(step_times)) if step_times else float("nan")
    avg_mfu = _mfu_estimate(model_cfg, 1.0 / (sum(step_times) / len(step_times)), B, S, n_devs) if step_times else float("nan")
    avg_loss = sum(l for _, l in losses) / len(losses) if losses else float("nan")

    print(
        f"\n[PP_RESULTS] steps={steps} elapsed={total_elapsed:.1f}s"
        f" avg_tok_s={avg_tok_s:.0f} avg_mfu={avg_mfu:.2f}%"
        f" avg_step_ms={avg_step_ms:.1f}ms avg_loss={avg_loss:.4f}",
        flush=True,
    )

    if wb_run is not None:
        wb_run.log({
            "summary/avg_tok_s": avg_tok_s,
            "summary/avg_mfu_pct": avg_mfu,
            "summary/avg_step_ms": avg_step_ms,
            "summary/avg_loss": avg_loss,
        })
        wb_run.finish()

    return {
        "avg_tok_s": avg_tok_s,
        "avg_mfu_pct": avg_mfu,
        "avg_step_ms": avg_step_ms,
        "avg_loss": avg_loss,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_pp_async()
    print(f"[PP_DONE] {result}")
