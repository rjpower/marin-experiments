#!/usr/bin/env python
"""Submit grug-MoE throughput/profile jobs via the iris Python client (client.submit) -- no bash
loops, no CLI shelling. Mirrors what `iris job run --gpu H100x8 --extra gpu` does, but as a single
durable python tool: it uploads the cwd workspace (launch.py/train.py/...), attaches the GPU device,
and salts each arm's run_id with SP_TAG so the marin executor does NOT skip it as "already succeeded"
(the content-addressed cache silently reuses prior artifacts otherwise -> no fresh training).

Usage:
  uv run python iris_jobs.py <sweep>  [--only suffix1,suffix2] [--tag TAG] [--gpus H100x8]
    <sweep> in {prof, dp, anchor}        # named grids defined below
  Env required: WANDB_API_KEY, HF_TOKEN  (forwarded to the worker)

Monitor with:  uv run python monitor.py <tag> --gpus 8 --warmup 8 --watch 120
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KUBECONFIG", os.path.expanduser("~/.kube/coreweave-iris-gpu"))

from iris.client import IrisClient  # noqa: E402
from iris.cluster.composer import provider_bundle  # noqa: E402
from iris.cluster.config import load_config  # noqa: E402
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec, gpu_device  # noqa: E402
from iris.cli.connect import IRIS_CLUSTER_CONFIG_DIRS, iap_config  # noqa: E402
from iris.cli.main import client_credentials, resolve_cluster_name  # noqa: E402
from rigging.config_discovery import resolve_cluster_config  # noqa: E402

# --- BASE env: D1536 d_e512, FA4, triton ragged_dot, synthetic data (no loader noise), 60 steps. ---
BASE = {
    "SP_DATA": "datakit", "SP_SYNTH_DATA": "1", "RAGGED_DOT_IMPL": "triton", "SP_ATTN": "gpu_fa4_cute",
    "SP_LOG_EVERY": "1", "SP_STEPS": "60", "SP_TOKENS": "2000000000",
    "SP_HIDDEN": "1536", "SP_EMBED": "512", "SP_EXPERTS": "128", "SP_TOPK": "8", "SP_SEQ": "4096",
    "SP_BATCH": "16", "SP_EP": "8", "SP_TP": "1", "SP_REPLICA": "1", "SP_REMAT": "save_moe",
    # Disable XLA's per-fusion autotune sub-cache: the remote (s3://) JAX_COMPILATION_CACHE_DIR can't be
    # written by XLA's C++ tsl::Env, which is FATAL for the batched_xla CE's __triton_gemm fusions
    # (levanter>=0.2.28). marin does this in resolve_training_env; the grug path bypasses it. See launch.py.
    "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none",
    # Tensorstore read cache (32GB). launch_cw.sh sets this; iris_jobs.py did NOT, which is what
    # SILENTLY HUNG cool2-sft (default 1GB -> the block-shuffle window's ~2GB working set is evicted
    # and re-fetched from R2 -> "re-fetch thrash" -> eventually one R2 GET hangs with no timeout ->
    # the data loader's get_batch() blocks forever, job stays state=running, --max-retries can't
    # recover). 32GB holds any of our real-data working sets in RAM after the first touch. No-op for
    # SP_SYNTH_DATA=1 benchmarks; not part of the run_id, so it never affects cache identity.
    "LEVANTER_TS_CACHE_LIMIT": "34359738368",
}

# --- Named sweeps. Each entry: suffix -> dict of SP_* overrides on top of BASE. ---
SWEEPS = {
    # Re-establish the EP=8 anchor with REAL (non-skipped) runs across the E-ladder.
    "anchor": {
        "e64k8":   {"SP_EXPERTS": "64",  "SP_TOPK": "8", "SP_BATCH": "16"},
        "e128k8":  {"SP_EXPERTS": "128", "SP_TOPK": "8", "SP_BATCH": "16"},
        "e256k4":  {"SP_EXPERTS": "256", "SP_TOPK": "4", "SP_BATCH": "8"},
    },
    # Does killing the EP all-to-all (DP: data=8, experts replicated+FSDP) recover MFU on E64 (fits 1 GPU)?
    # EP=1 earlier hit an XLA autotune crash; retry with autotune disabled + a level-3 fallback.
    "dp": {
        "e64dp":      {"SP_EXPERTS": "64", "SP_TOPK": "8", "SP_BATCH": "16", "SP_EP": "1"},
        "e64dpb32":   {"SP_EXPERTS": "64", "SP_TOPK": "8", "SP_BATCH": "32", "SP_EP": "1"},
        "e64ep8b32":  {"SP_EXPERTS": "64", "SP_TOPK": "8", "SP_BATCH": "32", "SP_EP": "8"},
    },
    # Profile-motivated levers on the production config (E64/K8/seq4096/EP8 b16 = 192K tok/s baseline).
    # The profile says time goes to FSDP collectives (29%) + optimizer/scatter (25%) + ATTENTION (22%),
    # expert matmul is ~0%. So the two real levers are: cheaper attention (global/local windowed) and
    # bigger batch (amortizes the fixed per-step collective/optimizer overhead). b32/EP1 OOM'd; try b24/EP8.
    "lever": {
        "e64gl":    {"SP_EXPERTS": "64", "SP_TOPK": "8", "SP_BATCH": "16",
                     "SP_GLOBAL_EVERY": "6", "SP_LOCAL_WINDOW": "1024"},
        "e64b24":   {"SP_EXPERTS": "64", "SP_TOPK": "8", "SP_BATCH": "24"},
        "e64glb24": {"SP_EXPERTS": "64", "SP_TOPK": "8", "SP_BATCH": "24",
                     "SP_GLOBAL_EVERY": "6", "SP_LOCAL_WINDOW": "1024"},
    },
    # LOSS-vs-WALL-CLOCK A/B (the decisive test for the user's goal: best loss at a TIME bound, not
    # best terminal loss per FLOP). Does a BIGGER model (more total capacity via E, or more active via
    # width) reach lower loss at matched elapsed time than the thin baseline? REAL data, constant LR.
    "lossab": {
        # GOAL: loss-vs-WALLCLOCK A/B (free TOTAL capacity via E vs ACTIVE capacity via chonky vs E64).
        # Real nemotron data, constant LR (no decay confound). Read loss-vs-elapsed at common slices
        # (60/90/120 min) + the slope at 2h. SP_STEPS is set PER ARM so each self-terminates near ~2.2h
        # despite different batch/tok-s -- equal STEPS would give 29/57/104 min (b8 vs b16, thin vs chonky)
        # and cap the comparison window at the shortest (b_bigE), the most token-starved E256 arm. Steps
        # target 2.2h = 7920s at measured tok/s: a_thin 192K@b16, b_bigE ~190K@b8 (padded), c_chonky 105K@b16.
        "a_thin":   {"SP_EXPERTS": "64",  "SP_TOPK": "8", "SP_BATCH": "16", "SP_REMAT": "save_moe",
                     "SP_SYNTH_DATA": "0", "SP_DATA": "datakit", "SP_SCHEDULE": "constant",
                     "SP_WARMUP": "0.02", "SP_STEPS": "23000", "SP_TOKENS": "5400000000"},
        "b_bigE":   {"SP_EXPERTS": "256", "SP_TOPK": "8", "SP_BATCH": "8", "SP_REMAT": "save_moe",
                     "SP_SYNTH_DATA": "0", "SP_DATA": "datakit", "SP_SCHEDULE": "constant",
                     "SP_WARMUP": "0.02", "SP_STEPS": "48000", "SP_TOKENS": "5400000000"},
        "c_chonky": {"SP_EXPERTS": "64",  "SP_TOPK": "8", "SP_BATCH": "16", "SP_INTERMEDIATE": "1536",
                     "SP_REMAT": "recompute_all", "SP_SYNTH_DATA": "0", "SP_DATA": "datakit",
                     "SP_SCHEDULE": "constant", "SP_WARMUP": "0.02", "SP_STEPS": "13000", "SP_TOKENS": "5400000000"},
    },
    # "Are we really at max HBM?" — NO: the b24/b32 OOM was a single ~29GiB *saved MoE activation*
    # buffer (SP_REMAT=save_moe in the bench BASE), not weights (static ~10-12GB/80GB). The model
    # DEFAULT is recompute_all (recompute the MoE in backward -> frees the buffer). Test whether
    # aggressive remat unlocks bigger batch + chonkier experts -> higher arithmetic intensity -> MFU.
    "mem": {
        "rc_b32":    {"SP_EXPERTS": "64", "SP_TOPK": "8", "SP_BATCH": "32", "SP_REMAT": "recompute_all"},
        "rc_b48":    {"SP_EXPERTS": "64", "SP_TOPK": "8", "SP_BATCH": "48", "SP_REMAT": "recompute_all"},
        # chonky experts: 4x wider (I=3072 vs heuristic 768) raises per-expert matmul size -> intensity
        # -> MFU, and pushes expert FLOP share up. Needs recompute_all (save_moe would OOM at 4x).
        "chonky4x":  {"SP_EXPERTS": "64", "SP_TOPK": "8", "SP_BATCH": "16", "SP_REMAT": "recompute_all",
                      "SP_INTERMEDIATE": "3072"},
    },
    # Nail the best MFU: chonky experts (confirmed +24% MFU at I=3072 but OOMs at b16) at sizes that
    # FIT, and test whether ragged_all_to_all dispatch (no all-tokens-on-all-devices materialization)
    # unlocks the batch that ring-dispatch OOMs on (29-34GiB forward transient).
    "best": {
        "chonky2x_b16":  {"SP_EXPERTS": "64", "SP_TOPK": "8", "SP_BATCH": "16", "SP_REMAT": "recompute_all",
                          "SP_INTERMEDIATE": "1536"},                 # 2x experts, should fit at b16
        "chonky4x_b8":   {"SP_EXPERTS": "64", "SP_TOPK": "8", "SP_BATCH": "8", "SP_REMAT": "recompute_all",
                          "SP_INTERMEDIATE": "3072"},                 # 4x experts, half batch -> stable high-MFU
        "rata_b24":      {"SP_EXPERTS": "64", "SP_TOPK": "8", "SP_BATCH": "24", "SP_MOE_IMPL": "ragged_all_to_all"},
    },
    # Does XLA's latency-hiding scheduler + pipelined collectives hide MORE of the ring-dispatch
    # all-gather? The trace shows ~55% already overlaps compute; these flags should push it higher by
    # pipelining the all-gather/reduce-scatter with the expert matmul. Baseline (no flags) = 192K tok/s.
    "lhs": {
        "e64lhs": {"SP_EXPERTS": "64", "SP_TOPK": "8", "SP_BATCH": "16",
                   "XLA_FLAGS": ("--xla_gpu_enable_latency_hiding_scheduler=true "
                                 "--xla_gpu_enable_pipelined_all_gather=true "
                                 "--xla_gpu_enable_pipelined_reduce_scatter=true "
                                 "--xla_gpu_enable_pipelined_all_reduce=true")},
    },
    # xprof profile of the best config (E64/K8 EP=8). SP_PROFILE=1 -> ProfilerConfig(enabled) ->
    # xprof trace -> launch.py upload patch copies it to the run's output_path/profiler on R2.
    "prof": {
        "e64k8prof": {"SP_EXPERTS": "64", "SP_TOPK": "8", "SP_BATCH": "16", "SP_EP": "8",
                      "SP_STEPS": "40", "SP_PROFILE": "1", "SP_PROF_START": "20", "SP_PROF_STEPS": "4"},
    },
    # SFT cooldown of the run2 production checkpoint. Geometry MUST match run2 (E128/K8/seq4096/b16/EP8/
    # save_moe) so train.py grafts the model params (SP_INIT_FROM = weights-only, step->0, fresh opt).
    # REAL data via the ROBUST static SFT path (data.build_sft_mix: auto_build_caches=False, pre-built
    # cache localized to the worker's local disk -> zero R2 reads -> no silent data-loader hang; the
    # cool2-sft hang was the MISSING LEVANTER_TS_CACHE_LIMIT, now in BASE). Assistant-only chat loss,
    # linear LR decay peak->0. SP_TOKENS=13.5e9 reproduces run2's pretrain peak LR (heuristic derives LR
    # from token magnitude). SP_STEPS=20000 -> ~1.31B SFT tokens (~2 epochs of the 3.3GB cache / ~10%
    # of the 13.5B pretrain budget / ~2h at run2's ~178K tok/s). Bump SP_STEPS for a longer cooldown.
    # Run: uv run python iris_jobs.py cool --tag run2cool --init-from <run2 final /checkpoints dir>
    "cool": {
        "sft": {"SP_EXPERTS": "128", "SP_TOPK": "8", "SP_BATCH": "16", "SP_EP": "8", "SP_SEQ": "4096",
                "SP_REMAT": "save_moe", "SP_SYNTH_DATA": "0", "SP_DATA": "sft",
                "SP_SCHEDULE": "linear", "SP_MIN_LR": "0", "SP_WARMUP": "0.02",
                "SP_TOKENS": "13500000000", "SP_STEPS": "20000"},
    },
    # CHEAP 1-GPU verification (submit with --gpus H100x1) BEFORE any 8-GPU PP run.
    # SP_PP_SMOKE=1 builds the production-geometry model, splits it, and runs fwd+bwd
    # for the first/mid/last stage types on ONE device at the real per-stage shape.
    # Confirms the attention-OOM fix (gpu_fa4_cute + bf16 + packed segment_ids) without
    # burning an 8-GPU node. Prints [PP_SMOKE] PASS/FAIL.
    "pp_smoke": {
        "e128k8": {
            "SP_PP_MODE": "async", "SP_PP_SMOKE": "1", "SP_PP_STAGES": "8", "SP_EP": "1",
            "SP_NO_DIST": "1", "SP_EXPERTS": "128", "SP_TOPK": "8", "SP_BATCH": "16",
        },
    },
    # ---------------------------------------------------------------------------
    # Async pipeline-parallel (PP) throughput sweep.
    #
    # Goal: measure real H100 throughput of async no-flush PP vs EP+FSDP baseline.
    # Baseline (run2-e128k8): ~187K tok/s, 15.1% MFU with E128/K8/b16/EP8.
    # PP eliminates ep_a2a (29% of step) + fsdp_comm (15%) → target ~1.7× raw speedup.
    # Staleness tax (delay_optim.py profile): ~1.16× at convergence → net ~1.47×.
    # Expected: ~275K tok/s, ~22% MFU (parametric model from h100_pp_model.py).
    #
    # All arms use:
    #   SP_PP_MODE=async  → launch.py dispatches to train_pp.run_pp_async()
    #   SP_EP=1           → no expert parallelism (experts local per stage)
    #   SP_SYNTH_DATA=1   → synthetic data (no loader noise in throughput measurement)
    #   SP_PP_STAGES=8    → 8 pipeline stages (= 8 H100s per node)
    #
    # Safety: unique pp-* prefix for all job names; check client.list_jobs(prefix="/power/")
    # before and after submit to verify no interference with run2-e128k8/spr1.
    # ---------------------------------------------------------------------------
    # Decisive GPU overlap probe (bare matmuls, seconds): does eager per-stage dispatch +
    # cross-device device_put + set_mesh OVERLAP the 8 GPUs when driven microbatch-major?
    # PROBE4/PROBE5 are the tell. Cheap; run this BEFORE building a microbatched pipeline.
    "pp_probe": {
        "probe": {"SP_PP_MODE": "probe"},
    },
    # SYNC microbatched (gradient-exact GPipe) PP -- the CORRECT overlapping pipeline.
    # Production E128/K8 geometry; SP_EP=1 (experts local). Reports overlap_factor +
    # tok/s + MFU vs the 187K/15.1% EP+FSDP anchor. M = microbatches (overlap fill).
    "pp_sync": {
        "e128k8m16": {
            "SP_PP_MODE": "sync", "SP_PP_STAGES": "8", "SP_EP": "1",
            "SP_PP_MICROBATCH": "16", "SP_EXPERTS": "128", "SP_TOPK": "8", "SP_BATCH": "16",
            "SP_PP_MUON": "1", "SP_PP_REMAT": "1", "SP_STEPS": "40",
        },
        # Larger global batch (PP's memory win lets us run batches FSDP can't): B=64, M=32.
        "e128k8b64m32": {
            "SP_PP_MODE": "sync", "SP_PP_STAGES": "8", "SP_EP": "1",
            "SP_PP_MICROBATCH": "32", "SP_EXPERTS": "128", "SP_TOPK": "8", "SP_BATCH": "64",
            "SP_PP_MUON": "1", "SP_PP_REMAT": "1", "SP_STEPS": "40",
        },
    },
    "pp_async": {
        # Primary benchmark: production E128/K8 geometry (7.62B model) under async PP
        # Compare directly to run2-e128k8 at 187K tok/s. SP_EP=1 = experts local per stage.
        "e128k8p8": {
            "SP_PP_MODE": "async", "SP_PP_STAGES": "8", "SP_EP": "1",
            "SP_EXPERTS": "128", "SP_TOPK": "8", "SP_BATCH": "16",
            "SP_PP_MUON": "1", "SP_PP_REMAT": "1",
            "SP_STEPS": "80",
        },
        # Smaller model: E64/K8 (3.9B, fits in single GPU HBM) — confirms PP overhead
        "e64k8p8": {
            "SP_PP_MODE": "async", "SP_PP_STAGES": "8", "SP_EP": "1",
            "SP_EXPERTS": "64", "SP_TOPK": "8", "SP_BATCH": "16",
            "SP_PP_MUON": "1", "SP_PP_REMAT": "1",
            "SP_STEPS": "80",
        },
        # No Muon: baseline AdamW to isolate Muon's effect on PP
        "e128k8p8nomunon": {
            "SP_PP_MODE": "async", "SP_PP_STAGES": "8", "SP_EP": "1",
            "SP_EXPERTS": "128", "SP_TOPK": "8", "SP_BATCH": "16",
            "SP_PP_MUON": "0", "SP_PP_REMAT": "1",
            "SP_STEPS": "80",
        },
        # Bigger batch: P=8 stages means the effective batch per-stage is B/P=2
        # (each stage processes B=16 seqs sharded across 1 device).
        # Try larger total batch to amortize optimizer overhead.
        "e128k8p8b32": {
            "SP_PP_MODE": "async", "SP_PP_STAGES": "8", "SP_EP": "1",
            "SP_EXPERTS": "128", "SP_TOPK": "8", "SP_BATCH": "32",
            "SP_PP_MUON": "1", "SP_PP_REMAT": "1",
            "SP_STEPS": "80",
        },
    },
    # Post-hoc held-out bpb headline (task #17): SP_EVAL_ONLY=1 -> train.py loads the ckpt (geometry
    # MUST match the run that produced it), holds out SP_VAL_SEQS seqs/component from the datakit mix as
    # validation, evals ONCE on the frozen weights, logs `eval/bpb` + per-tag, and exits (no training).
    # SP_STEPS is unused (only needed >0 so the LR-schedule build doesn't divide by zero). Pass --init-from.
    "eval": {
        "spr1": {"SP_EXPERTS": "64", "SP_TOPK": "8", "SP_BATCH": "16", "SP_EP": "8", "SP_SEQ": "4096",
                 "SP_REMAT": "save_moe", "SP_SYNTH_DATA": "0", "SP_DATA": "datakit_eval", "SP_STEPS": "1000",
                 "SP_EVAL": "1", "SP_EVAL_ONLY": "1", "SP_VAL_SEQS": "256", "SP_EVAL_BS": "16"},
    },
    # NEXT-RUN scaling smoke (task: 4x tokens*params). spr1 (E64/K8/I768, 3.97B/0.82B) ran the OLD CE
    # at ~6% MFU; the new CE + full 24h already gives ~3x tokens. design_scan.py says the lever now is a
    # BIGGER model (HBM has tens-of-B headroom) sized so experts stay FED (>= spr1's 0.67B tok/expert)
    # while MFU rises. These arms measure REAL tok/s + fit (OOM) + MFU at b16 for the fed frontier:
    #   more experts (E128/E256), more-experts+chonky (E128/I1536), bigger-D (D2048/E128),
    #   more-experts-less-sparse (E256/K16). Synthetic data, 60 steps -> ~10 min each.
    "scale": {
        "e128k8":      {"SP_EXPERTS": "128", "SP_TOPK": "8",  "SP_BATCH": "16", "SP_REMAT": "save_moe"},
        "e256k8":      {"SP_EXPERTS": "256", "SP_TOPK": "8",  "SP_BATCH": "16", "SP_REMAT": "save_moe"},
        "e256k16":     {"SP_EXPERTS": "256", "SP_TOPK": "16", "SP_BATCH": "16", "SP_REMAT": "save_moe"},
        "e128i1536":   {"SP_EXPERTS": "128", "SP_TOPK": "8",  "SP_BATCH": "16", "SP_INTERMEDIATE": "1536",
                        "SP_REMAT": "recompute_all"},
        "d2048e128":   {"SP_HIDDEN": "2048", "SP_EXPERTS": "128", "SP_TOPK": "8", "SP_BATCH": "16",
                        "SP_REMAT": "save_moe"},
    },
    # ROUND 2: sc1 showed b16 OOMs at >=14.9B total (optimizer+forward-transient, ~21-31GB, NOT freed by
    # recompute_all). E128/K8 (7.6B) is the largest at full-speed b16 (187K, 15.1%). Round 2 nails:
    #   (1) the b16 expert ceiling above E128 (e192 11.4B);  (2) chonky-vs-experts at the same 7.6B b16
    #   (e64i1536: bigger active + higher MFU);  (3) a combo (e128i1024 ~10B);  (4) the b8 throughput
    #   penalty for the 14.9B max-capacity model (e256k8b8);  (5) whether bigger-D fits b16 at all (d2048e64).
    "scale2": {
        "e192k8":     {"SP_EXPERTS": "192", "SP_TOPK": "8", "SP_BATCH": "16", "SP_REMAT": "save_moe"},
        "e64i1536":   {"SP_EXPERTS": "64",  "SP_TOPK": "8", "SP_BATCH": "16", "SP_INTERMEDIATE": "1536",
                       "SP_REMAT": "recompute_all"},
        "e128i1024":  {"SP_EXPERTS": "128", "SP_TOPK": "8", "SP_BATCH": "16", "SP_INTERMEDIATE": "1024",
                       "SP_REMAT": "recompute_all"},
        "e256k8b8":   {"SP_EXPERTS": "256", "SP_TOPK": "8", "SP_BATCH": "8",  "SP_REMAT": "save_moe"},
        "d2048e64":   {"SP_HIDDEN": "2048", "SP_EXPERTS": "64", "SP_TOPK": "8", "SP_BATCH": "16",
                       "SP_REMAT": "save_moe"},
    },
    # ROUND 3: pin the EXACT thin-expert (I768) b16 ceiling. sc1/sc2: E128 fits clean (187K/15.1%),
    # E192 OOMs. The biggest E that runs CLEAN to step 60 at b16 = the recommended full-speed model
    # (more thin experts = free total capacity, experts stay fed >= spr1's 0.675B tok/expert at 24h).
    "scale3": {
        "e144k8":  {"SP_EXPERTS": "144", "SP_TOPK": "8", "SP_BATCH": "16", "SP_REMAT": "save_moe"},
        "e160k8":  {"SP_EXPERTS": "160", "SP_TOPK": "8", "SP_BATCH": "16", "SP_REMAT": "save_moe"},
        "e176k8":  {"SP_EXPERTS": "176", "SP_TOPK": "8", "SP_BATCH": "16", "SP_REMAT": "save_moe"},
    },
    # THE NEXT-RUN PRETRAIN (recommended by NEXT_RUN.md): E128/K8 = the largest thin-expert model that
    # runs CLEAN at full-throughput b16 (187K tok/s, 15.1% MFU), 7.62B total / 0.82B active (1.9x spr1),
    # experts fed at 0.84B tok/expert. Mirrors spr1's pretrain (real datakit, constant-LR WSD stable
    # phase) but doubles total capacity via experts. ~13.5B tok / ~20h at 187K, leaving ~4h for the SFT
    # cooldown. SP_TOKENS sets the LR magnitude (heuristic), SP_STEPS the horizon. b16 risks a startup
    # OOM only above ~E144; E128 is below the fragmentation edge. (Aggressive: bump SP_EXPERTS to 144.)
    "run2": {
        "e128k8": {"SP_EXPERTS": "128", "SP_TOPK": "8", "SP_BATCH": "16", "SP_EP": "8", "SP_SEQ": "4096",
                   "SP_REMAT": "save_moe", "SP_SYNTH_DATA": "0", "SP_DATA": "datakit",
                   "SP_SCHEDULE": "constant", "SP_WARMUP": "0.02", "SP_MIN_LR": "0",
                   "SP_TOKENS": "13500000000", "SP_STEPS": "205000"},
    },
}


def connect(cluster, workspace):
    cfg = load_config(resolve_cluster_config(cluster, dirs=IRIS_CLUSTER_CONFIG_DIRS))
    name = resolve_cluster_name(cfg, None, cluster)
    creds = client_credentials(cfg, name)
    iap = iap_config(cfg)
    if iap is not None:
        return IrisClient.remote(iap.url, workspace=workspace, credentials=creds), (lambda: None)
    bundle = provider_bundle(cfg)
    addr = cfg.controller_address() or bundle.controller.discover_controller(cfg.controller)
    cm = bundle.controller.tunnel(address=addr)
    url = cm.__enter__()
    return IrisClient.remote(url, workspace=workspace, credentials=creds), (lambda: cm.__exit__(None, None, None))


def submit_one(client, name, env, gpu, cpu, memory, disk):
    variant, count = gpu.split("x")[0], int(gpu.split("x")[1])
    res = ResourceSpec(cpu=cpu, memory=memory, disk=disk)
    res.device = gpu_device(variant, count)
    job = client.submit(
        entrypoint=Entrypoint.from_command("python", "launch.py"),
        name=name,
        resources=res,
        environment=EnvironmentSpec(env_vars=env, extras=["gpu"]),
        max_retries_failure=0,
    )
    return job.job_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", choices=list(SWEEPS))
    ap.add_argument("--only", default="", help="comma-separated suffix filter")
    ap.add_argument("--tag", default="", help="job-name prefix + SP_TAG cache-bust salt (default=sweep name)")
    ap.add_argument("--gpus", default="H100x8")
    ap.add_argument("--cpu", type=float, default=32)
    ap.add_argument("--memory", default="512GB")
    ap.add_argument("--disk", default="200GB")
    ap.add_argument("--cluster", default="cw-us-east-02a")
    ap.add_argument("--init-from", default="", help="SP_INIT_FROM checkpoint path (cool sweep)")
    a = ap.parse_args()

    for k in ("WANDB_API_KEY", "HF_TOKEN"):
        if not os.environ.get(k):
            sys.exit(f"missing required env {k}")
    tag = a.tag or a.sweep
    only = set(s for s in a.only.split(",") if s)
    grid = SWEEPS[a.sweep]

    client, close = connect(a.cluster, Path.cwd())
    try:
        print(f"submitting sweep={a.sweep} tag={tag} gpus={a.gpus}")
        for suffix, ov in grid.items():
            if only and suffix not in only:
                continue
            env = {**BASE, **ov,
                   "SP_TAG": tag,
                   "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
                   "HF_TOKEN": os.environ["HF_TOKEN"],
                   "SP_GROUP": f"megagpt-{tag}"}
            if a.init_from:
                env["SP_INIT_FROM"] = a.init_from
            if a.sweep in ("cool", "eval") and not a.init_from:
                sys.exit(f"{a.sweep} sweep requires --init-from <checkpoint path>")
            jid = submit_one(client, f"{tag}-{suffix}", env, a.gpus, a.cpu, a.memory, a.disk)
            print(f"  submitted {tag}-{suffix:12s} {jid}  [{' '.join(f'{k}={v}' for k,v in ov.items())}]")
    finally:
        close()


if __name__ == "__main__":
    main()
