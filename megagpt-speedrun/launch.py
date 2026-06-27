# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adaptive-sparsity sweep launcher for the grug MoE template.

Runs grug-MoE training directly on the TPU job (``run_inline`` -> ``_run_grug_local``,
no Fray driver/dispatch hop). One module submits every arm of the sparsity sweep;
the arm is selected entirely by environment variables. Every arm shares the same
batch size and step count (iso-token) and the same AdamH learning rate (the
heuristic LR depends only on hidden dim / tokens, not on the expert geometry), so
the only thing that varies across arms is the routing sparsity.

    SPARSITY_MODE   fixed | adaptive               (default fixed)
    SP_HIDDEN       model hidden dim               (default 512 -> ~2.6B total at E=1024)
    SP_EXPERTS      number of experts E            (default 1024; E=1024 reaches 1/1024 active at K=1)
    SP_TOPK         top-k routing width K          (default 4; the K_max capacity in adaptive mode)
    SP_MIN_K        adaptive per-token floor        (default 0; 0 lets a token use the shared expert alone)
    SP_COEF         sparsity penalty weight λ       (default 0.0; only meaningful when SPARSITY_MODE=adaptive)
    SP_TEMP         soft keep-gate temperature      (default 1.0)
    SP_TOKENS       target token budget             (e.g. 10e9, 100e9; sets batch/steps/LR via the heuristic)
    SP_REF_TOPK     reference K for token sizing    (default 4; iso-token sizing uses this fixed K)
    SP_BUDGET       compute budget (if SP_TOKENS unset) (default 1.7e18)
    SP_STEPS        override train steps            (default: heuristic value; overriding desyncs LR)
    SP_BATCH        override batch size (sequences) (default: heuristic value; overriding desyncs LR)
    SP_DATA         datakit | fineweb | nemotron    (default datakit; CoreWeave R2 nemotron parquet)
    SP_DATA_REGION  GCS region for nemotron caches  (default us-east5; match the run's --region)
    SP_EMBED        factorized embedding/CE dim d_e (default ""/None -> d_e = SP_HIDDEN, no factorization)
    SP_SEQ          sequence length                 (default 4096)
    SP_EP           expert_axis_size (expert parallel) (default 1)
    SP_TP           model_axis_size (tensor parallel)  (default 1; data = devices/(EP*TP))
    SP_REPLICA      replica_axis_size               (default ""/None=process_count; set 1 for single-node FSDP)
    SP_SEED         seed                            (default 0)
    SP_TPU          TPU type                        (default v6e-16; vestigial on GPU/inline path)
    SP_GROUP        wandb group                     (default adaptive-sparsity)
    SP_SMOKE        1 -> use a single-split subset for a fast cluster smoke check (default 0)

Submit one arm directly on a TPU (training runs in-process on the job that holds the
TPU). Do not pin ``--region`` or ``MARIN_PREFIX``: the worker derives its own region
bucket from VM metadata, so iris can take v6e capacity anywhere and the FineWeb-Edu
cache (HF-backed) materializes in whatever region you land — no cross-region guard.
Pass the arm via ``-e`` so it reaches the remote worker that runs this module:

    uv run iris --cluster=marin job run --no-wait \
      --tpu v6e-8 --enable-extra-resources --extra marin-core:tpu \
      --max-retries 1 --cpu 32 --memory 128GB --disk 50GB \
      -e WANDB_API_KEY "$WANDB_API_KEY" \
      -e SPARSITY_MODE fixed -e SP_EXPERTS 128 -e SP_TOPK 4 \
      -- python launch.py
"""

import dataclasses
import os
from dataclasses import dataclass, field
from datetime import timedelta

# XLA's per-fusion autotune (and kernel) sub-cache is written via C++ tsl::Env, which only understands
# LOCAL paths -- it CANNOT write the s3://-backed JAX_COMPILATION_CACHE_DIR that marin sets on the
# CoreWeave workers. On the streaming `batched_xla` GPU cross-entropy (levanter>=0.2.28) the LM-head
# matmuls become __triton_gemm fusions that XLA must autotune, and the failed s3:// cache write is
# FATAL: "Could not get config for HLO ... UNIMPLEMENTED: File system scheme 's3' not implemented" ->
# "Failed to get configs for N instructions" -> train_step compile dies. marin's resolve_training_env
# disables the sub-cache for remote caches, but the grug `_run_grug_local` path bypasses it. Replicate
# the fix HERE, before jax is imported. No-op when the cache is local/unset. (See
# marin.training.training._disable_xla_autotune_subcache.)
if "://" in os.environ.get("JAX_COMPILATION_CACHE_DIR", ""):
    os.environ.setdefault("JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES", "none")

# MUST be imported BEFORE marin/levanter so its in-process S3 env override + s3fs client patch are
# installed before any cached boto/s3fs session is created at their import time (otherwise an s3fs
# instance gets cached pointing at iris's R2 endpoint and our cwobject writes leak to R2).
import cw_patch  # noqa: F401  -- redirects S3 to cwobject (virtual-hosted) when LEVANTER_S3_VIRTUAL_HOSTED=1

import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LmDataConfig
from levanter.optim import OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.distributed import DistributedConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned
from marin.training.training import temporary_checkpoint_base_path

from data import (
    build_fineweb_edu_mix,
    build_nemotron_cw_mix,
    build_nemotron_datakit_eval_mix,
    build_nemotron_datakit_mix,
    build_nemotron_mix,
)
from heuristic import MoeAdamHHeuristic, build_from_heuristic, compute_flops_per_token
from model import GrugModelConfig
from train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, _run_grug_local


@dataclass(frozen=True)
class GrugMoeLaunchConfig:
    """Last-mile run config for the MoE grug template (inlined for self-containment)."""

    model: GrugModelConfig
    data: LmDataConfig
    output_path: str
    run_id: str
    resources: ResourceConfig
    steps: int
    batch_size: int
    seed: int
    mp: str  # jmp policy string, e.g. "params=float32,compute=bfloat16,output=bfloat16".
    tracker: TrackerConfig
    optimizer: OptimizerConfig
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    checkpointer: CheckpointerConfig | None = None
    k_schedule: tuple[tuple[int, int], ...] | None = None
    # Weights-only init from a prior run's checkpoint dir (SFT cooldown <- pretrain). None = fresh.
    init_from: str | None = None


def _resolve_tracker(tracker: TrackerConfig, run_id: str) -> TrackerConfig:
    if isinstance(tracker, WandbConfig):
        return dataclasses.replace(tracker, name=run_id)
    return tracker


def run_inline(config: GrugMoeLaunchConfig) -> None:
    """Run grug-MoE training in-process (no Fray driver/dispatch hop)."""
    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        profiler=config.profiler,
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id),
        use_explicit_mesh_axes=True,
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        # Single-GPU benchmark jobs (SP_NO_DIST=1) skip jax.distributed init entirely: a 1-device
        # process needs no coordinator, and iris co-locates many H100x1 jobs on one 8-GPU node where
        # they would otherwise collide on the node-local JAX coordinator port (segfault / "different
        # incarnation" abort). Multi-GPU jobs leave this True so iris.runtime.jax_init runs normally.
        distributed=DistributedConfig(
            initialize_jax_distributed=(os.environ.get("SP_NO_DIST", "0") != "1")
        ),
        # Weights-only init from a prior checkpoint (SFT cooldown <- pretrain); None for pretrain.
        # train.py grafts only the model params (keeps step=0 + fresh optimizer) when this is set
        # and the run has no checkpoint of its own yet.
        initialize_from=config.init_from,
        checkpointer=config.checkpointer
        or CheckpointerConfig(
            base_path=os.path.join(config.output_path, "checkpoints"),
            temporary_base_path=temporary_checkpoint_base_path(config.output_path),
            append_run_id_to_base_path=False,
            # Long (10B-100B token) runs on preemptible v6e need to survive preemption: save
            # often and keep the last couple of rolling checkpoints so an iris auto-retry (or
            # a manual re-submit of the same run_id) resumes from the latest step instead of
            # restarting from zero. The nemotron caches are pre-built, so a retry just reads
            # them and resumes -- there is no per-region build race to re-trip.
            save_interval=timedelta(minutes=30),
            keep_last_temporary_checkpoints=2,
            keep=None,
        ),
    )
    # Eval wiring (post-hoc bpb headline). Pretrain/cooldown run with eval=None (convergence metric is
    # train/loss; no validation split in the mixtures). SP_EVAL=1 holds out SP_VAL_SEQS sequences per
    # component as a held-out validation set and attaches a bpb evaluator; with SP_EVAL_ONLY=1 train.py
    # evals the loaded checkpoint once and exits (no training). Defaults keep training behaviour identical.
    data = config.data
    eval_cfg = None
    if os.environ.get("SP_EVAL") == "1":
        val_seqs = int(os.environ.get("SP_VAL_SEQS", "256"))
        eval_bs = int(os.environ.get("SP_EVAL_BS", "16"))
        max_eb = os.environ.get("SP_EVAL_MAXB", "")
        data = dataclasses.replace(data, num_validation_sequences={n: val_seqs for n in data.components})
        eval_cfg = GrugEvalConfig(
            eval_batch_size=eval_bs,
            steps_per_eval=10**9,  # never fires during training; the SP_EVAL_ONLY path calls it directly
            max_eval_batches=(int(max_eb) if max_eb else None),
            eval_current=True,
            eval_ema=False,
            compute_bpb=True,
        )
    grug_trainer = dataclasses.replace(config.grug_trainer, trainer=trainer)
    run_config = GrugRunConfig(
        model=config.model,
        data=data,
        resources=config.resources,
        optimizer=config.optimizer,
        trainer=grug_trainer,
        eval=eval_cfg,
        k_schedule=config.k_schedule,
    )
    _run_grug_local(run_config)

    # jax.profiler writes to worker-local logs/<run_id>/profiler (ephemeral). Upload it to the run's
    # output_path (R2/cwobject) via the SAME fsspec/s3fs stack that writes checkpoints, so the
    # xplane.pb + trace.json.gz survive the worker. Primary process only.
    if config.profiler.is_enabled:
        import glob

        import fsspec
        import jax

        if jax.process_index() == 0:
            local_root = os.path.join("logs", config.run_id, "profiler")
            dst = config.output_path.rstrip("/") + "/profiler"
            n = 0
            for lpath in glob.glob(os.path.join(local_root, "**"), recursive=True):
                if not os.path.isfile(lpath):
                    continue
                rpath = dst + "/" + os.path.relpath(lpath, local_root)
                fs, _ = fsspec.core.url_to_fs(rpath)
                with open(lpath, "rb") as f, fs.open(rpath, "wb") as g:
                    g.write(f.read())
                n += 1
            print(f"[profiler] uploaded {n} files {local_root} -> {dst}", flush=True)


def _env(key: str, default: str) -> str:
    raw = os.environ.get(key, "")
    return raw if raw else default


def _make_step() -> ExecutorStep:
    mode = _env("SPARSITY_MODE", "fixed")
    hidden = int(_env("SP_HIDDEN", "512"))
    num_experts = int(_env("SP_EXPERTS", "1024"))
    top_k = int(_env("SP_TOPK", "4"))
    min_k = int(_env("SP_MIN_K", "0"))
    coef = float(_env("SP_COEF", "0.0"))
    temp = float(_env("SP_TEMP", "1.0"))
    ref_top_k = int(_env("SP_REF_TOPK", "4"))
    target_tokens = _env("SP_TOKENS", "")
    # SP_FIT_BATCH: pin the per-step batch (memory cap) while keeping the LR/step schedule
    # self-consistent. The big deeply-sparse MoE OOMs above ~64 seqs/step (the ring/a2a
    # dispatch + triton ragged_dot grow ~linearly with tokens/step), but the heuristic's
    # default policy fixes steps (2^14) and *grows* batch with the budget -> it would pick a
    # batch that doesn't fit for a 24h budget. Instead of overriding SP_BATCH after the fact
    # (which desyncs the AdamH LR, computed from the heuristic's own batch/steps), we derive
    # target_steps = tokens / (fit_batch * seq) and feed it to the heuristic so it *itself*
    # picks batch=fit_batch with a matching step count and a consistent LR. Use this (not
    # SP_BATCH) for real runs; SP_BATCH stays a raw override for short fit/throughput smokes.
    fit_batch = _env("SP_FIT_BATCH", "")
    seed = int(_env("SP_SEED", "0"))
    tpu = _env("SP_TPU", "v6e-16")
    group = _env("SP_GROUP", "adaptive-sparsity")
    data_kind = _env("SP_DATA", "datakit")
    data_region = _env("SP_DATA_REGION", "us-east5")
    smoke = _env("SP_SMOKE", "0") == "1"
    # Factorized embedding ("reduce the CE dimension"): the token table + LM head live at
    # this narrow d_e, with small up/down projections to the model dim D. Keeps vocab 128256
    # (no retokenize). Empty -> d_e = D (no factorization). Sizing/LR are unaffected: the
    # heuristic LR depends on D/tokens/batch, and the iso-token budget excludes the LM head.
    embed_env = _env("SP_EMBED", "")
    embed_dim = int(embed_env) if embed_env else None
    seq_len = int(_env("SP_SEQ", "4096"))
    # Device mesh geometry on the 8xH100 node: data = num_devices / (EP * TP).
    expert_axis = int(_env("SP_EP", "1"))
    model_axis = int(_env("SP_TP", "1"))
    replica_env = _env("SP_REPLICA", "")
    replica_axis = int(replica_env) if replica_env else None
    # Performance-tuning knobs (default to the production behavior; only the perf
    # scaffold flips these). SP_PROFILE turns on the levanter xprof callback;
    # SP_REMAT / SP_MOE_IMPL / SP_LOG_EVERY expose the throughput levers as env.
    profile = _env("SP_PROFILE", "0") == "1"
    remat_mode = _env("SP_REMAT", "recompute_all")
    moe_impl = _env("SP_MOE_IMPL", "") or None
    log_every = int(_env("SP_LOG_EVERY", "1"))
    fast_qb = _env("SP_FAST_QB", "0") == "1"
    # Attention backend. On GPU the model default (None) falls to reference_attention,
    # which materializes the full [B,H,S,S] scores (slow + per-device OOM at seq>=2048).
    # Default to the FlashAttention-4 CuTe backend so the big-seq MoE fits and is fast;
    # set SP_ATTN=reference to force the O(S^2) path (debug only), or "" to use the model
    # default. gpu_fa4_thd is the packed/varlen variant.
    attn_impl = _env("SP_ATTN", "gpu_fa4_cute") or None
    # Expert intermediate width override. The heuristic hardcodes I_expert = D/2 (thin
    # experts, so total params scale with a large E); set this to make experts "fat" (e.g.
    # 2*D) for a geometry where the routed experts are a large FLOP fraction and sparsity is
    # a real compute lever. 0 keeps the heuristic default. Sizing/LR are unaffected (LR
    # depends on D/tokens/batch, not I); iso-token budgeting uses the reference (heuristic-I)
    # model, so the token count is unchanged and only wall-clock FLOPs grow.
    intermediate = int(_env("SP_INTERMEDIATE", "0"))
    # Global/local attention interleave (a top attention-throughput lever): SP_GLOBAL_EVERY = the
    # period N so every Nth layer is GLOBAL (full sliding_window) and the rest are LOCAL. 4 -> the
    # legacy 3 local:1 global; 6 -> 5:1; 8 -> 7:1. SP_LOCAL_WINDOW sets the local-layer window
    # (0 -> sliding_window//2). Smaller window + larger period = much cheaper attention.
    global_every = int(_env("SP_GLOBAL_EVERY", "0"))
    local_window = int(_env("SP_LOCAL_WINDOW", "0"))

    # LR-schedule overrides (for WSD pretrain->cooldown phasing). The heuristic fixes the LR
    # *magnitude* (from D/tokens/batch) and a default warmup=0.1 + linear-decay-to-0 shape.
    # For a WSD speedrun we want the pretrain phase at a STABLE (constant) high LR so a later
    # SFT cooldown can anneal it down -- a pretrain that decays to 0 leaves nothing to cool.
    #   SP_SCHEDULE  : "constant" (stable pretrain) | "linear" | "cosine" | "inv_sqrt" (default: heuristic's "linear")
    #   SP_WARMUP    : warmup fraction (default: heuristic 0.1; use ~0.02-0.05 for long runs)
    #   SP_DECAY     : decay-phase fraction at the END (None=decay over all-after-warmup; 0=no decay)
    #   SP_MIN_LR    : min_lr_ratio (final LR as a fraction of peak; default heuristic 0.0)
    #   SP_REWARMUP  : rewarmup fraction (for a cooldown run that re-warms from a resumed ckpt)
    # Each is "" -> leave the heuristic value untouched.
    sched = _env("SP_SCHEDULE", "")
    warmup_env = _env("SP_WARMUP", "")
    decay_env = _env("SP_DECAY", "")
    min_lr_env = _env("SP_MIN_LR", "")
    rewarmup_env = _env("SP_REWARMUP", "")
    # Weights-only init from a prior run's checkpoint DIR (e.g. the pretrain's .../checkpoints) for
    # the SFT cooldown. train.py grafts model params only (step=0, fresh optimizer) on first launch.
    init_from = _env("SP_INIT_FROM", "") or None

    # Active-expert curriculum: SP_CURRICULUM="k0:frac0,k1:frac1,..." ramps the routed
    # expert width k over training (cheap exposure at small k, then cash in expert
    # capacity at the end). Forces fixed routing; the model starts at the first phase's
    # k and is widened in place at each boundary (see train._swap_active_k).
    curriculum = _env("SP_CURRICULUM", "")
    curric_ks: list[int] = []
    curric_fracs: list[float] = []
    if curriculum:
        for part in curriculum.split(","):
            k_str, f_str = part.split(":")
            curric_ks.append(int(k_str))
            curric_fracs.append(float(f_str))
        mode = "fixed"
        top_k = curric_ks[0]

    if mode not in ("fixed", "adaptive"):
        raise ValueError(f"SPARSITY_MODE must be 'fixed' or 'adaptive', got {mode!r}")
    adaptive = mode == "adaptive"

    # Token budget + AdamH LR are fixed once from a reference geometry (this expert pool at
    # SP_REF_TOPK) and applied identically to every arm (iso-token), so the only variable
    # across arms is routing sparsity. When SP_TOKENS is given we hit that exact token count
    # by converting it to a compute budget through the reference FLOPs/token (the heuristic
    # then derives the same token count back, sizing batch/steps/LR consistently). The LR
    # itself depends only on hidden dim / tokens / batch — not on K or E — so it is identical
    # across arms regardless of the reference K.
    _ref_model = MoeAdamHHeuristic().build_model_config(
        hidden, seq_len=seq_len, num_experts=num_experts, num_experts_per_token=ref_top_k, embed_dim=embed_dim
    )
    if target_tokens:
        budget = 3.0 * compute_flops_per_token(_ref_model) * float(target_tokens)
    else:
        budget = float(_env("SP_BUDGET", "1.7e18"))
    # When SP_FIT_BATCH is set, pin the per-step batch to exactly that value (the
    # memory-fitting cap) and let the heuristic derive steps + a consistent LR from it.
    fit_batch_size = None
    if fit_batch:
        if not target_tokens:
            raise ValueError("SP_FIT_BATCH requires SP_TOKENS (need a token budget to size the step count)")
        fit_batch_size = int(fit_batch)
    _ref_model2, optimizer, ref_batch, ref_steps = build_from_heuristic(
        budget=budget,
        hidden_dim=hidden,
        seq_len=seq_len,
        batch_size=fit_batch_size,
        model_overrides=dict(num_experts=num_experts, num_experts_per_token=ref_top_k, embed_dim=embed_dim),
    )
    # Apply LR-schedule overrides for WSD phasing (magnitude stays from the heuristic).
    _sched_over: dict = {}
    if sched:
        _sched_over["lr_schedule"] = sched
    if warmup_env != "":
        _sched_over["warmup"] = float(warmup_env)
    if decay_env != "":
        _sched_over["decay"] = float(decay_env)
    if min_lr_env != "":
        _sched_over["min_lr_ratio"] = float(min_lr_env)
    if rewarmup_env != "":
        _sched_over["rewarmup"] = float(rewarmup_env)
    if _sched_over:
        optimizer = dataclasses.replace(optimizer, **_sched_over)
    # This arm's model carries its own expert geometry / routing mode.
    model = MoeAdamHHeuristic().build_model_config(
        hidden,
        seq_len=seq_len,
        num_experts=num_experts,
        num_experts_per_token=top_k,
        adaptive_routing=adaptive,
        min_experts_per_token=min_k,
        sparsity_loss_coef=coef if adaptive else 0.0,
        sparsity_temp=temp,
        embed_dim=embed_dim,
    )
    # Apply the perf-tuning overrides (remat policy + MoE dispatch backend + fast QB β) and
    # the optional fat-expert geometry override (intermediate width).
    overrides = dict(
        remat_mode=remat_mode,
        moe_implementation=moe_impl,
        fast_qb_beta=fast_qb,
        attention_implementation=attn_impl,
    )
    if intermediate > 0:
        overrides["intermediate_dim"] = intermediate
    if global_every > 0:
        overrides["global_attn_every"] = global_every
    if local_window > 0:
        overrides["local_window"] = local_window
    model = dataclasses.replace(model, **overrides)
    batch = int(_env("SP_BATCH", str(ref_batch)))
    steps = int(_env("SP_STEPS", str(ref_steps)))
    # Capture a few STEADY-STATE steps (compile is step 0; SP_SYNTH_DATA has no loader warmup, so
    # step ~20 is firmly steady). Small window -> small xplane/trace, fast to load. Env-tunable.
    prof_start = int(_env("SP_PROF_START", "20"))
    prof_steps = int(_env("SP_PROF_STEPS", "3"))
    profiler = (
        ProfilerConfig(enabled=True, start_step=prof_start, num_steps=prof_steps)
        if profile
        else ProfilerConfig()
    )

    # Build the curriculum (active_k, end_step) phases from the cumulative token
    # fractions, with the final phase ending exactly at the total step count.
    k_schedule = None
    if curriculum:
        ends: list[int] = []
        acc = 0.0
        for i, frac in enumerate(curric_fracs):
            acc += frac
            ends.append(steps if i == len(curric_fracs) - 1 else max(1, round(acc * steps)))
        k_schedule = tuple((k, e) for k, e in zip(curric_ks, ends))

    active_frac = top_k / num_experts
    if curriculum:
        arm = f"curric-d{hidden}-E{num_experts}-k{curric_ks[0]}to{curric_ks[-1]}"
    elif adaptive:
        arm = f"adapt-d{hidden}-E{num_experts}-k{top_k}-min{min_k}-c{coef:g}-t{temp:g}"
    else:
        arm = f"fixed-d{hidden}-E{num_experts}-k{top_k}"
    if intermediate > 0:
        arm = f"{arm}-I{intermediate}"
    if embed_dim is not None:
        arm = f"{arm}-de{embed_dim}"
    run_id = f"sparsity-{arm}-s{seed}-st{steps}"
    # The marin executor is content-addressed: an identical run_id -> identical step name +
    # output_path -> the step is SKIPPED ("already succeeded") and reuses the cached artifact.
    # That's correct for resumable production runs, but for BENCHMARK sweeps it silently returns
    # stale results (no fresh training, no [THRUPUT]). Set SP_TAG=<sweep-id> to salt the run_id and
    # force a fresh run/output_path. Empty default -> historical resumable behavior is unchanged.
    _tag = _env("SP_TAG", "")
    if _tag:
        run_id = f"{run_id}-{_tag}"

    if data_kind == "datakit":
        # The real CoreWeave pretraining data: nemotron datakit flat parquet on R2
        # (llama3 vocab 128256). smoke -> a single high-quality split for a fast path check.
        data = build_nemotron_datakit_mix(smoke=smoke)
    elif data_kind == "datakit_eval":
        # Same caches as datakit but flat_cache=True at /train so SP_EVAL's num_validation_sequences
        # can slice a held-out eval split (no separate /validation cache exists). Eval-only path.
        data = build_nemotron_datakit_eval_mix(smoke=smoke)
    elif data_kind == "cw":
        # SAME caches as `datakit`, but read from the CoreWeave cluster-local cwobject mirror
        # (much faster than R2; no cross-stream contention -> enables concurrent runs). Requires
        # the cwobject env (AWS_*=CW creds, AWS_ENDPOINT_URL, LEVANTER_S3_VIRTUAL_HOSTED=1) and a
        # completed mirror. SP_CW_COMPONENTS (comma list) limits to mirrored components.
        only = [c.strip() for c in _env("SP_CW_COMPONENTS", "").split(",") if c.strip()]
        data = build_nemotron_cw_mix(smoke=smoke, only=only or None)
    elif data_kind == "sft":
        # SFT cooldown mixture (chat, assistant-only loss): tulu-3 + smoltalk + OpenThoughts-Agent,
        # tokenized inline from HF on the worker -> R2. Pair with SP_INIT_FROM (resume the pretrain
        # checkpoint) + SP_SCHEDULE=linear SP_MIN_LR=0 (decay the WSD cooldown).
        from data import build_sft_mix

        data = build_sft_mix()
    elif data_kind == "fineweb":
        data = build_fineweb_edu_mix(smoke=smoke)
    elif data_kind == "nemotron":
        # gs:// caches -- NOT readable on CoreWeave (R2-only); kept for TPU/GCP parity.
        data = build_nemotron_mix(region=data_region)
    else:
        raise ValueError(f"SP_DATA must be 'datakit', 'cw', 'sft', 'fineweb', or 'nemotron', got {data_kind!r}")

    launch = GrugMoeLaunchConfig(
        model=versioned(model),
        data=data,
        output_path=this_output_path(),
        run_id=run_id,
        resources=versioned(ResourceConfig.with_tpu(tpu)),
        steps=versioned(steps),
        batch_size=versioned(batch),
        seed=versioned(seed),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        checkpointer=None,
        profiler=profiler,
        tracker=WandbConfig(
            project="marin_moe",
            tags=["moe", "adaptive-sparsity", mode, f"E{num_experts}", f"k{top_k}"]
            + (["curriculum"] if curriculum else []),
            group=group,
            name=None,
        ),
        optimizer=versioned(optimizer),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=1e-4,
                ema_beta=None,
                log_every=log_every,
                expert_axis_size=expert_axis,
                model_axis_size=model_axis,
                replica_axis_size=replica_axis,
            )
        ),
        k_schedule=k_schedule,
        init_from=init_from,
    )

    tokens = batch * steps * seq_len
    print(
        f"[arm] {run_id}  mode={mode} D={hidden} d_e={model.inferred_embed_dim}"
        f"(factorized={model.is_factorized_embed}) seq={seq_len} E={num_experts} k={top_k} min_k={min_k} "
        f"coef={coef} nominal_active_frac={active_frac:.4%} batch={batch} steps={steps} "
        f"tokens={tokens/1e9:.1f}B layers={model.num_layers} I_expert={model.intermediate_dim} "
        f"mesh(EP={expert_axis},TP={model_axis},replica={replica_axis}) "
        f"data={data_kind} budget={budget:g} remat={remat_mode} moe_impl={moe_impl} attn={attn_impl} "
        f"sched={sched or 'heuristic'} warmup={warmup_env or 'def'} min_lr={min_lr_env or 'def'} init_from={init_from} "
        f"log_every={log_every} profile={profile} curriculum={k_schedule}"
    )
    return ExecutorStep(name=f"grug/sparsity/{run_id}", fn=run_inline, config=launch)


if __name__ == "__main__":
    # Async pipeline-parallel (PP) training path.  When SP_PP_MODE=async the standard
    # FSDP+EP training is bypassed: train_pp.run_pp_async() builds the per-stage
    # submeshes, runs the async no-flush schedule, emits [PP_THRUPUT] lines, and exits.
    # This avoids the EP all-to-all (29%) + FSDP all-gather (15%) overhead by keeping
    # all experts and params local to each stage's device.
    if os.environ.get("SP_PP_MODE", "") == "async":
        import sys
        from train_pp import run_pp_async
        run_pp_async()
        sys.exit(0)

    # SYNC microbatched (gradient-exact GPipe) pipeline parallelism -- the CORRECT
    # overlapping PP (see sync_pipeline.py / pp_overlap_probe.py). Measures overlap
    # factor + tok/s + MFU vs the EP+FSDP anchor.
    if os.environ.get("SP_PP_MODE", "") == "sync":
        import sys
        from train_pp import run_pp_sync
        run_pp_sync()
        sys.exit(0)

    # GPU overlap probe: bare-matmul micro-probes that decide whether eager per-stage
    # dispatch + cross-device device_put + set_mesh can OVERLAP the 8 GPUs when driven
    # the correct (microbatch-major) way -- the prerequisite for any real PP speedup.
    if os.environ.get("SP_PP_MODE", "") == "probe":
        import sys
        import pp_overlap_probe
        pp_overlap_probe.main()
        sys.exit(0)

    # Fast worker-side diagnostics that reuse the normal job bundle but skip the
    # full training pipeline (e.g. SP_DIAG=ragged probes the MoE grouped-matmul
    # backend that drives the 278 GiB OOM).
    _diag = os.environ.get("SP_DIAG", "").strip()
    if _diag in ("ragged", "ring", "fit", "fa4"):
        import sys

        import _diag_ragged

        if _diag == "ring":
            _diag_ragged.ring()
        elif _diag == "fit":
            _diag_ragged.fit()
        elif _diag == "fa4":
            _diag_ragged.fa4()
        else:
            _diag_ragged.main()
        sys.exit(0)
    executor_main(
        steps=[_make_step()],
        description="Grug MoE adaptive-sparsity sweep.",
    )
