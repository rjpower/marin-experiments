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
    SP_DATA         nemotron | fineweb              (default nemotron; the real marin MoE mixture)
    SP_DATA_REGION  GCS region for nemotron caches  (default us-east5; match the run's --region)
    SP_SEED         seed                            (default 0)
    SP_TPU          TPU type                        (default v6e-16)
    SP_GROUP        wandb group                     (default adaptive-sparsity)
    SP_SMOKE        1 -> use the 10M-token FineWeb-Edu subset for a cluster smoke check (default 0)

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

import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LmDataConfig
from levanter.optim import OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned
from marin.training.training import temporary_checkpoint_base_path

from data import build_fineweb_edu_mix, build_nemotron_mix
from heuristic import MoeAdamHHeuristic, build_from_heuristic, compute_flops_per_token
from model import GrugModelConfig
from train import GrugRunConfig, GrugTrainerConfig, _run_grug_local


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
    grug_trainer = dataclasses.replace(config.grug_trainer, trainer=trainer)
    run_config = GrugRunConfig(
        model=config.model,
        data=config.data,
        resources=config.resources,
        optimizer=config.optimizer,
        trainer=grug_trainer,
        # No validation sets in the FineWeb-Edu mixture; convergence metric is train/loss.
        eval=None,
        k_schedule=config.k_schedule,
    )
    _run_grug_local(run_config)


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
    seed = int(_env("SP_SEED", "0"))
    tpu = _env("SP_TPU", "v6e-16")
    group = _env("SP_GROUP", "adaptive-sparsity")
    data_kind = _env("SP_DATA", "nemotron")
    data_region = _env("SP_DATA_REGION", "us-east5")
    smoke = _env("SP_SMOKE", "0") == "1"
    # Performance-tuning knobs (default to the production behavior; only the perf
    # scaffold flips these). SP_PROFILE turns on the levanter xprof callback;
    # SP_REMAT / SP_MOE_IMPL / SP_LOG_EVERY expose the throughput levers as env.
    profile = _env("SP_PROFILE", "0") == "1"
    remat_mode = _env("SP_REMAT", "recompute_all")
    moe_impl = _env("SP_MOE_IMPL", "") or None
    log_every = int(_env("SP_LOG_EVERY", "1"))
    fast_qb = _env("SP_FAST_QB", "0") == "1"
    # Expert intermediate width override. The heuristic hardcodes I_expert = D/2 (thin
    # experts, so total params scale with a large E); set this to make experts "fat" (e.g.
    # 2*D) for a geometry where the routed experts are a large FLOP fraction and sparsity is
    # a real compute lever. 0 keeps the heuristic default. Sizing/LR are unaffected (LR
    # depends on D/tokens/batch, not I); iso-token budgeting uses the reference (heuristic-I)
    # model, so the token count is unchanged and only wall-clock FLOPs grow.
    intermediate = int(_env("SP_INTERMEDIATE", "0"))

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
        hidden, num_experts=num_experts, num_experts_per_token=ref_top_k
    )
    if target_tokens:
        budget = 3.0 * compute_flops_per_token(_ref_model) * float(target_tokens)
    else:
        budget = float(_env("SP_BUDGET", "1.7e18"))
    _ref_model2, optimizer, ref_batch, ref_steps = build_from_heuristic(
        budget=budget,
        hidden_dim=hidden,
        model_overrides=dict(num_experts=num_experts, num_experts_per_token=ref_top_k),
    )
    # This arm's model carries its own expert geometry / routing mode.
    model = MoeAdamHHeuristic().build_model_config(
        hidden,
        num_experts=num_experts,
        num_experts_per_token=top_k,
        adaptive_routing=adaptive,
        min_experts_per_token=min_k,
        sparsity_loss_coef=coef if adaptive else 0.0,
        sparsity_temp=temp,
    )
    # Apply the perf-tuning overrides (remat policy + MoE dispatch backend + fast QB β) and
    # the optional fat-expert geometry override (intermediate width).
    overrides = dict(remat_mode=remat_mode, moe_implementation=moe_impl, fast_qb_beta=fast_qb)
    if intermediate > 0:
        overrides["intermediate_dim"] = intermediate
    model = dataclasses.replace(model, **overrides)
    batch = int(_env("SP_BATCH", str(ref_batch)))
    steps = int(_env("SP_STEPS", str(ref_steps)))
    profiler = ProfilerConfig(enabled=True, start_step=10, num_steps=10) if profile else ProfilerConfig()

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
    run_id = f"sparsity-{arm}-s{seed}-st{steps}"

    if smoke or data_kind == "fineweb":
        data = build_fineweb_edu_mix(smoke=smoke)
    elif data_kind == "nemotron":
        data = build_nemotron_mix(region=data_region)
    else:
        raise ValueError(f"SP_DATA must be 'nemotron' or 'fineweb', got {data_kind!r}")

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
        grug_trainer=versioned(GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=log_every)),
        k_schedule=k_schedule,
    )

    tokens = batch * steps * 4096
    print(
        f"[arm] {run_id}  mode={mode} E={num_experts} k={top_k} min_k={min_k} "
        f"coef={coef} nominal_active_frac={active_frac:.4%} batch={batch} steps={steps} "
        f"tokens={tokens/1e9:.1f}B layers={model.num_layers} I_expert={model.intermediate_dim} "
        f"data={data_kind} budget={budget:g} remat={remat_mode} moe_impl={moe_impl} "
        f"log_every={log_every} profile={profile} curriculum={k_schedule}"
    )
    return ExecutorStep(name=f"grug/sparsity/{run_id}", fn=run_inline, config=launch)


if __name__ == "__main__":
    executor_main(
        steps=[_make_step()],
        description="Grug MoE adaptive-sparsity sweep.",
    )
