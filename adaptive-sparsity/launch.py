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
    SP_HIDDEN       model hidden dim               (default 768  -> ~1.1B total at E=128)
    SP_EXPERTS      number of experts E            (default 128)
    SP_TOPK         top-k routing width K          (default 4; the K_max capacity in adaptive mode)
    SP_MIN_K        adaptive per-token floor        (default 0; 0 lets a token use the shared expert alone)
    SP_COEF         sparsity penalty weight λ       (default 0.0; only meaningful when SPARSITY_MODE=adaptive)
    SP_TEMP         soft keep-gate temperature      (default 1.0)
    SP_BUDGET       compute budget for sizing/LR    (default 1.7e18 -> d768 row, ~2.7B tokens)
    SP_STEPS        override train steps            (default: heuristic value for the budget)
    SP_BATCH        override batch size (sequences) (default: heuristic value for the budget)
    SP_SEED         seed                            (default 0)
    SP_TPU          TPU type                        (default v6e-16)
    SP_GROUP        wandb group                     (default adaptive-sparsity)
    SP_SMOKE        1 -> use the 10M-token FineWeb-Edu subset for a cluster smoke check (default 0)

Submit one arm directly on a TPU (training runs in-process on the job that holds the
TPU). Omit ``--region`` so iris can take v6e capacity in any region; the FineWeb-Edu
cache is region-agnostic (HF-backed), so it materializes in whatever region you land:

    MARIN_PREFIX=gs://marin-us-east5 \
    SPARSITY_MODE=fixed SP_EXPERTS=128 SP_TOPK=4 \
      uv run iris --cluster=marin job run --no-wait \
        --tpu v6e-16 --enable-extra-resources --extra marin-core:tpu \
        --max-retries 3 --cpu 32 --memory 128GB --disk 50GB \
        -e WANDB_API_KEY "$WANDB_API_KEY" -- python launch.py
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

from data import build_fineweb_edu_mix
from heuristic import MoeAdamHHeuristic, build_from_heuristic
from model import GrugModelConfig
from train import GrugRunConfig, GrugTrainerConfig, _run_grug_local

# Reference expert geometry used to fix the token budget and learning rate across all
# arms. The AdamH LR depends only on hidden dim / tokens, and we want every arm to
# train on the *same* token budget (iso-token) so the only variable is sparsity, so
# we size batch / steps / optimizer once from this reference rather than per-arm
# (the per-arm flops-per-token, which depends on K, would otherwise shift the budget).
_REF_EXPERTS = 128
_REF_TOP_K = 4


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
            save_interval=timedelta(hours=24),
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
    )
    _run_grug_local(run_config)


def _env(key: str, default: str) -> str:
    raw = os.environ.get(key, "")
    return raw if raw else default


def _make_step() -> ExecutorStep:
    mode = _env("SPARSITY_MODE", "fixed")
    hidden = int(_env("SP_HIDDEN", "768"))
    num_experts = int(_env("SP_EXPERTS", "128"))
    top_k = int(_env("SP_TOPK", "4"))
    min_k = int(_env("SP_MIN_K", "0"))
    coef = float(_env("SP_COEF", "0.0"))
    temp = float(_env("SP_TEMP", "1.0"))
    budget = float(_env("SP_BUDGET", "1.7e18"))
    seed = int(_env("SP_SEED", "0"))
    tpu = _env("SP_TPU", "v6e-16")
    group = _env("SP_GROUP", "adaptive-sparsity")
    smoke = _env("SP_SMOKE", "0") == "1"

    if mode not in ("fixed", "adaptive"):
        raise ValueError(f"SPARSITY_MODE must be 'fixed' or 'adaptive', got {mode!r}")
    adaptive = mode == "adaptive"

    # Fixed reference: token budget + AdamH LR, identical for every arm (iso-token).
    _ref_model, optimizer, ref_batch, ref_steps = build_from_heuristic(
        budget=budget,
        hidden_dim=hidden,
        model_overrides=dict(num_experts=_REF_EXPERTS, num_experts_per_token=_REF_TOP_K),
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
    batch = int(_env("SP_BATCH", str(ref_batch)))
    steps = int(_env("SP_STEPS", str(ref_steps)))

    active_frac = top_k / num_experts
    if adaptive:
        arm = f"adapt-d{hidden}-E{num_experts}-k{top_k}-min{min_k}-c{coef:g}-t{temp:g}"
    else:
        arm = f"fixed-d{hidden}-E{num_experts}-k{top_k}"
    run_id = f"sparsity-{arm}-s{seed}-st{steps}"

    launch = GrugMoeLaunchConfig(
        model=versioned(model),
        data=build_fineweb_edu_mix(smoke=smoke),
        output_path=this_output_path(),
        run_id=run_id,
        resources=versioned(ResourceConfig.with_tpu(tpu)),
        steps=versioned(steps),
        batch_size=versioned(batch),
        seed=versioned(seed),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        checkpointer=None,
        tracker=WandbConfig(
            project="marin_moe",
            tags=["moe", "adaptive-sparsity", mode, f"E{num_experts}", f"k{top_k}"],
            group=group,
            name=None,
        ),
        optimizer=versioned(optimizer),
        grug_trainer=versioned(GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1)),
    )

    print(
        f"[arm] {run_id}  mode={mode} E={num_experts} k={top_k} min_k={min_k} "
        f"coef={coef} nominal_active_frac={active_frac:.4%} batch={batch} steps={steps} "
        f"layers={model.num_layers} budget={budget:g}"
    )
    return ExecutorStep(name=f"grug/sparsity/{run_id}", fn=run_inline, config=launch)


if __name__ == "__main__":
    executor_main(
        steps=[_make_step()],
        description="Grug MoE adaptive-sparsity sweep.",
    )
