# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Re-entrant (self-looping) grug MoE experiment series, as a standalone template.

A copy-first variant of the ``delayed-gradient-pp`` template used to prototype
recurrent-depth / looped transformers: a weight-tied core block applied R times at
inference instead of stacking R independent layers ("thinking in activation space"
as an alternative to thinking *tokens*). Model, train loop, eval sweep, and launch
wiring are all self-contained in this directory so each experiment (E0 baseline,
E1 re-entrant, ...) can be iterated independently. See ``REPORT.md`` for the full
write-up and ``README.md`` for the layout and arms.

The experiment is selected by the ``GRUG_EXPERIMENT`` env var (comma-separated names
from the registry below; default ``e0``). Submit directly on a TPU (training runs
in-process on the job that holds the accelerator when ``GRUG_DIRECT`` is set):

    GRUG_EXPERIMENT=e1 GRUG_DIRECT=1 MARIN_PREFIX=gs://marin-us-central1 \
      uv run iris --cluster=marin job run --no-wait \
        --tpu v5p-8 --region us-central1 --enable-extra-resources --extra marin-core:tpu \
        --cpu 32 --memory 128GB --disk 50GB \
        -e WANDB_API_KEY "$WANDB_API_KEY" -e GRUG_DIRECT 1 -- python launch.py
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

from data import build_nemotron_mix_with_validation
from heuristic import build_from_heuristic
from model import GrugModelConfig
from train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug


@dataclass(frozen=True)
class GrugMoeLaunchConfig:
    """Last-mile run config for the MoE grug template.

    Keep this as the main entry point for day-to-day edits (model/data/optimizer/trainer/eval knobs).
    """

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
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)
    checkpointer: CheckpointerConfig | None = None
    """Override the checkpointer. None builds the default (periodic + final saves
    under output_path)."""


# Training mixture + Paloma/uncheatable validation, pinned to existing GCS caches.
# Built once at import time (the eval sweep imports this same object).
NEMOTRON_MIX_WITH_DEFAULT_VALIDATION = build_nemotron_mix_with_validation()


def env_int(key: str, default: int) -> int:
    """Read an int from ``os.environ[key]``, falling back to ``default`` when unset/empty."""
    raw = os.environ.get(key, "")
    return int(raw) if raw else default


def env_float(key: str, default: float) -> float:
    """Read a float from ``os.environ[key]``, falling back to ``default`` when unset/empty."""
    raw = os.environ.get(key, "")
    return float(raw) if raw else default


def _resolve_run_id(default_run_id: str) -> str:
    """Resolve run id, allowing an explicit ``GRUG_RUN_ID`` override."""
    return os.environ.get("GRUG_RUN_ID", default_run_id)


def _resolve_tracker(tracker: TrackerConfig, run_id: str) -> TrackerConfig:
    if isinstance(tracker, WandbConfig):
        return dataclasses.replace(tracker, name=run_id)
    return tracker


def run_grug_moe_trial(config: GrugMoeLaunchConfig) -> None:
    # Map template launch knobs onto a full Levanter TrainerConfig.
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
            save_interval=timedelta(minutes=10),
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
        eval=config.eval,
    )
    run_grug(run_config)


# Re-entrant experiment series. All runs share one compute budget / model size
# (the d512 ~130M-class MoE point) so re-entrant variants are directly comparable
# to the E0 baseline. Only the model architecture changes between experiments.
_BUDGET: float = 2.19e17
_HIDDEN_DIM: int = 512
_TARGET_STEPS: int = 2**14
_baseline_model, _baseline_optimizer, _baseline_batch, _baseline_steps = build_from_heuristic(
    budget=_BUDGET,
    hidden_dim=_HIDDEN_DIM,
    target_steps=_TARGET_STEPS,
)

# Public alias for the heuristic-derived baseline GrugModelConfig.
GRUG_MOE_TRIAL_MODEL: GrugModelConfig = _baseline_model

# v5p-8 in us-central1: region-local to the gs://marin-us-central1 data and
# checkpoint bucket and to the cluster controller. This matches the README d512
# baseline hardware (the v5p pool's smallest slice is 8 chips).
_REENTRANT_RESOURCES = ResourceConfig.with_tpu("v5p-8", regions=["us-central1"])


def reentrant_step(*, name: str, run_id: str, model: GrugModelConfig, tags: list[str]) -> ExecutorStep:
    """Build an ExecutorStep for one re-entrant experiment.

    Every experiment shares optimizer / batch / steps / data / trainer / eval with
    the E0 baseline; only ``model`` (and the run metadata) changes, so curves are
    directly comparable.
    """
    return ExecutorStep(
        name=name,
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(_REENTRANT_RESOURCES),
            steps=versioned(_baseline_steps),
            batch_size=versioned(_baseline_batch),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(project="marin_moe", tags=tags, group="reentrant", name=None),
            optimizer=versioned(_baseline_optimizer),
            grug_trainer=versioned(GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1)),
            eval=versioned(
                GrugEvalConfig(
                    eval_batch_size=512,
                    steps_per_eval=1000,
                    max_eval_batches=8,
                    eval_current=True,
                    eval_ema=False,
                )
            ),
        ),
    )


# E0 — baseline: unchanged d512 Grug MoE. Reference curve for all re-entrant variants.
e0_baseline = reentrant_step(
    name="grug/reentrant_e0_d512",
    run_id=_resolve_run_id("reentrant_e0_d512"),
    model=_baseline_model,
    tags=["moe", "reentrant", "e0-baseline"],
)

# E1 — basic re-entrant: 1 prelude + 1 weight-tied core looped 4x + 1 coda.
# Effective depth 6 (compute-matched to E0) with 3 unique blocks (~half the block
# params). Tests Saunshi's "looped k-layer ~= kL-layer" at fixed compute.
_E1_MODEL = dataclasses.replace(
    _baseline_model,
    num_layers=3,
    num_prelude_layers=1,
    num_coda_layers=1,
    recurrence_steps=4,
)
e1_reentrant = reentrant_step(
    name="grug/reentrant_e1_loop4",
    run_id=_resolve_run_id("reentrant_e1_loop4"),
    model=_E1_MODEL,
    tags=["moe", "reentrant", "e1-loop4"],
)

# E2 — iteration-conditioned re-entrant: E1 + per-iteration FiLM (adaLN) on the
# shared core block. Tests whether telling the looped block which step it is on
# (coarse-to-fine) helps, at ~free parameter cost. Identity at init == E1.
_E2_MODEL = dataclasses.replace(_E1_MODEL, iteration_film=True)
e2_reentrant = reentrant_step(
    name="grug/reentrant_e2_filmloop4",
    run_id=_resolve_run_id("reentrant_e2_filmloop4"),
    model=_E2_MODEL,
    tags=["moe", "reentrant", "e2-film-loop4"],
)

# E3 — randomized-depth re-entrant: E1 with the core loop count sampled per step
# from {2,4,8} during training. Trains one weight-tied core to be correct at many
# depths so the SAME checkpoint can be evaluated at higher loop counts at test time
# (the depth-scaling experiment). recurrence_steps=4 stays the default/eval depth.
_E3_MODEL = dataclasses.replace(
    _E1_MODEL,
    randomize_recurrence=True,
    recurrence_choices=(2, 4, 8),
)
e3_reentrant = reentrant_step(
    name="grug/reentrant_e3_randdepth",
    run_id=_resolve_run_id("reentrant_e3_randdepth"),
    model=_E3_MODEL,
    tags=["moe", "reentrant", "e3-randdepth"],
)

# E5 — convergence-regularized re-entrant: E3 plus a training-only core-consistency
# penalty (mean normalized squared delta between consecutive core-loop states). Pulls
# the weight-tied core toward a contractive/fixed-point map so deeper-than-trained
# test-time loops stop drifting. CONSISTENCY_WEIGHT tunes lambda; eval metric is
# unchanged (penalty is training-only).
_E5_MODEL = dataclasses.replace(_E3_MODEL, core_consistency_weight=env_float("CONSISTENCY_WEIGHT", 1.0))
e5_reentrant = reentrant_step(
    name="grug/reentrant_e5_consistency",
    run_id=_resolve_run_id("reentrant_e5_consistency"),
    model=_E5_MODEL,
    tags=["moe", "reentrant", "e5-consistency"],
)

# E6 — depth-conditioned MoE routing: E3 plus a learned per-(iteration, core-layer)
# additive router-logit bias, so each core traversal activates a DIFFERENT expert
# mixture. Directly targets the E0-E5 structural failure (a single residual core
# re-applied in place can't both keep computing and converge): depth-varying expert
# selection makes each loop a genuinely different transformation while staying
# weight-tied. Zero-init bias => numerically identical to E3 at step 0.
_E6_MODEL = dataclasses.replace(_E3_MODEL, depth_conditioned_routing=True)
e6_reentrant = reentrant_step(
    name="grug/reentrant_e6_depthroute",
    run_id=_resolve_run_id("reentrant_e6_depthroute"),
    model=_E6_MODEL,
    tags=["moe", "reentrant", "e6-depthroute"],
)

# E4 — anytime deep-supervision: E3 plus a training-only CE term read off the shared
# output head after EACH core iteration (averaged). Rewards USEFUL refinement at
# every depth (the direct fix for E5's freeze, which suppressed refinement) and makes
# every depth anytime-decodable. ANYTIME_WEIGHT tunes the term. No new params, so the
# checkpoint is depth-sweepable with the E3 model config.
_E4_MODEL = dataclasses.replace(
    _E3_MODEL,
    anytime_supervision=True,
    anytime_supervision_weight=env_float("ANYTIME_WEIGHT", 1.0),
)
e4_reentrant = reentrant_step(
    name="grug/reentrant_e4_anytime",
    run_id=_resolve_run_id("reentrant_e4_anytime"),
    model=_E4_MODEL,
    tags=["moe", "reentrant", "e4-anytime"],
)

# E64 — combined: depth-conditioned routing (E6) AND anytime deep-supervision (E4).
# Tests whether genuinely depth-varying computation plus a per-depth usefulness
# signal together beat each alone.
_E64_MODEL = dataclasses.replace(
    _E6_MODEL,
    anytime_supervision=True,
    anytime_supervision_weight=env_float("ANYTIME_WEIGHT", 1.0),
)
e64_reentrant = reentrant_step(
    name="grug/reentrant_e64_depthroute_anytime",
    run_id=_resolve_run_id("reentrant_e64_depthroute_anytime"),
    model=_E64_MODEL,
    tags=["moe", "reentrant", "e64-combined"],
)

# E7 — PonderNet: E3 plus a learned per-token halting head + expected-over-halting CE
# and a KL-to-geometric-prior regularizer (added on top of the standard final CE, so
# the eval metric stays comparable). Adaptive test-time compute. PONDER_KL / PONDER_PRIOR
# env knobs. No effect on the eval path; checkpoint depth-sweepable with the E3 config.
_E7_MODEL = dataclasses.replace(
    _E3_MODEL,
    ponder_halting=True,
    ponder_kl_weight=env_float("PONDER_KL", 0.01),
    ponder_prior_lambda=env_float("PONDER_PRIOR", 0.2),
)
e7_reentrant = reentrant_step(
    name="grug/reentrant_e7_ponder",
    run_id=_resolve_run_id("reentrant_e7_ponder"),
    model=_E7_MODEL,
    tags=["moe", "reentrant", "e7-ponder"],
)

# Experiment registry. Select with GRUG_EXPERIMENT (comma-separated names); default E0.
_STEPS = {
    "e0": e0_baseline,
    "e1": e1_reentrant,
    "e2": e2_reentrant,
    "e3": e3_reentrant,
    "e5": e5_reentrant,
    "e6": e6_reentrant,
    "e4": e4_reentrant,
    "e64": e64_reentrant,
    "e7": e7_reentrant,
}


if __name__ == "__main__":
    _selected = os.environ.get("GRUG_EXPERIMENT", "e0").split(",")
    executor_main(
        steps=[_STEPS[name.strip()] for name in _selected],
        description="Re-entrant grug MoE experiments (d512).",
    )
