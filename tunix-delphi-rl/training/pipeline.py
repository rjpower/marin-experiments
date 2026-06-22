# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Idempotent runner for the Delphi coding tuning rounds (the obvious entrypoint).

This is the single reproducible driver for the *rounds of tuning* behind
``REPORT.md`` §9 (single-turn coding, issue #7) and §11 (multi-turn coding,
issue #8). It exists so a newcomer can reproduce the experiment arc without
hand-assembling ``train_coding`` / ``train_multiturn`` calls and remembering the
SFT/RL budgets.

What it does
------------
Each tuning ROUND is declared as data (a :class:`Stage`): a name plus the exact
keyword arguments handed to an in-process entrypoint
(:func:`train_coding.train_coding` for ``--experiment coding``,
:func:`train_multiturn.train_multiturn` for ``--experiment multiturn``). Running a
stage:

  1. runs the train function (which itself does SFT warm-up [-> Dr.GRPO] and a
     greedy eval on the fixed ladder, exactly as the iris launchers do), then
  2. writes a small per-stage results JSON under ``--results-dir``.

It is **idempotent**: a stage whose results JSON already exists is SKIPPED, so
re-running resumes the sequence rather than recomputing. Delete a stage's JSON to
force a re-run of just that stage.

The stage tables mirror ``REPORT.md`` so the reproduction is legible:

  * ``coding``     -- the SFT-amount ladder (few-shot -> SFT 150/300/1000) plus
    one SFT -> Dr.GRPO round (the §9.2 table). The crown-jewel finding: SFT scales
    monotonically to ~48-50/50 and Dr.GRPO is MARGINAL (the target -- writing a
    program -- is fully demonstrable by SFT).
  * ``multiturn`` -- few-shot -> SFT-only -> SFT + Dr.GRPO on the harder tiers
    (3,4,5) write->run->revise loop (the §11 hypothesis: where iterating on
    execution feedback should make Dr.GRPO finally beat SFT).

How a newcomer runs it
----------------------
This SUBMITS work that needs a TPU. Do NOT run training from here directly; the
coordinator submits an iris job whose entrypoint is one of the ``launch_*.py``
files (see ``AGENTS.md``). Use this script to:

  * inspect the plan (no TPU, no model download)::

        uv run python -m training.pipeline --dry-run --experiment coding
        uv run python -m training.pipeline --dry-run --experiment multiturn

  * run one stage on a host with a TPU + Delphi weights at ``DELPHI_MODEL_DIR``::

        uv run python -m training.pipeline --experiment coding --stage sft1000
        uv run python -m training.pipeline --experiment multiturn --all

``--dry-run`` works on CPU: it imports nothing that needs a TPU and only prints
the plan (which stages would run vs are already done).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from typing import Any, Callable, Dict, List

# Delphi weights directory on the worker (downloaded by the launchers); the train
# functions load from here. Overridable via the env var the launchers also read.
DEFAULT_MODEL_DIR = os.environ.get("DELPHI_MODEL_DIR", "./delphi")


@dataclasses.dataclass(frozen=True)
class Stage:
  """One tuning round: a name + the kwargs for the experiment's train fn.

  Attributes:
    name: stable identifier; also the results-JSON filename stem and ``--stage``
      selector. Keep stable -- it is the idempotency key.
    description: one-line human summary (shown by ``--dry-run``).
    kwargs: keyword arguments passed verbatim to the experiment's train function
      (``train_coding`` or ``train_multiturn``). ``model_dir`` is injected at run
      time, so it is omitted here.
  """

  name: str
  description: str
  kwargs: Dict[str, Any]


# ---------------------------------------------------------------------------
# coding (single-turn, issue #7 / REPORT.md §9): the SFT-amount ladder + one
# SFT -> Dr.GRPO round. Mirrors the §9.2 results table. eval-on-50 is built into
# train_coding (do_eval defaults True), so every stage reports per-tier solve.
# ---------------------------------------------------------------------------
CODING_STAGES: List[Stage] = [
    Stage(
        name="fewshot",
        description="few-shot only (no SFT, no RL) -- the §9.2 baseline (~3/50)",
        kwargs=dict(sft_steps=0, steps=0),
    ),
    Stage(
        name="sft150",
        description="SFT warm-up 150 transcripts, no RL (~45/50)",
        kwargs=dict(sft_steps=150, steps=0),
    ),
    Stage(
        name="sft300",
        description="SFT warm-up 300 transcripts, no RL (~46/50)",
        kwargs=dict(sft_steps=300, steps=0),
    ),
    Stage(
        name="sft300_drgrpo80",
        description="SFT 300 -> Dr.GRPO 80 -- shows RL is MARGINAL here (~45/50)",
        kwargs=dict(sft_steps=300, steps=80),
    ),
    Stage(
        name="sft1000",
        description="SFT warm-up 1000 transcripts, no RL -- the plateau (~48/50)",
        kwargs=dict(sft_steps=1000, steps=0),
    ),
]

# ---------------------------------------------------------------------------
# multiturn (issue #8 / REPORT.md §11): few-shot -> SFT-only -> SFT + Dr.GRPO on
# the harder tiers (3,4,5) write->run->revise loop. eval reports first-attempt vs
# best-across-rounds solve. No results are claimed yet (the §11 hypothesis).
# ---------------------------------------------------------------------------
MULTITURN_STAGES: List[Stage] = [
    Stage(
        name="fewshot",
        description="few-shot only (no SFT, no RL) -- the multi-turn baseline",
        kwargs=dict(tiers=(3, 4, 5), sft_steps=0, steps=0),
    ),
    Stage(
        name="sft600",
        description="SFT-only warm-up on the multi-turn format (600), no RL",
        kwargs=dict(tiers=(3, 4, 5), sft_steps=600, steps=0),
    ),
    Stage(
        name="sft600_drgrpo120",
        description="SFT 600 -> Dr.GRPO 120 -- the regime RL should finally help",
        kwargs=dict(tiers=(3, 4, 5), sft_steps=600, steps=120),
    ),
]


def _experiment_specs() -> Dict[str, Dict[str, Any]]:
  """Maps experiment name -> {stages, train fn factory, result serializer}.

  The train fn and serializer are returned as zero-arg factories / closures so
  ``--dry-run`` never imports the (TPU-bound) train modules.
  """
  return {
      "coding": {
          "stages": CODING_STAGES,
          "train_fn": _coding_train_fn,
          "serialize": _serialize_coding,
      },
      "multiturn": {
          "stages": MULTITURN_STAGES,
          "train_fn": _multiturn_train_fn,
          "serialize": _serialize_multiturn,
      },
  }


def _coding_train_fn(model_dir: str, kwargs: Dict[str, Any]):
  from training.train_coding import train_coding

  return train_coding(model_dir=model_dir, **kwargs)


def _multiturn_train_fn(model_dir: str, kwargs: Dict[str, Any]):
  from training.train_multiturn import train_multiturn

  return train_multiturn(model_dir=model_dir, **kwargs)


def _eval_to_dict(ev) -> Dict[str, Any] | None:
  """Serializes a CodingEvalResult OR MultiTurnEvalResult to plain dicts."""
  if ev is None:
    return None
  out: Dict[str, Any] = {"total": ev.total, "per_tier": {}}
  per_tier = ev.per_tier()
  # CodingEvalResult.per_tier -> (solved, total); MultiTurnEvalResult -> (first, best, total).
  if any(len(v) == 3 for v in per_tier.values()):
    out["first_solved"] = ev.first_solved
    out["best_solved"] = ev.best_solved
    out["per_tier"] = {str(t): {"first": v[0], "best": v[1], "total": v[2]} for t, v in per_tier.items()}
  else:
    out["solved"] = ev.solved
    out["per_tier"] = {str(t): {"solved": v[0], "total": v[1]} for t, v in per_tier.items()}
  return out


def _serialize_coding(result) -> Dict[str, Any]:
  return {
      "steps_ran": result.steps_ran,
      "reward_history": result.reward_history,
      "solve_ratio_history": result.solve_ratio_history,
      "eval_after_sft": _eval_to_dict(result.eval_after_sft),
      "eval_after_rl": _eval_to_dict(result.eval_after_rl),
  }


def _serialize_multiturn(result) -> Dict[str, Any]:
  return {
      "steps_ran": result.steps_ran,
      "reward_history": result.reward_history,
      "first_solve_history": result.first_solve_history,
      "solve_ratio_history": result.solve_ratio_history,
      "eval_after_sft": _eval_to_dict(result.eval_after_sft),
      "eval_after_rl": _eval_to_dict(result.eval_after_rl),
  }


def _results_path(results_dir: str, experiment: str, stage_name: str) -> str:
  return os.path.join(results_dir, experiment, f"{stage_name}.json")


def _select_stages(spec: Dict[str, Any], stage: str | None, run_all: bool) -> List[Stage]:
  stages: List[Stage] = spec["stages"]
  if run_all or stage is None:
    return stages
  matched = [s for s in stages if s.name == stage]
  if not matched:
    names = ", ".join(s.name for s in stages)
    raise SystemExit(f"unknown --stage {stage!r}; choose one of: {names} (or --all)")
  return matched


def _print_plan(experiment: str, stages: List[Stage], results_dir: str) -> None:
  print(f"[pipeline] experiment={experiment} results-dir={results_dir}")
  print(f"[pipeline] plan ({len(stages)} stage(s)):")
  for s in stages:
    path = _results_path(results_dir, experiment, s.name)
    status = "DONE (skip)" if os.path.exists(path) else "TODO"
    print(f"[pipeline]   {status:12s} {s.name:18s} {s.description}")
    print(f"[pipeline]                {' ':18s} kwargs={s.kwargs}")


def run(
    experiment: str,
    *,
    stage: str | None,
    run_all: bool,
    results_dir: str,
    model_dir: str,
    dry_run: bool,
) -> None:
  """Runs (or, with ``dry_run``, just prints) the selected tuning stages."""
  specs = _experiment_specs()
  if experiment not in specs:
    raise SystemExit(f"unknown --experiment {experiment!r}; choose: {', '.join(specs)}")
  spec = specs[experiment]
  stages = _select_stages(spec, stage, run_all)

  _print_plan(experiment, stages, results_dir)
  if dry_run:
    print("[pipeline] --dry-run: not running training.")
    return

  train_fn: Callable[[str, Dict[str, Any]], Any] = spec["train_fn"]
  serialize: Callable[[Any], Dict[str, Any]] = spec["serialize"]
  os.makedirs(os.path.join(results_dir, experiment), exist_ok=True)

  for s in stages:
    path = _results_path(results_dir, experiment, s.name)
    if os.path.exists(path):
      print(f"[pipeline] SKIP {s.name}: {path} already exists.", flush=True)
      continue
    print(f"[pipeline] RUN  {s.name}: {s.description}", flush=True)
    result = train_fn(model_dir, s.kwargs)
    payload = {
        "experiment": experiment,
        "stage": s.name,
        "description": s.description,
        "kwargs": {k: list(v) if isinstance(v, tuple) else v for k, v in s.kwargs.items()},
        "result": serialize(result),
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
      json.dump(payload, fh, indent=2)
    os.replace(tmp, path)  # atomic write so a crash never leaves a half stage "done"
    print(f"[pipeline] WROTE {path}", flush=True)

  print(f"[pipeline] DONE experiment={experiment}", flush=True)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--experiment", choices=["coding", "multiturn"], required=True)
  group = parser.add_mutually_exclusive_group()
  group.add_argument("--stage", help="run a single named stage (default: the first stage)")
  group.add_argument("--all", action="store_true", help="run every stage in order")
  parser.add_argument("--results-dir", default="results", help="per-stage results JSON dir (default: results)")
  parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Delphi weights dir (default: $DELPHI_MODEL_DIR or ./delphi)")
  parser.add_argument("--dry-run", action="store_true", help="print the plan and exit (no TPU, no training)")
  args = parser.parse_args()

  run(
      args.experiment,
      stage=args.stage,
      run_all=args.all,
      results_dir=args.results_dir,
      model_dir=args.model_dir,
      dry_run=args.dry_run,
  )


if __name__ == "__main__":
  main()
