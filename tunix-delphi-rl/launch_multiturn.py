# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris entrypoint for Delphi MULTI-TURN agentic coding (issue #8).

Bootstraps Delphi into a multi-turn coder that writes a program, runs it via the
:mod:`micropython` interpreter, reads the output, and revises -- up to ``MT_ROUNDS``
rounds -- trained with the SFT warm-up -> Dr.GRPO recipe in :mod:`train_multiturn`,
and reports greedy first-attempt vs best-across-rounds solve on the fixed ladder
(:mod:`coding_tasks`, filtered to the eval tiers).

Config is read from env vars so the coordinator can size the run:
  * ``MT_TIERS`` (default ``3,4,5``)        -- curriculum tiers to TRAIN on.
  * ``MT_EVAL_TIERS`` (default = ``MT_TIERS``) -- tiers to EVAL on.
  * ``MT_ROUNDS`` (default 5)               -- max write->run->revise rounds.
  * ``MT_STEPS`` (default 120)              -- Dr.GRPO steps (0 = SFT/eval only).
  * ``MT_SFT_STEPS`` (default 0)            -- SFT warm-up steps.
  * ``MT_SFT_FIX_PROB`` (default 0.3)       -- fraction of SFT fix transcripts.
  * ``MT_SFT_LR`` (1e-4), ``MT_LR`` (1e-5).
  * ``MT_NUM_GENERATIONS`` (16), ``MT_BATCH_SIZE`` (8), ``MT_TEMPERATURE`` (1.0).
  * ``MT_MAX_PROMPT`` (1024), ``MT_MAX_RESPONSE`` (640), ``MT_EVAL_TOKENS`` (192).
  * ``MT_SEED`` (0), ``DELPHI_MODEL_DIR`` (``./delphi`` on the worker).

Submit on a single-host TPU (the coordinator submits; do NOT submit from here):

    uv run iris --cluster=marin job run --no-wait \
      --tpu v6e-4 --enable-extra-resources --extra tpu --region europe-west4 \
      --cpu 8 --memory 64GB --disk 60GB --max-retries 5 --job-name mt-coding-sft \
      -e MT_TIERS 3,4,5 -e MT_SFT_STEPS 600 -e MT_STEPS 0 \
      -- python launch_multiturn.py
"""

import os

import jax
from huggingface_hub import snapshot_download

from coding_tasks import load_tasks
from train_multiturn import train_multiturn

DELPHI_REPO = "marin-community/delphi-3e18-447Mparams-1.2Btokens"


def _ensure_delphi(model_dir: str) -> str:
  if not os.path.exists(os.path.join(model_dir, "model.safetensors")):
    snapshot_download(repo_id=DELPHI_REPO, local_dir=model_dir)
  return model_dir


def _parse_tiers(raw: str) -> tuple[int, ...]:
  return tuple(int(t) for t in raw.split(",") if t.strip() != "")


def _print_tier_table(label, ev) -> None:
  """Prints per-tier first-attempt vs best-across-rounds solve for one eval."""
  if ev is None:
    return
  print(f"[mt-coding] PER-TIER ({label}) first -> best:", flush=True)
  for tier in sorted(ev.per_tier()):
    f, b, n = ev.per_tier()[tier]
    print(f"[mt-coding]   tier {tier}: first={f}/{n} best={b}/{n}", flush=True)
  print(
      f"[mt-coding]   TOTAL ({label}): first={ev.first_solved}/{ev.total} "
      f"best={ev.best_solved}/{ev.total}",
      flush=True,
  )


def _print_examples(ev, n_per_tier: int = 2) -> None:
  """Prints a few (task -> final program -> output) examples from an eval."""
  if ev is None:
    return
  tasks_by_id = {t.id: t for t in load_tasks()}
  print("[mt-coding] SAMPLE FINAL PROGRAMS:", flush=True)
  shown_ok: dict[int, int] = {}
  shown_xx: dict[int, int] = {}
  for want_solved in (True, False):
    bucket = shown_ok if want_solved else shown_xx
    for row in ev.rows:
      if row.best_solved != want_solved or bucket.get(row.tier, 0) >= n_per_tier:
        continue
      bucket[row.tier] = bucket.get(row.tier, 0) + 1
      task = tasks_by_id[row.task_id]
      mark = "OK " if row.best_solved else "XX "
      prog = row.final_program.replace("\n", " ; ")
      print(
          f"[mt-coding]   {mark}[t{row.tier} {row.task_id}] rounds={row.rounds_used} "
          f"first={row.first_solved}\n"
          f"[mt-coding]       program: {prog[:200]}\n"
          f"[mt-coding]       output={row.final_output!r} gold={task.answer!r}",
          flush=True,
      )


def main() -> None:
  tiers = _parse_tiers(os.environ.get("MT_TIERS", "3,4,5"))
  eval_tiers = _parse_tiers(os.environ["MT_EVAL_TIERS"]) if os.environ.get("MT_EVAL_TIERS") else tiers
  rounds = int(os.environ.get("MT_ROUNDS", "5"))
  steps = int(os.environ.get("MT_STEPS", "120"))
  sft_steps = int(os.environ.get("MT_SFT_STEPS", "0"))
  sft_fix_prob = float(os.environ.get("MT_SFT_FIX_PROB", "0.3"))
  sft_lr = float(os.environ.get("MT_SFT_LR", "1e-4"))
  learning_rate = float(os.environ.get("MT_LR", "1e-5"))
  num_generations = int(os.environ.get("MT_NUM_GENERATIONS", "16"))
  batch_size = int(os.environ.get("MT_BATCH_SIZE", "8"))
  temperature = float(os.environ.get("MT_TEMPERATURE", "1.0"))
  max_prompt = int(os.environ.get("MT_MAX_PROMPT", "1024"))
  max_response = int(os.environ.get("MT_MAX_RESPONSE", "640"))
  eval_tokens = int(os.environ.get("MT_EVAL_TOKENS", "192"))
  seed = int(os.environ.get("MT_SEED", "0"))
  model_dir = os.environ.get("DELPHI_MODEL_DIR", "./delphi")

  print(f"[mt-coding] jax {jax.__version__} devices={jax.devices()}", flush=True)
  print(
      f"[mt-coding] tiers={tiers} eval_tiers={eval_tiers} rounds={rounds} "
      f"steps={steps} sft_steps={sft_steps} sft_fix_prob={sft_fix_prob} "
      f"lr={learning_rate} sft_lr={sft_lr} num_generations={num_generations} "
      f"batch_size={batch_size} temp={temperature} max_prompt={max_prompt} "
      f"max_response={max_response}",
      flush=True,
  )

  model_dir = _ensure_delphi(model_dir)
  print(f"[mt-coding] Delphi ready at {model_dir}", flush=True)

  result = train_multiturn(
      model_dir=model_dir,
      tiers=tiers,
      eval_tiers=eval_tiers,
      rounds=rounds,
      steps=steps,
      num_generations=num_generations,
      batch_size=batch_size,
      learning_rate=learning_rate,
      temperature=temperature,
      max_prompt_length=max_prompt,
      max_response_length=max_response,
      seed=seed,
      sft_steps=sft_steps,
      sft_learning_rate=sft_lr,
      sft_fix_prob=sft_fix_prob,
      eval_max_new_tokens=eval_tokens,
  )

  for i in range(result.steps_ran):
    def _at(history, idx):
      return history[idx] if idx < len(history) else float("nan")

    print(
        f"[mt-coding] step {i:4d}: reward={_at(result.reward_history, i):.4f} "
        f"first_solve={_at(result.first_solve_history, i):.4f} "
        f"best_solve={_at(result.solve_ratio_history, i):.4f}",
        flush=True,
    )

  _print_tier_table("after-sft" if sft_steps > 0 else "few-shot", result.eval_after_sft)
  _print_tier_table("after-rl", result.eval_after_rl)
  _print_examples(result.eval_after_rl or result.eval_after_sft)

  print(f"[mt-coding] MULTITURN COMPLETE (RL steps={result.steps_ran}, tiers={tiers})", flush=True)


if __name__ == "__main__":
  main()
