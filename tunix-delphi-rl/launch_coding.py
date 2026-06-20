# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris entrypoint for Delphi agentic-CODING (issue #7).

Bootstraps the Delphi 447M base LM into a code generator that writes small Python
programs graded by the purely-functional :mod:`micropython` interpreter, via the
SFT warm-up -> Dr.GRPO RL recipe in :mod:`train_coding`, and evaluates greedy
solve-rate on the fixed 50-task ladder (:mod:`coding_tasks`).

Config is read from env vars so the coordinator can size the run:
  * ``CODING_TIERS`` (default ``0,1,2,3,4``) -- curriculum tiers to train on.
  * ``CODING_STEPS`` (default 120) -- Dr.GRPO steps (0 = SFT/few-shot eval only).
  * ``CODING_SFT_STEPS`` (default 0) -- SFT warm-up steps (0 = few-shot only).
  * ``CODING_SFT_LR`` (default 1e-4), ``CODING_LR`` (default 1e-5).
  * ``CODING_NUM_GENERATIONS`` (default 16), ``CODING_BATCH_SIZE`` (default 8).
  * ``CODING_TEMPERATURE`` (default 0.9).
  * ``CODING_MAX_TOKENS`` (default 160), ``CODING_MAX_PROMPT`` (default 384).
  * ``CODING_SEED`` (default 0).
  * ``DELPHI_MODEL_DIR`` (default ``./delphi`` on the worker).

Submit on a single-host TPU (the coordinator submits; do NOT submit from here):

    uv run iris --cluster=marin job run --no-wait \
      --tpu v6e-4 --enable-extra-resources --extra tpu --region europe-west4 \
      --cpu 8 --memory 64GB --disk 60GB --max-retries 5 --job-name coding-sft \
      -e CODING_TIERS 0,1,2,3,4 -e CODING_SFT_STEPS 300 -e CODING_STEPS 150 \
      -e CODING_NUM_GENERATIONS 16 -e CODING_BATCH_SIZE 8 -e CODING_LR 1e-5 \
      -- python launch_coding.py
"""

import os

import jax
from huggingface_hub import snapshot_download

from environments.coding_env import strip_answer_hint
from problems.coding_tasks import load_tasks
from training.train_coding import train_coding

DELPHI_REPO = "marin-community/delphi-3e18-447Mparams-1.2Btokens"


def _ensure_delphi(model_dir: str) -> str:
  """Downloads Delphi to ``model_dir`` on the worker if not already present."""
  if not os.path.exists(os.path.join(model_dir, "model.safetensors")):
    snapshot_download(repo_id=DELPHI_REPO, local_dir=model_dir)
  return model_dir


def _parse_tiers(raw: str) -> tuple[int, ...]:
  return tuple(int(t) for t in raw.split(",") if t.strip() != "")


def _print_tier_table(label_a, eval_a, label_b, eval_b) -> None:
  """Prints a per-tier solve comparison (e.g. few-shot/sft vs after-rl)."""
  print(f"[coding] PER-TIER SOLVE ({label_a} -> {label_b}):", flush=True)
  tiers = sorted(set(eval_a.per_tier()) | (set(eval_b.per_tier()) if eval_b else set()))
  for tier in tiers:
    sa, na = eval_a.per_tier().get(tier, (0, 0))
    cell_b = ""
    if eval_b is not None:
      sb, nb = eval_b.per_tier().get(tier, (0, 0))
      cell_b = f"  {label_b}={sb}/{nb}"
    print(f"[coding]   tier {tier}: {label_a}={sa}/{na}{cell_b}", flush=True)
  total_b = f"  {label_b}={eval_b.solved}/{eval_b.total}" if eval_b else ""
  print(
      f"[coding]   TOTAL: {label_a}={eval_a.solved}/{eval_a.total}{total_b}",
      flush=True,
  )


def _print_misses(result) -> None:
  """Prints every UNSOLVED task in the final eval (id, tier, output vs gold)."""
  final = result.eval_after_rl or result.eval_after_sft
  if final is None:
    return
  tasks_by_id = {t.id: t for t in load_tasks()}
  misses = [r for r in final.rows if not r.solved]
  print(f"[coding] UNSOLVED ({len(misses)}/{final.total}):", flush=True)
  for row in sorted(misses, key=lambda r: (r.tier, r.task_id)):
    gold = tasks_by_id[row.task_id].answer
    prog = row.program.replace("\n", " ; ")
    print(
        f"[coding]   XX [t{row.tier} {row.task_id}] ran_ok={row.ran_ok} "
        f"out={row.output!r} gold={gold!r} prog={prog[:90]!r}",
        flush=True,
    )


def _print_examples(result, n_per_tier: int = 2) -> None:
  """Prints a few (task -> program -> output) examples from the after-RL eval."""
  final = result.eval_after_rl or result.eval_after_sft
  if final is None:
    return
  tasks_by_id = {t.id: t for t in load_tasks()}
  print("[coding] SAMPLE PROGRAMS (after-rl):", flush=True)
  # Show up to n_per_tier solved AND n_per_tier failed per tier (separate quotas
  # so failures are not crowded out by solved rows).
  shown_ok: dict[int, int] = {}
  shown_xx: dict[int, int] = {}
  for want_solved in (True, False):
    bucket = shown_ok if want_solved else shown_xx
    for row in final.rows:
      if row.solved != want_solved:
        continue
      if bucket.get(row.tier, 0) >= n_per_tier:
        continue
      bucket[row.tier] = bucket.get(row.tier, 0) + 1
      task = tasks_by_id[row.task_id]
      prompt = strip_answer_hint(task.prompt, task.answer)
      mark = "OK " if row.solved else "XX "
      prog1 = row.program.replace("\n", " ; ")
      print(
          f"[coding]   {mark}[t{row.tier} {row.task_id}] {prompt}\n"
          f"[coding]       program: {prog1[:160]}\n"
          f"[coding]       output={row.output!r} gold={task.answer!r}",
          flush=True,
      )


def main() -> None:
  """Downloads Delphi and runs the coding SFT+Dr.GRPO pipeline on the worker."""
  tiers = _parse_tiers(os.environ.get("CODING_TIERS", "0,1,2,3,4"))
  steps = int(os.environ.get("CODING_STEPS", "120"))
  sft_steps = int(os.environ.get("CODING_SFT_STEPS", "0"))
  sft_lr = float(os.environ.get("CODING_SFT_LR", "1e-4"))
  learning_rate = float(os.environ.get("CODING_LR", "1e-5"))
  num_generations = int(os.environ.get("CODING_NUM_GENERATIONS", "16"))
  batch_size = int(os.environ.get("CODING_BATCH_SIZE", "8"))
  temperature = float(os.environ.get("CODING_TEMPERATURE", "0.9"))
  max_tokens = int(os.environ.get("CODING_MAX_TOKENS", "160"))
  max_prompt = int(os.environ.get("CODING_MAX_PROMPT", "384"))
  seed = int(os.environ.get("CODING_SEED", "0"))
  model_dir = os.environ.get("DELPHI_MODEL_DIR", "./delphi")

  print(f"[coding] jax {jax.__version__} devices={jax.devices()}", flush=True)
  print(
      f"[coding] tiers={tiers} steps={steps} sft_steps={sft_steps} "
      f"lr={learning_rate} sft_lr={sft_lr} num_generations={num_generations} "
      f"batch_size={batch_size} temp={temperature} max_tokens={max_tokens}",
      flush=True,
  )

  model_dir = _ensure_delphi(model_dir)
  print(f"[coding] Delphi ready at {model_dir}", flush=True)

  result = train_coding(
      model_dir=model_dir,
      tiers=tiers,
      steps=steps,
      num_generations=num_generations,
      batch_size=batch_size,
      learning_rate=learning_rate,
      temperature=temperature,
      max_prompt_length=max_prompt,
      max_tokens_to_generate=max_tokens,
      seed=seed,
      sft_steps=sft_steps,
      sft_learning_rate=sft_lr,
      eval_max_new_tokens=max_tokens,
  )

  for i in range(result.steps_ran):
    def _at(history, idx):
      return history[idx] if idx < len(history) else float("nan")

    print(
        f"[coding] step {i:4d}: mean_reward={_at(result.reward_history, i):.4f} "
        f"solve_ratio={_at(result.solve_ratio_history, i):.4f} "
        f"ran_ok={_at(result.ran_ok_history, i):.4f} "
        f"has_code={_at(result.has_code_history, i):.4f}",
        flush=True,
    )

  baseline_label = "after-sft" if sft_steps > 0 else "few-shot"
  if result.eval_after_sft is not None:
    _print_tier_table(
        baseline_label, result.eval_after_sft, "after-rl", result.eval_after_rl
    )
    _print_misses(result)
    _print_examples(result)

  print(f"[coding] CODING COMPLETE (RL steps={result.steps_ran}, tiers={tiers})", flush=True)


if __name__ == "__main__":
  main()
