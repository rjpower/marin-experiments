"""M2 gate test: the toy "emit more cats" GRPO loop LEARNS and is WIRED.

Runs :func:`toy_cats.train_toy` and asserts BOTH milestone gates the critique
required (they are independent):

  1. LEARNS -- the mean cat-fraction reward strictly increases from the start of
     training to the end, with a clear margin (first-window mean vs last-window
     mean). A flat curve would mean the rollout -> reward -> advantage ->
     optimizer-update loop is not actually learning.

  2. WIRED / cache path exercised -- DIRECT mechanism assertions, not inferred
     from reward-go-up:
       a. the rollout decoded multiple tokens per completion (length > 1) and
          completions VARY across the group (sampling diversity is live);
       b. a direct KV-cache probe shows the cache ``end_index`` strictly
          advances across decode steps (the cache path is threaded, not a
          full-sequence re-forward).

Run with::

    JAX_PLATFORMS=cpu python test_smoke_cats.py

or under pytest (``test_smoke_cats`` is discoverable).
"""

import numpy as np

import toy_cats


# Window size for the start-vs-end reward comparison (the LEARNS gate).
_WINDOW = 10
# Minimum reward improvement required to call the loop "learning".
_MIN_IMPROVEMENT = 0.15


def _summarize_curve(history: list[float]) -> None:
  """Prints the per-checkpoint reward trajectory."""
  print(f"[curve] {len(history)} steps recorded")
  for i in range(0, len(history), _WINDOW):
    print(f"[curve]   step {i:3d}: mean cat-fraction = {history[i]:.4f}")
  if (len(history) - 1) % _WINDOW != 0:
    print(
        f"[curve]   step {len(history) - 1:3d}: mean cat-fraction = "
        f"{history[-1]:.4f}"
    )


def _run_gates() -> toy_cats.ToyResult:
  """Runs the toy GRPO loop and asserts the LEARNS and WIRED gates."""
  result = toy_cats.train_toy(steps=80)
  history = result.reward_history

  assert len(history) >= 2 * _WINDOW, (
      f"Too few reward points ({len(history)}) to evaluate the gate over "
      f"windows of {_WINDOW}."
  )
  _summarize_curve(history)

  # ---- Gate 1: LEARNS ------------------------------------------------------
  start_mean = float(np.mean(history[:_WINDOW]))
  end_mean = float(np.mean(history[-_WINDOW:]))
  improvement = end_mean - start_mean
  print(
      f"[GATE 1 LEARNS] first-{_WINDOW} mean = {start_mean:.4f}  "
      f"last-{_WINDOW} mean = {end_mean:.4f}  improvement = {improvement:+.4f}"
  )
  assert end_mean > start_mean, (
      f"Mean reward did not increase: start={start_mean:.4f} "
      f"end={end_mean:.4f}."
  )
  assert improvement >= _MIN_IMPROVEMENT, (
      f"Mean reward improvement {improvement:.4f} below the required "
      f"margin {_MIN_IMPROVEMENT}."
  )
  print("[GATE 1 LEARNS] PASS")

  # ---- Gate 2a: rollout decoded multiple, varying tokens -------------------
  # Diversity is checked on the FIRST rollout (early training): by the end the
  # policy has converged to all-cats, so identical completions there are the
  # CORRECT converged behavior, not a wiring failure. Multi-token decode is
  # checked on both first and last rollouts.
  first = result.first_completions
  last = result.last_completions
  assert first and last, "No completions captured from the rollouts."
  first_lengths = [len(c.split()) for c in first]
  last_lengths = [len(c.split()) for c in last]
  print(
      f"[GATE 2a WIRED-rollout] first rollout: {len(first)} completions, "
      f"token-lengths min={min(first_lengths)} max={max(first_lengths)}, "
      f"distinct={len(set(first))}"
  )
  print(
      f"[GATE 2a WIRED-rollout] last  rollout: {len(last)} completions, "
      f"token-lengths min={min(last_lengths)} max={max(last_lengths)}, "
      f"distinct={len(set(last))}"
  )
  # The decode loop must run many steps: the longest completion spans many
  # tokens. (Early, an untrained policy may emit EOS immediately for a
  # zero-length completion -- that is valid, so we do not require a positive
  # MIN early. The converged last rollout, which runs the full budget, is the
  # strong multi-step witness.)
  assert max(first_lengths) > 1, (
      "First rollout never decoded more than one token; the decode loop did "
      "not run multiple steps."
  )
  assert min(last_lengths) > 1, (
      "Converged rollout produced single-token completions; the decode loop "
      "did not run multiple steps."
  )
  assert len(set(first)) > 1, (
      "Early-training completions were all identical; rollout sampling produced "
      "no diversity (GRPO would have zero within-group variance)."
  )
  print("[GATE 2a WIRED-rollout] PASS")

  # ---- Gate 2b: KV cache end_index strictly advances -----------------------
  generated, end_indices = toy_cats.probe_cache_advances(
      result.model, max_new_tokens=8
  )
  print(
      f"[GATE 2b WIRED-cache] generated {len(generated)} tokens, "
      f"cache end_index trajectory = {end_indices}"
  )
  assert len(generated) > 1, "Cache probe decoded only a single token."
  assert all(
      end_indices[i + 1] == end_indices[i] + 1
      for i in range(len(end_indices) - 1)
  ), f"KV cache end_index did not advance by 1 per step: {end_indices}"
  print("[GATE 2b WIRED-cache] PASS")

  print("\nM2 GATES PASSED.")
  return result


def test_smoke_cats() -> None:
  """Pytest entry point: runs the toy GRPO loop and asserts both M2 gates."""
  _run_gates()


if __name__ == "__main__":
  _run_gates()
