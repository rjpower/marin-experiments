# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris entrypoint for the tunix toy-GRPO smoke test (milestone M3).

Runs the same "emit more cats" GRPO loop validated locally in M2 (`toy_cats`),
but on whatever worker iris schedules it on. The worker installs this
experiment's pinned deps via `uv sync` before invoking `python launch.py`, so a
green run here proves the full path: bundle -> uv sync (tunix + jax) ->
jax.distributed init -> GRPO loop learns.

Submit on CPU (M3a — proves packaging + submission, no accelerator):

    uv run iris --cluster=marin job run --no-wait \
      --cpu 2 --memory 12GB --disk 20GB \
      -- python launch.py

Submit on a single-host TPU (M3b — additionally proves jax[tpu]/libtpu + the
TPU jax.distributed path):

    uv run iris --cluster=marin job run --no-wait \
      --tpu v5litepod-4 --enable-extra-resources --extra tpu \
      --zone us-central2-b --cpu 8 --memory 32GB --disk 40GB \
      -- python launch.py
"""

import os

import jax

from toy_cats import probe_cache_advances, train_toy

# Margins kept generous so transient sampling noise can't fail the job; the
# local M2 run improves by ~+0.8, far above this floor.
_MIN_IMPROVEMENT = 0.15
_TAIL = 10


def main() -> None:
    """Runs the toy GRPO loop on the iris worker and asserts it learned."""
    steps = int(os.environ.get("TOY_STEPS", "80"))
    print(f"[launch] jax {jax.__version__} devices={jax.devices()}", flush=True)

    result = train_toy(steps=steps)
    history = result.reward_history
    if not history:
        raise RuntimeError("toy GRPO produced no reward history")

    first = sum(history[:_TAIL]) / min(_TAIL, len(history))
    last = sum(history[-_TAIL:]) / min(_TAIL, len(history))
    improvement = last - first
    for step, mean in result.checkpoint_means:
        print(f"[launch] step {step:4d}: mean cat-fraction = {mean:.4f}", flush=True)
    print(
        f"[launch] LEARNS: first-{_TAIL} mean={first:.4f} last-{_TAIL} mean={last:.4f}"
        f" improvement={improvement:+.4f}",
        flush=True,
    )

    # WIRED gate: directly exercise the KV-cache decode path on the worker.
    tokens, end_index = probe_cache_advances(result.model, max_new_tokens=8)
    expected = list(range(1, len(tokens) + 1))
    print(f"[launch] WIRED: cache end_index trajectory={end_index}", flush=True)

    if improvement < _MIN_IMPROVEMENT:
        raise RuntimeError(
            f"toy GRPO did not learn: improvement {improvement:+.4f} < {_MIN_IMPROVEMENT}"
        )
    if end_index != expected:
        raise RuntimeError(f"KV cache did not advance correctly: {end_index} != {expected}")

    print("[launch] M3 SMOKE TEST PASSED", flush=True)


if __name__ == "__main__":
    main()
