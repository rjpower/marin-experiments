# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for the fixed-cadence :mod:`environments.curriculum` scheduler.

Extracted from the module's former ``__main__`` self-check: advances on mastery,
holds-then-forces below threshold, samples only unlocked levels, biases to the
frontier, never forgets, and graduates saturated levels to the floor.
"""

from __future__ import annotations

from environments.curriculum import Curriculum, CurriculumConfig


def test_curriculum_scheduler():
  # Advances on mastery, holds-then-forces below threshold, samples only
  # unlocked levels, biases to the frontier, never forgets.
  cfg = CurriculumConfig(num_levels=3, steps_per_level=5, promote_threshold=0.6, max_holds=2)
  cur = Curriculum(cfg)
  assert cur.active_levels() == [1]
  assert cur.sample_level(0) == 1

  # Mastered level 1 (EMA crosses promote_threshold) -> unlock level 2 after a window.
  for _ in range(5):
    cur.record(1, 0.9)
    cur.on_step()
  assert cur.k == 2, (cur.k, cur.ema[1])

  # Sampling over {1,2}: newest (2) biased, level 1 floored but present.
  w = cur.sampling_weights()
  assert set(w) == {1, 2}
  assert w[2] > w[1] >= cfg.floor_weight
  assert abs(sum(w.values()) - 1.0) < 1e-9
  counts = {1: 0, 2: 0}
  for s in range(400):
    counts[cur.sample_level(s)] += 1
  assert counts[2] > counts[1] > 0  # frontier-biased, but level 1 still rehearsed

  # Level 2 stuck below threshold -> hold for max_holds windows, then force.
  start_k = cur.k
  forced_advance_step = None
  for w_idx in range(cfg.max_holds + 1):
    for _ in range(5):
      cur.record(2, 0.2)  # below promote_threshold
      cur.on_step()
    if cur.k > start_k and forced_advance_step is None:
      forced_advance_step = w_idx
  assert cur.k == 3, (cur.k, cur._holds)  # forced to the last level despite low success

  # Graduation: a saturated level drops to the floor.
  cur2 = Curriculum(CurriculumConfig(num_levels=3, graduate_threshold=0.95))
  cur2.k = 3
  for _ in range(100):
    cur2.record(1, 1.0)
  assert 1 in cur2.graduated
  w2 = cur2.sampling_weights()
  assert abs(w2[1] - cfg.floor_weight) < 1e-9 or w2[1] <= w2[3]
