# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Simple fixed-cadence curriculum over ordered difficulty levels (issue #8).

Inspired by marin's adaptive curriculum (``lib/marin/src/marin/rl/curriculum.py``)
but simplified per ``CURRICULUM_DESIGN.md``: we keep the **frontier principle**
(spend gradient where the model is still improving) but replace the
DAG/plateau-regression machinery with a **fixed-cadence schedule + mastery gate**.

State is tiny and deterministic: the highest unlocked level ``k`` (1-indexed), a
per-level exponential-moving-average success rate, and a step counter. The trainer
loop is:

    level = cur.sample_level(seed)        # pick a level for the next batch
    ...rollout + grade the batch...
    cur.record(level, batch_success)      # update that level's EMA
    cur.on_step()                         # advance the schedule (once per step)

Advancement: every ``steps_per_level`` steps we try to unlock ``k+1``. With the
**mastery gate** we only unlock when level ``k``'s EMA success >= ``promote_threshold``;
otherwise we hold and spend another window at ``k`` (the *fixed cadence* is the
fallback -- after ``max_holds`` windows we advance regardless, so a stuck level
never blocks the curriculum). Sampling is **cumulative** over ``{1..k}`` biased to
the newest (frontier) level with a floor on earlier levels (anti-forgetting), and
a level whose success exceeds ``graduate_threshold`` is dropped to the floor so
gradient flows to harder levels (marin's graduation, simplified).

CPU-only, dependency-light (numpy + random), with a ``__main__`` self-check.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Dict, List

import numpy as np


@dataclasses.dataclass
class CurriculumConfig:
  """Knobs for the fixed-cadence curriculum (see ``CURRICULUM_DESIGN.md``)."""

  num_levels: int
  """Number of ordered difficulty levels (1..num_levels)."""

  steps_per_level: int = 40
  """Cadence: training steps in a window before considering advancement."""

  promote_threshold: float = 0.7
  """EMA success on the current top level required to unlock the next (mastery gate)."""

  graduate_threshold: float = 0.95
  """EMA success above which a level is dropped to the floor weight."""

  max_holds: int = 3
  """After this many held windows at a level, advance anyway (fixed-cadence fallback)."""

  newest_weight: float = 0.6
  """Sampling mass given to the newest unlocked (frontier) level."""

  floor_weight: float = 0.05
  """Minimum sampling probability for any active (non-graduated) level."""

  ema_alpha: float = 0.1
  """Exponential-smoothing weight for per-level success (higher = more recent)."""

  prior_success: float = 0.5
  """Bayesian prior for a level's success EMA before any data."""


class Curriculum:
  """A fixed-cadence + mastery-gate scheduler over ordered difficulty levels."""

  def __init__(self, config: CurriculumConfig):
    if config.num_levels < 1:
      raise ValueError("num_levels must be >= 1")
    self.config = config
    self.k = 1  # highest unlocked level (1-indexed)
    self.step = 0
    self._steps_in_window = 0
    self._holds = 0
    self.ema: Dict[int, float] = {
        lvl: config.prior_success for lvl in range(1, config.num_levels + 1)
    }
    self.graduated: set[int] = set()

  # -- sampling -------------------------------------------------------------

  def active_levels(self) -> List[int]:
    """Unlocked levels (1..k)."""
    return list(range(1, self.k + 1))

  def sampling_weights(self) -> Dict[int, float]:
    """Cumulative weights over {1..k}, biased to the newest level, floored.

    The newest unlocked level (the frontier) gets ``newest_weight``; the rest
    share the remainder uniformly; graduated levels are pinned to the floor. Every
    active level keeps at least ``floor_weight`` so earlier skills are rehearsed.
    """
    active = self.active_levels()
    if len(active) == 1:
      return {active[0]: 1.0}

    newest = self.k
    others = [lvl for lvl in active if lvl != newest]
    weights: Dict[int, float] = {}
    weights[newest] = self.config.newest_weight
    rest = max(0.0, 1.0 - self.config.newest_weight)
    per_other = rest / len(others) if others else 0.0
    for lvl in others:
      weights[lvl] = per_other

    # Graduated levels drop to the floor (gradient flows to harder levels).
    for lvl in active:
      if lvl in self.graduated and lvl != newest:
        weights[lvl] = 0.0

    # Apply the floor, then renormalise.
    for lvl in active:
      weights[lvl] = max(weights[lvl], self.config.floor_weight)
    total = sum(weights.values())
    return {lvl: w / total for lvl, w in weights.items()}

  def sample_level(self, seed: int) -> int:
    """Sample a level for the next batch from the current weights."""
    weights = self.sampling_weights()
    levels = list(weights.keys())
    probs = np.array([weights[lvl] for lvl in levels], dtype=np.float64)
    probs = probs / probs.sum()
    rng = np.random.default_rng(seed)
    return int(levels[rng.choice(len(levels), p=probs)])

  # -- updates --------------------------------------------------------------

  def record(self, level: int, success_rate: float) -> None:
    """Update a level's EMA success from a batch's mean success (reward-based)."""
    a = self.config.ema_alpha
    self.ema[level] = (1.0 - a) * self.ema[level] + a * float(success_rate)
    if self.ema[level] >= self.config.graduate_threshold:
      self.graduated.add(level)

  def on_step(self) -> Dict[str, float]:
    """Advance the schedule by one step; unlock the next level on cadence/gate.

    Returns a small metrics dict for logging (current top level, window, EMA).
    """
    self.step += 1
    self._steps_in_window += 1
    if self._steps_in_window >= self.config.steps_per_level and self.k < self.config.num_levels:
      mastered = self.ema[self.k] >= self.config.promote_threshold
      forced = self._holds >= self.config.max_holds
      if mastered or forced:
        self.k += 1
        self._holds = 0
      else:
        self._holds += 1
      self._steps_in_window = 0
    return self.metrics()

  def metrics(self) -> Dict[str, float]:
    return {
        "curriculum/top_level": float(self.k),
        "curriculum/window_step": float(self._steps_in_window),
        "curriculum/holds": float(self._holds),
        "curriculum/ema_top": float(self.ema[self.k]),
        "curriculum/graduated": float(len(self.graduated)),
    }


if __name__ == "__main__":
  # CPU self-check: advances on mastery, holds-then-forces below threshold,
  # samples only unlocked levels, biases to the frontier, never forgets.
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
      m = cur.on_step()
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

  print("curriculum self-check OK")
