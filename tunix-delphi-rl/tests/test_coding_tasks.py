# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for the held-out coding task ladder (:mod:`problems.coding_tasks`).

Extracted from the module's former ``__main__`` self-check: structural
invariants (count, unique ids) plus the hard guarantee that every reference
solution runs cleanly under :mod:`environments.micropython` and reproduces its
gold stdout exactly.
"""

from __future__ import annotations

import environments.micropython as micropython
from problems.coding_tasks import load_tasks


def test_task_ladder_structural_invariants():
  tasks = load_tasks()
  assert len(tasks) == 68, f"expected 68 tasks, got {len(tasks)}"
  ids = [t.id for t in tasks]
  assert len(set(ids)) == len(ids), "task ids are not unique"


def test_every_reference_solution_reproduces_gold():
  tasks = load_tasks()
  failures = []
  for t in tasks:
    r = micropython.run(t.solution)
    if not r.ok:
      failures.append(f"{t.id}: solution errored: {r.error}")
      continue
    if r.stdout != t.answer:
      failures.append(
          f"{t.id}: stdout mismatch expected={t.answer!r} actual={r.stdout!r}"
      )
  assert not failures, f"{len(failures)} task(s) failed validation: {failures}"
