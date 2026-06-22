# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for the curriculum problem families (:mod:`problems.coding_problems`).

Extracted verbatim from the module's former ``__main__`` self-check. Covers:
references solve all their own tests; a constant program is discriminated; an
empty program scores zero; partial credit is genuinely partial; eval-set
determinism; and feedback budget + no-hidden-test leakage.
"""

from __future__ import annotations

import random

import problems.coding_problems as cp
from problems.coding_problems import (
    NUM_LEVELS,
    Problem,
    format_test_feedback,
    grade_problem,
    load_eval_problems,
    problem_reward,
    sample_problem,
)


def test_references_solve_all_their_own_tests():
  # Every family at every level: the reference solves ALL its tests.
  assert set(cp._FAMILIES_BY_LEVEL) == set(range(1, NUM_LEVELS + 1)), (
      cp._FAMILIES_BY_LEVEL.keys()
  )
  for level in range(1, NUM_LEVELS + 1):
    fams = cp._FAMILIES_BY_LEVEL[level]
    assert 3 <= len(fams) <= 4, f"level {level} has {len(fams)} families"
    for family in fams:
      rng = random.Random(12345 + family.level)
      problem = cp._build_problem(family, rng)
      assert problem.level == level
      assert problem.family == family.key
      # Minimum public/hidden counts.
      assert len(problem.public_tests) <= cp._N_PUBLIC
      assert len(problem.hidden_tests) >= cp._MIN_HIDDEN, (
          family.key,
          len(problem.hidden_tests),
      )
      # The reference passes every one of its own tests, exactly.
      grade = grade_problem(family.reference, problem)
      assert grade["frac_passed"] == 1.0, (family.key, grade)
      assert grade["exact"] == 1.0, (family.key, grade)
      assert grade["ran_ok"] == 1.0, (family.key, grade)
      assert abs(problem_reward(grade) - 1.0) < 1e-9, (family.key, grade)


def test_constant_program_is_discriminated():
  # A trivially-wrong program scores frac_passed < 1.0 on MOST problems and
  # still registers has_code == 1. (micropython rejects ``def solve(*a)``, so
  # the spec's "return 0" program is written with fixed-arity defaults that
  # accept the 0/1/2-arg call shapes across all families.)
  wrong_program = "def solve(a=0, b=0, c=0):\n  return 0"
  n_discriminated = 0
  n_checked = 0
  rng = random.Random(7)
  for level in range(1, NUM_LEVELS + 1):
    for _ in range(4):
      problem = sample_problem(rng, level)
      grade = grade_problem(wrong_program, problem)
      assert grade["has_code"] == 1.0, (problem.family, grade)
      n_checked += 1
      if grade["frac_passed"] < 1.0:
        n_discriminated += 1
  assert n_discriminated >= int(0.9 * n_checked), (
      f"constant program passed too often: {n_discriminated}/{n_checked}"
  )


def test_empty_program_scores_zero():
  # An empty program scores all-zero components and zero reward.
  empty_problem = sample_problem(random.Random(0), 1)
  empty_grade = grade_problem("", empty_problem)
  assert empty_grade["has_code"] == 0.0
  assert empty_grade["ran_ok"] == 0.0
  assert empty_grade["frac_passed"] == 0.0
  assert empty_grade["exact"] == 0.0
  assert problem_reward(empty_grade) == 0.0


def test_partial_credit_is_genuinely_partial():
  # A program that passes some-but-not-all tests scores a reward strictly
  # between 0 and 1 (the gradient the whole design needs). sum_1_to_n with an
  # off-by-one upper bound passes for n<=1 but fails for larger n.
  partial_problem = None
  partial_rng = random.Random(99)
  for _ in range(200):
    cand = sample_problem(partial_rng, 2)
    if cand.family == "l2_sum_1_to_n":
      partial_problem = cand
      break
  assert partial_problem is not None, "could not sample l2_sum_1_to_n"
  buggy = "def solve(n):\n  total = 0\n  for i in range(1, n):\n    total += i\n  return total"
  partial_grade = grade_problem(buggy, partial_problem)
  partial = problem_reward(partial_grade)
  assert partial_grade["has_code"] == 1.0
  assert 0.0 < partial_grade["frac_passed"] < 1.0, partial_grade
  assert 0.0 < partial < 1.0, partial


def test_eval_set_is_deterministic_per_seed():
  # sample_problem / load_eval_problems return valid Problems for every level;
  # load_eval_problems is deterministic for a fixed seed, and a different seed
  # yields a different held-out set (checked in aggregate across all levels).
  ids_seed_a: list[str] = []
  ids_seed_b: list[str] = []
  for level in range(1, NUM_LEVELS + 1):
    p = sample_problem(random.Random(level), level)
    assert isinstance(p, Problem) and p.level == level
    assert len(p.public_tests) + len(p.hidden_tests) >= cp._MIN_TESTS

    evA = load_eval_problems(level, 8, seed=2024)
    evB = load_eval_problems(level, 8, seed=2024)
    assert len(evA) == 8
    assert [x.id for x in evA] == [x.id for x in evB], "eval set not deterministic"
    assert all(x.level == level for x in evA)
    assert all(len(x.hidden_tests) >= cp._MIN_HIDDEN for x in evA)
    ids_seed_a.extend(x.id for x in evA)
    ids_seed_b.extend(x.id for x in load_eval_problems(level, 8, seed=777))
  assert ids_seed_a != ids_seed_b, "different seeds gave an identical eval set"


def test_feedback_budget_and_no_hidden_test_leak():
  # Prompt mentions solve and ends with the END instruction. Feedback stays
  # within the char budget and never reveals a hidden test's full call line.
  fb_rng = random.Random(5)
  prob5 = sample_problem(fb_rng, 5)
  assert "solve" in prob5.prompt
  assert prob5.prompt.rstrip().endswith("END.")
  wrong = "def solve(a=0, b=0, c=0):\n  return 0"
  for level in range(1, NUM_LEVELS + 1):
    for _ in range(3):
      prob = sample_problem(fb_rng, level)
      for graded in (cp._family_for(prob.family).reference, wrong):
        fb = format_test_feedback(graded, prob)
        assert len(fb) <= cp._MAX_FEEDBACK_CHARS, (prob.family, len(fb))
        # The feedback references only public-test calls; no hidden call line.
        for args, _expected in prob.hidden_tests:
          call = "solve(" + ", ".join(repr(a) for a in args) + ")"
          public_calls = {
              "solve(" + ", ".join(repr(a) for a in pa) + ")"
              for (pa, _pe) in prob.public_tests
          }
          if call not in public_calls:
            assert call not in fb, ("hidden call leaked", prob.family, call)
