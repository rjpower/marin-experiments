# Test-case RL redesign + curriculum (issue #8)

A design to turn the coding task into a regime where Dr.GRPO gets a **clear win**,
by fixing the three things that starve RL today and adding a simple curriculum
that keeps the model training at its frontier.

## Why the current setup starves RL (recap)

Dr.GRPO's advantage is `Aᵢ = rᵢ − mean(r_group)` — it learns **only from
intra-group reward variance**. Our setup produces almost none:

1. **Binary single-gold reward** (exact stdout match on one input) → a group of G
   samples is ~all-pass or ~all-fail → advantage ≈ 0.
2. **Bimodal difficulty** (template-known pass@1≈0.94 vs unknown ≈0.00) → nothing
   sits at the frontier (pass@1 ∈ (0.1, 0.9)) where samples disagree.
3. **Empty-program collapse** on hard tasks → ~zero entropy → never samples a
   reward → can't bootstrap.

Measured proof the *machinery* is fine: in-distribution, `mt-rl-light` training
`first_solve` climbed 0.5 → 0.94. RL worked exactly where variance existed.

## What we borrow from marin's curriculum (`lib/marin/src/marin/rl/curriculum.py`)

Marin runs an **adaptive** curriculum: each *lesson* is an (env, difficulty);
sampling weights use a quadratic that **peaks at ~50% success** (the frontier —
where reward variance and learning are maximal), with an exploration bonus for
under-sampled lessons and a minimum floor. Lessons **unlock** when their
dependencies plateau and **graduate** when their own success plateaus high.
Success is an EMA (Bayesian prior 0.5); plateau is a regression-slope test.

We keep the **frontier principle** (train where success ≈ 0.5) but, per the ask,
replace the adaptive machinery with a **simple fixed-cadence schedule** plus an
optional mastery gate.

## Design

### 1. Test-case problem representation (the foundation)

Replace "write a script whose stdout equals one gold string" with **"write
`def solve(...)` graded on N inputs"** — the canonical RLVR-for-code setup.

- A `Problem` has: `id`, `level` (1..L), `prompt` (describes the `solve`
  signature + behaviour), `public_tests` (≤2 shown to the agent), `hidden_tests`
  (graded, not shown), and a procedural `generator` (random params per instance).
- **Grading** (`grade(program)`): for each test `(args, expected)`, run
  `program + "\nprint(solve(*args))"` through `micropython` and compare stdout to
  `expected` (computed by a reference oracle at generation time). Return the
  **fraction of tests passed** plus `ran_ok`/`has_code`.
- **Dense reward**: `0.10·has_code + 0.20·ran_ok + 0.70·frac_passed`
  (so all-tests-pass → 1.0; a program passing 3/8 scores ~0.36). This makes
  groups have **continuous variance** even when no sample is perfect — the fix
  for cause (1).
- **Anti-hardcode**: random params per instance + many hidden tests incl. edge
  cases (empty, n=0, negatives, duplicates), so memorising a constant fails — the
  same trick that worked for arithmetic/CALC, now enforced by the tests.
- **Multi-turn**: the public tests + the *failing* hidden-test summary are the
  `Tool result:` feedback the agent iterates on (logic-bug signal, not just
  crashes).

### 2. Graded difficulty levels (the search space)

`L` ordered levels, each a set of procedurally-parameterised `solve` families with
increasing composition depth. A continuous knob (input size / param range) varies
difficulty *within* a level so the frontier is dense, not bimodal — the fix for
cause (2).

| level | theme | example families |
|------|-------|------------------|
| 1 | return / arithmetic | `add(a,b)`, `const`, `abs_diff` |
| 2 | one loop / branch | `sum_1_n`, `max_of_list`, `count_evens` |
| 3 | basic algorithms | `factorial`, `fib`, `reverse_list`, `gcd` |
| 4 | simple composition | `is_prime`, `bubble_sort`, `digit_sum`, `count_vowels` |
| 5 | multi-step | `second_largest`, `run_length_encode`, `digital_root`, `caesar` |
| 6 | harder | `nth_prime`, `collatz_len`, `most_common_word`, `is_palindrome_sentence` |

Each family ships a **reference oracle** (plain Python) to compute expected test
outputs, and a generator for random instances.

### 3. The curriculum: simple fixed cadence + mastery gate

A tiny scheduler (`curriculum.py`, ours) over the ordered levels. State = the
highest **unlocked** level `k` (start `k=1`) and a per-level success EMA.

- **Sampling (cumulative, frontier-biased).** Each training batch samples a level
  from the unlocked set `{1..k}`, weighted toward the newest level (the frontier)
  with a floor on earlier levels to prevent forgetting. Simple default weights:
  newest level `0.6`, the rest share `0.4` uniformly (all ≥ a small floor).
- **Advancement (fixed cadence + gate).** Every `N` steps, consider unlocking
  `k+1`:
  - default **fixed cadence**: unlock `k+1` after `N` steps at level `k`;
  - **mastery gate** (the "as it gains mastery" part): only unlock if the EMA
    success on level `k` ≥ `promote_threshold` (e.g. 0.7); otherwise spend
    another `N` steps at `k` (don't advance into a wall). A `max_holds` cap stops
    a stuck level from blocking forever.
- **Graduation (optional).** If level `k`'s success ≥ `graduate_threshold` (e.g.
  0.95) and stable, drop its weight to the floor so gradient flows to harder
  levels (marin's graduation, simplified).

This is ~80 lines, deterministic, and checkpoint-free (state is `(k, step,
ema[])`). It captures marin's "train at the frontier, advance on mastery" without
the DAG/plateau-regression machinery.

### 4. Exploration (break the empty-collapse — cause 3)

- **CoT prompt**: "Reason briefly, then write `def solve(...)` ending with END."
  A few-shot demo that *reasons then codes* so the policy always emits a full,
  diverse attempt instead of collapsing to empty.
- Reward `ran_ok` (0.2) so wrong-but-running programs are pulled off the empty
  attractor; partial-test reward (0.7·frac) gives gradient before any full pass.
- Sampling `temperature=1.0`; consider a small entropy bonus if collapse persists.

### 5. Train + measure (so a win is visible)

- **SFT warm-up**: light, on **levels 1–2 only** (format + the `def solve` + CoT
  shape) — enough to make the policy explore, *not* enough to memorise higher
  levels. Keeps the higher-level frontier open for RL.
- **Dr.GRPO**: curriculum-sampled batches, graded test-case reward, multi-turn
  with failing-test feedback.
- **Eval = the headline**: per level, **pass@1 and pass@k on HELD-OUT instances**
  (fresh seeds, unseen params) plus an **SFT-only control at matched compute**.
  A clear win = RL pass@1 climbs above the SFT-only plateau on the mid/high levels
  (the "RL generalises where SFT memorises" signature), and the frontier advances
  through more levels than SFT alone.

## Implementation plan (files)

- `coding_problems.py` *(new)* — `Problem`, the leveled `solve` families + oracles
  + generators, the test-case grader and dense reward, `sample_problem(level)`,
  `load_eval_problems()` (fixed held-out instances per level). CPU self-check:
  every family's oracle passes its own tests; random instances grade 1.0 for the
  oracle and <1.0 for a mutated program.
- `curriculum.py` *(new)* — the fixed-cadence + mastery-gate scheduler above, with
  a unit self-check (advances on cadence, holds below threshold, never forgets).
- `train_curriculum.py` *(new, or extend `train_multiturn`)* — graded-reward env +
  curriculum batch sampling + CoT prompt; reuses the agentic Dr.GRPO wiring, the
  `END` stop, and the `evaluate_passk` instrument (per level, held-out instances).
- `launch_curriculum.py` *(new)* — `CURRIC_*` env knobs (levels, N, thresholds,
  SFT steps, k).
- Reuse unchanged: `micropython`, `agentic_common`/`agentic_sft`, the Dr.GRPO
  config, `delphi_qwen3`.

## Experiment ladder

1. **Smoke** — graded reward + curriculum end-to-end on a tiny run (few steps,
   2 levels): confirm reward variance is non-zero and the scheduler advances.
2. **Headline** — SFT(L1–2) → curriculum Dr.GRPO vs SFT-only control at matched
   steps; eval pass@1/pass@k on held-out instances per level.
3. **Sweep** — N (steps/level), promote_threshold, #levels, temperature; free
   v6e jobs in parallel.

Success criterion: a level (and ideally several) where **RL pass@1 on held-out
instances is clearly above the SFT-only control** — the clear win.
