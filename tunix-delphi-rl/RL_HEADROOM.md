# Where is the RL headroom? Refining the Delphi coding task so Dr.GRPO earns its keep

A research note for the `tunix-delphi-rl` coding experiments (#7 single-turn, #8
multi-turn). Goal: diagnose *why* SFT saturates our coding task, ground that in
the 2023–2026 RL-vs-SFT literature, and propose **concrete, implementable**
refinements to our task space / reward / protocol that would open a genuine RL
gap — plus an honest note on the chance that, at 447M, none of them will.

All file/line references are to this directory. Numbers ("SFT-300 17/18", "few-shot
3/50 → SFT 50/50") are from `REPORT.md` §9/§11 and `AGENTS.md`.

---

## 0. TL;DR

- **Why SFT saturates us:** our task has **no exploration gap**. After SFT,
  pass@1 ≈ pass@k on the eval set — there is essentially nothing for RL to
  *concentrate mass onto* that SFT hasn't already put at the mode. The dominant
  recent finding (Yue et al., "Invisible Leash", spurious-rewards) is that RLVR's
  primary effect is to **compress pass@k into pass@1** — it raises reliability,
  not capability, *within* the base/SFT support. We removed the very signature
  RL exploits.
- **Two extra structural defects** kill RL on top of that: (a) our reward can
  only see **crashes and obvious prefix mismatches**, never a clean-but-wrong
  answer (the gold is a hidden string, single test); (b) our task families have
  **canonical single solutions** (`fib`, `gcd`) — tiny solution space, so pass@1
  is already near pass@k after SFT.
- **The reframe the brief asks for is correct and is the spine of the fix:** stop
  trying to make RL teach a *new* program. Instead **engineer tasks where SFT
  pass@1 is LOW but pass@k is HIGH** (the base/SFT model *can* produce a correct
  program, just not reliably / not first), then measure **pass@1 climbing toward
  a fixed pass@k ceiling**. That is the regime where every camp agrees RL wins
  (ProRL, RLEF, MURPHY, even Yue et al. as the *defender* of "RL = sample
  efficiency").
- **Highest-signal first experiment:** add a **multi-test, hidden+public reward**
  (so logic bugs are observable and reward-hackable shortcuts are punished) on a
  **new "hard, high-pass@k-low-pass@1" tier**, and report the pass@1→pass@k gap
  before vs after Dr.GRPO. Effort: ~1 day, almost all in `coding_env.py` /
  `coding_agent_env.py`.
- **The honest risk:** at 447M we are *below* the scale where the long-CoT/RL
  literature reliably sees gains ("Through the Valley": a degradation "valley"
  for sub-2B models). It is plausible that *no* reasonable refinement makes RL
  cleanly beat SFT here. That itself is a publishable, decision-relevant result
  (see §6).

---

## 1. Diagnosis: why SFT saturates, in the language of the literature

### 1.1 The core mechanism — RLVR compresses pass@k into pass@1

The single most-cited 2025 result on this question is **Yue et al., "Does RL
Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?"** (NeurIPS
2025, arXiv:2504.13837). Their finding: RLVR-trained models beat the base model at
**small k (pass@1)** but the **base model overtakes them at large k (pass@256+)**,
and coverage/perplexity analyses show "all correct solutions from RL-trained
models already exist in the base model's distribution" — RLVR "enhances sampling
efficiency" while "inadvertently shrinking the solution space." The mechanism is
made precise by **"The Invisible Leash" (Wu et al., arXiv:2507.14843)**: RLVR
"cannot sample solutions with zero initial probability" and acts as "a
conservative reweighting mechanism"; token-level entropy may rise but
**answer-level entropy falls** — the policy "converges onto a smaller set of
distinct answers." The community shorthand (advantage-design surveys, Pass@k
Training, arXiv:2508.10751) is blunt: **GRPO compresses pass@k accuracy into
pass@1.**

Now apply this to us. After SFT-300, tier-5 multi-turn is **first 17/18 == best
17/18** (`REPORT.md` §11) and single-turn SFT is **50/50** (§9). The model
*one-shots* the task. Concretely: **pass@1 ≈ pass@k ≈ ceiling after SFT.** RL's
entire documented value — moving probability mass from "right but rare" toward
"right and first" — has *no mass to move*, because SFT already put the correct
program at the mode. We didn't find that "RL doesn't work"; we built a task whose
**exploration gap is ~0**, which is exactly the condition under which the whole
literature predicts RL is marginal. This is the same wall `AGENTS.md` records
("RL only sharpens what the base policy already puts mass on") — stated as a
*measurable signature*: **the SFT pass@1-to-pass@k gap is the RL headroom**, and
ours is closed.

### 1.2 The second defect — the reward can't see logic bugs

Our reward (`coding_env.code_reward`, mirrored in `coding_agent_env`) is
`0.1*has_code + 0.3*ran_ok + 0.6*partial + 1.0*exact`, graded against **one hidden
gold string** via longest-common-*prefix* overlap (`_output_similarity`). This
means a **clean-running-but-wrong** program gets a smooth signal *only if its
output shares a prefix with the gold*; a subtly wrong program with a
divergent-but-plausible output (off-by-one in the middle, wrong separator after
char 1, wrong branch) is graded almost like a crash. In the multi-turn loop the
agent is fed this same stdout as `Tool result:` — so as the brief notes, it can
only iterate on **(a) crashes and (b) spec-obvious prefix errors**, never "this
ran cleanly but is logically wrong." This *caps the value of iteration*, which is
the only thing multi-turn RL has to amplify. The program-repair RL literature
(RLEF; MURPHY, arXiv:2511.07833) gets its signal precisely from **rich
per-test-case execution feedback** that localizes *which* behavior is wrong; we
throw that away.

### 1.3 The third defect — single canonical solutions ⇒ no pass@k spread

Our families (`coding_env.FAMILIES`) are **templated programs with one canonical
form** (`def fib...`, Euclidean `gcd`, the FizzBuzz ladder). The space of
*correct* programs the model could find is tiny and SFT demonstrates it directly.
So even before measuring, we expect pass@k to be flat in k after SFT: there is one
way to be right and SFT taught it. Chu et al. ("SFT Memorizes, RL Generalizes",
ICML 2025, arXiv:2501.17161) show the flip side — RL's advantage shows up as
**generalization to unseen rule/visual variants** where SFT overfits the training
rule. Our eval is *in-distribution with the training families* (same task types,
just held-out params), so we are testing exactly the axis where SFT is strong and
RL is redundant.

### 1.4 The reframe — yes, reframe around pass@1 ↑ toward a pass@k ceiling

The brief's hypothesis is right and is the organizing principle for everything
below. If RL mostly concentrates mass on modes the base already samples, then a
*demonstration that RL helps* must be a task where:

> **base/SFT pass@1 is LOW, but pass@k (k≈16–64) is HIGH** — the correct program
> *is* in the model's support (so RL is allowed to reach it), it is just not
> reliable / not first. RL's job, and its measurable win, is **pass@1 climbing
> toward the pass@k ceiling** without collapsing pass@k.

Every camp endorses *this* framing of an RL win, even the skeptics: Yue et al.
*concede* RL improves sample efficiency; "Invisible Leash" concedes "RLVR
consistently improves pass@1." The disagreement is only about whether RL can
exceed the pass@k ceiling — and we do **not** need it to. We just need a real gap
to close. So the deliverable metric changes from "solve N/50 greedy" to a
**(pass@1, pass@k) pair per tier, before and after RL**, with success = pass@1
rises toward pass@k and the pass@k ceiling does not drop. (Caveat from Pass@k
Training and "Beyond Pass@1"/SvS, arXiv:2508.14029 and 2508.10751: naive GRPO can
*lower* pass@k while raising pass@1 — so we must *watch* pass@k, not just pass@1.)

---

## 2. Prioritized refinements (each: what / why-with-citation / expected
pass@1-vs-pass@k signature / effort in our code)

Ordered by signal-per-unit-effort. R1+R2 are the recommended first cut; they
compose.

### R1 — Multi-test reward with hidden + public cases (make logic bugs observable)

**What.** Replace the single hidden-gold grade with **N input→output test cases
per task** (trivial: our families already compute golds via the
`micropython` oracle — just sample *several* parameterizations of the *same*
family as the test suite for one task). Reward becomes a **fraction of tests
passed** (dense, continuous in [0,1]) instead of all-or-nothing exact. Show the
agent a **public subset** (e.g. 2 of 6) of (input, expected) in the prompt and
**hide the rest** for grading. In multi-turn, feed back *per-public-test*
pass/fail + the failing input and the produced-vs-expected output as the
`Tool result:` — so the agent acts on **logic-bug feedback**, not just crashes.

**Why it creates RL headroom (cited).**
- A fraction-of-tests reward is the canonical RLVR code signal that makes
  *correctness gradient* dense and **localizes which behavior is wrong** — the
  mechanism RLEF / MURPHY exploit to beat single-shot and SFT
  (MURPHY, arXiv:2511.07833: multi-turn GRPO with execution feedback improves
  pass@1 *and* pass@k, "smaller models particularly benefit … additional
  opportunities to correct initial mistakes"). This directly removes our §1.2
  defect.
- **Hidden vs public split punishes the shortcut SFT/constant-printing relies
  on.** With a single fixed gold, a model can pass by matching surface; with
  hidden tests, "passing visible tests but failing hidden ones" is the standard
  reward-hacking signal (HumanEval visible/hidden splits; "Is the Cure Still
  Worse Than the Disease?" test-overfitting, arXiv:2511.16858; HardTests,
  arXiv:2505.24098). RL on hidden-test reward must find a *general* program — the
  same anti-hardcode logic our random params already use, now at the
  test-case level. (Watch for reward hacking of the *interpreter* — see
  "LLMs Gaming Verifiers", arXiv:2604.15149; our sandbox blocks `import`/IO so the
  classic "overwrite the test" hacks are out, but keep the verifier oracle-side.)

**Expected pass@1-vs-pass@k signature.** With a *dense fraction-passed* reward on
tasks where some tests are easy and some are edge cases, SFT will pass the easy
tests one-shot but miss edge cases ⇒ **pass@1 (all-tests) < pass@k**, because the
model *sometimes* samples the edge-case-correct program. RL should raise the
all-tests pass@1 toward pass@k. The per-test reward gives non-zero advantage even
when *no* sample is fully exact (every group member differs on partial credit) —
fixing the cold-start "all-equal-reward ⇒ no gradient" failure §5/REPORT documents
for wide answer spaces.

**Effort.** Low–medium (~0.5–1 day). `coding_env.py`: `sample_task` already
returns `(prompt, solution, gold)`; add `sample_task_suite(rng, family, n_tests)`
returning a list of `(input_params, gold)` for one family, render a public subset
into the prompt, store the hidden golds in the `answer` column (as JSON). Rewrite
`grade_program`/`code_reward` to run the program against each hidden test and
return fraction-passed (the program must read inputs — see note). `coding_agent_env`:
`_format_feedback` and `_execute_tool_calls` already have the seam to inject
richer feedback; change the `Tool result:` body to a per-public-test report.
**One real design choice:** our tasks currently *bake params into the prompt text*
("Print fib(10)"), so "multiple test cases" means either (i) the program reads a
value from a convention (e.g. a leading `n = …` line the env supplies — but
micropython has no `input()`), or (ii) the task asks for a **function** and the
env calls it with several args (cleanest — define the contract "define `f(...)`,
do not print" and have the *grader* call `f` on hidden inputs). Option (ii) is the
better fit and is itself refinement R4.

### R2 — A "hard tier" engineered for low pass@1 / high pass@k

**What.** Add task families whose **correct-program space is broad and whose
first-attempt success is low but k-attempt success is high** for a *post-SFT*
Delphi. Levers that raise pass@k-minus-pass@1 without leaving the model's support:
- **Many small independent sub-decisions** (e.g. multi-rule classification: a
  FizzBuzz with 4–5 interacting divisor rules; a grading scheme with 6 bands and
  edge ties) — the model gets *most* branches right per sample but rarely *all*,
  so repeated sampling covers the space.
- **Easy-to-state, finicky-to-emit output formats** (exact separators, trailing
  spaces, joined vs newline) — our §9.4 misses were exactly format-precision
  (`Hello, World!` casing; countdown). These are high-variance: sometimes right,
  often off — classic pass@1 << pass@k.
- **Search-y tasks with cheap verification** (find the smallest n with property P;
  decode a short cipher; satisfy a small constraint set) — "exploration is hard,
  verification is cheap" is the RLVR sweet spot (the RLVR-for-code/math
  literature broadly; AceReason-Nemotron, arXiv:2505.16400).

**Why (cited).** **ProRL (Liu et al., arXiv:2505.24864)** is the strongest
counter to the "RL can't expand the boundary" camp, and its key conditional is
exactly our design target: RL expands the pass@k boundary **most on tasks where
the base model is WEAK** ("scenarios where base models fail entirely … gains
correlate with task competence of the base model and training duration"). So we
want tasks that are *near* the SFT model's competence frontier — solvable
sometimes (high pass@k) but not reliably (low pass@1) — not tasks it one-shots
(current ladder) nor tasks it never solves (would give RL no gradient, our §5
wide-answer-space stall). Chu et al. add: bias toward **rule variation** at eval
so RL's *generalization* edge over memorizing-SFT can show.

**Expected signature.** This is the tier where we *expect to see* SFT pass@1 ≈
40–70% but pass@16 ≈ 85–95% — a real gap — and Dr.GRPO lifting pass@1 by 10–30
points toward that ceiling. If instead pass@1 ≈ pass@k even here, the model is
either one-shotting (too easy) or never sampling correct (too hard / out of
support) — both diagnostic, see §6.

**Effort.** Low (~0.5 day) — pure data work in `coding_env.FAMILIES`, the exact
mechanism `AGENTS.md` "How to add a coding task / family" describes; the CPU
self-check (`uv run python coding_env.py`) already gates oracle-solvability.

### R3 — Measure (pass@1, pass@k) explicitly; make it the headline metric

**What.** Add a **sampled** eval (temperature > 0, k samples) alongside the greedy
one. Report **pass@1 (or greedy-solve), pass@k, and the gap** per tier,
**before SFT-only, after SFT, after RL**. In multi-turn, also report
first-attempt vs best-across-rounds (we already log this) — note best-across-N-
rounds is a *sequential* pass@N and a legitimate "ceiling" proxy.

**Why (cited).** The entire RL-vs-SFT debate is *defined* on the pass@k curve
(Yue et al.; Invisible Leash; ProRL). Reporting only greedy-solve makes our result
uninterpretable in that frame: "Dr.GRPO 46→45" could be "RL is useless" *or* "RL
moved pass@1 up but we measured at the ceiling." Pass@k Training
(arXiv:2508.10751) and SvS (arXiv:2508.14029) both warn naive GRPO can raise
pass@1 while *lowering* pass@k — we cannot see that without measuring it.

**Expected signature.** This is the *instrument*, not a task change: it lets every
other refinement be judged by the right signature (pass@1 ↑, pass@k flat-or-up =
win; pass@1 ↑, pass@k ↓ = mode collapse / over-sharpening = invariant-C-style
over-collapse but for RL).

**Effort.** Low (~0.5 day). `coding_env.greedy_completions` already wraps the
tunix `Sampler`; add a `sampled_completions(..., temperature=t, k=n, seed=...)`
and a pass@k aggregator over `evaluate_completions`. Our per-call rollout-seed
machinery (`install_per_call_rollout_seed`) is the same diversity fix we'd need.

### R4 — Function-contract tasks graded on held-out inputs (multi-test generalization)

**What.** Shift the action from "print the answer for *this* instance" to
"**define a function `f` meeting this spec**; do not print." The grader (oracle)
calls `f` on **many hidden inputs** and scores fraction-correct. This is R1's
clean implementation and a *task-type* change.

**Why (cited).** This is the purest "RL generalizes, SFT memorizes" setup
(Chu et al.): SFT can memorize the demonstrated `f` for demonstrated inputs;
generalizing to **unseen inputs** is where outcome-reward RL pulls ahead. It also
matches how real code RLVR is graded (function + hidden tests: HumanEval+/MBPP+,
HardTests) and gives the densest possible correctness gradient (fraction of inputs
correct), maximizing per-group advantage variance on a tiny actor.

**Expected signature.** SFT will pass demonstrated/easy inputs but fail edge
inputs (n=0, empty list, boundary) ⇒ pass@1-on-full-suite < pass@k; RL closes it.
This *also* exposes reward-hack attempts (a function that special-cases the public
inputs) via the hidden inputs — turning the hidden/public split into the
generalization probe.

**Effort.** Medium (~1 day). Needs a small **calling convention** in the
interpreter path: instead of running the program's stdout, the grader runs
`program + "\nprint(f(<input>))"` per hidden input (micropython has no module
import, but we control the source we run — appending a call line is trivial and
stays in-sandbox). New family bodies (`def f(...)` form), new few-shot demos
(`CODE_FEWSHOT` / `CODE_AGENT_SYSTEM_PROMPT`) — must keep train/RL prompt match
(invariant D). Highest-fidelity to the literature; slightly more plumbing.

### R5 — Debugging / repair as the task (not "write from scratch")

**What.** A **repair tier**: the prompt contains a *buggy* program (an off-by-one,
a wrong operator — we *already generate these* in `coding_agent_env._mutate_to_bug`!)
plus its wrong `Tool result:`, and the task is to emit the **fixed** program. Pair
with the multi-turn loop so the agent iterates.

**Why (cited).** Repair-with-execution-feedback is the clearest documented case of
RL beating SFT/single-shot: **RLEF** (RL from execution feedback) and **MURPHY**
report RL-trained multi-turn repair beating single-shot/SFT, with smaller models
benefiting *most* from extra correction turns (arXiv:2511.07833); the agentic-RL
survey (arXiv:2509.02547) and real-world code-repair RL (arXiv:2510.22075) concur.
Crucially, *which fix to apply given which error* is a **narrow, feedback-
conditioned behavior** — exactly the "narrow behavior amplified from rare base
samples" regime where our own §8 (CALC copy) showed RL is *essential*. This is the
§8↔§9 inversion `AGENTS.md` predicts for multi-turn.

**Expected signature.** First-attempt (= no-repair) solve LOW; best-across-rounds
HIGH; SFT can demonstrate the *form* of a fix but not the *policy* (which fix for
which error), so SFT's first→best gap is small and RL grows it. This is the metric
we already log (`code_agent_metric_fn`'s `first_solve` vs `solve_ratio`).

**Effort.** Low–medium (~0.5–1 day). The bug-generation
(`_mutate_to_bug`), the multi-turn harness, and the first-vs-best metric **already
exist**. New work: a repair *prompt* family (buggy program in the prompt), a
matching few-shot demo, and ideally R1's richer feedback so the repair is
guided by *which test* failed.

### R6 (lower priority) — Reward-shape and exploration knobs, only after R1–R3

**What.** Once a real gap exists: tune for exploration — higher rollout
temperature, larger `num_generations`, entropy bonus / clip-higher
(DAPO-style), and *watch pass@k*. Consider a pass@k-aware objective
(arXiv:2508.10751) or SvS-style problem synthesis (arXiv:2508.14029) if naive
Dr.GRPO collapses pass@k.

**Why (cited).** "Exploration vs Exploitation: Rethinking RLVR through Clipping,
Entropy, and Spurious Reward" (arXiv:2512.16912) and Pass@k Training show the
exploration/exploitation balance is what governs whether RL raises pass@1 *without*
crushing pass@k. But these are **second-order** — useless until R1/R2 create a gap.

**Effort.** Low per knob (env vars / `GRPOConfig` fields already exposed in
`train_multiturn.py`), but it's a sweep, and TPUs are free, so parallelize.

---

## 3. The single best next experiment (highest signal, lowest effort)

**Run R3 first as pure instrumentation, then R1+R2+R5 together as one task
upgrade.** Concretely, one experiment:

> **"Hidden-test, multi-test, repair-augmented hard tier, measured on the pass@k
> curve."**

1. **Instrument (R3):** add sampled pass@k eval (k=16, temperature 0.8) to
   `coding_env`/`coding_agent_env`; report (pass@1, pass@16) per tier for
   few-shot, SFT, SFT→Dr.GRPO. *Do this even on the current ladder first* — it
   will quantify the "no gap" diagnosis (expected: pass@1 ≈ pass@16 ≈ ceiling on
   tiers 0–4) and validate the instrument cheaply.
2. **New tier 6 (R1+R2+R4):** ~6 **function-contract** families with **6 hidden +
   2 public tests** each, chosen for low-pass@1/high-pass@k (multi-rule classify,
   finicky format, small search). Reward = fraction of hidden tests passed
   (dense). The CPU self-check must show the oracle solves them and that a
   *plausible buggy* variant fails ≥1 hidden test.
3. **Repair variant (R5):** for the same tier, a multi-turn run where round-1 is
   often wrong and the per-public-test feedback drives the fix.
4. **Read the signature:** success = on tier 6, SFT shows pass@1 ≪ pass@16, and
   Dr.GRPO lifts pass@1 toward pass@16 **without** dropping pass@16; multi-turn
   first→best gap grows under RL.

**Config knobs (start here, then sweep on free TPUs):**
- Task: `MT_TIERS=6 MT_EVAL_TIERS=6`, 6 families × (6 hidden, 2 public) tests.
- SFT: `MT_SFT_STEPS` in 300–800 (broad target ⇒ monotone, per invariant C);
  `sft_fix_prob ≈ 0.3` (keep repair rare so RL amplifies it).
- RL: Dr.GRPO `steps≈120`, `num_generations=16`, `temperature=1.0`,
  `learning_rate=1e-5` (clipped — invariant B), `beta=0`, `rounds=5`.
- Eval: greedy + sampled pass@16 @ T=0.8, both before and after RL.
- Reward weights: keep dense climb (`has_code`/`ran_ok`) but make the **exact
  term a fraction-of-hidden-tests** so partial correctness is graded by
  *behavior*, not output-prefix overlap.

This reuses ~90% of existing plumbing (interpreter, families, SFT warm-up,
multi-turn env, first-vs-best metric, the bug mutator) — the new code is the
multi-test grader, the public-test prompt rendering, and the pass@k eval.

---

## 4. Which task TYPES are more RL-favorable than "write a templated program"

Ranked by how directly the literature ties them to an RL>SFT gap:

1. **Repair / debugging with execution feedback** (R5) — RLEF, MURPHY,
   agentic-RL survey: the canonical RL>SFT/single-shot result; "which fix for
   which error" is a feedback-conditioned policy SFT can't supply. *Best fit to
   our §8 inversion + existing harness.*
2. **Function-contract + hidden multi-test generalization** (R1/R4) — Chu et al.
   ("RL generalizes to unseen inputs/rules; SFT memorizes"); HumanEval+/HardTests
   grading. Densest correctness gradient.
3. **Constraint-satisfaction / small search with cheap verification** (R2) — the
   RLVR sweet spot (exploration hard, verification cheap); ProRL shows the
   boundary expands most where the base is weak-but-nonzero.
4. **Multi-rule / many-branch tasks** (R2) — maximize pass@k-minus-pass@1 by
   making "all branches right at once" rare per sample but reachable across
   samples.

Anti-pattern (our current ladder): **single-canonical-solution, single-test,
in-distribution** templated programs — minimal solution space, no exploration gap,
no logic-bug feedback. The literature predicts (and we observed) SFT saturation.

---

## 5. Summary table — refinement × headroom mechanism

| # | Change | Headroom mechanism (cite) | pass@1 vs pass@k signature | Effort |
|---|---|---|---|---|
| R1 | Hidden+public multi-test reward (fraction passed) | dense correctness gradient + anti-shortcut (MURPHY 2511.07833; test-overfitting 2511.16858; HardTests 2505.24098) | SFT pass@1<pass@k via edge-case tests; RL closes gap; non-zero advantage pre-exact | Low–Med |
| R2 | Hard tier: low pass@1 / high pass@k (multi-rule, finicky format, small search) | RL expands boundary where base is weak-but-nonzero (ProRL 2505.24864); RLVR sweet spot | creates the gap SFT can't close one-shot | Low |
| R3 | Measure (pass@1, pass@k) as headline | the debate is defined on pass@k (Yue 2504.13837; Invisible Leash 2507.14843) | the *instrument* for all others; catches pass@k collapse (Pass@k Training 2508.10751) | Low |
| R4 | Function-contract, held-out inputs | RL generalizes / SFT memorizes (Chu 2501.17161) | generalization gap to unseen inputs; exposes reward hacks | Med |
| R5 | Repair/debugging as the task | RL from execution feedback beats single-shot/SFT (RLEF; MURPHY); narrow feedback-conditioned behavior (our §8) | first-attempt low, best-across-rounds high, RL grows the gap | Low–Med |
| R6 | Exploration knobs (temp, entropy, pass@k objective) | exploration/exploitation governs pass@1-vs-pass@k (2512.16912; SvS 2508.14029) | tune pass@1 ↑ without pass@k ↓ | Low (sweep) |

---

## 6. Honest risk: maybe nothing makes RL beat SFT at 447M — and what that teaches

There is a real chance that **no** refinement above produces a clean RL>SFT result
*at this model size*, for reasons the literature names:

- **Scale floor.** "Through the Valley" (arXiv:2506.07712) finds a **degradation
  "valley" for sub-2B models** on long-CoT/RL training — small models "lack
  sufficient capacity to … learn from extended reasoning sequences." Delphi is
  **447M** and **1.2B tokens** — well below that floor. The boundary-expansion
  camp (ProRL) needs the base to be *weak-but-nonzero* on a task; if Delphi's
  pass@k is **0** even at k=64 on anything genuinely hard, RL has no support to
  reweight (Invisible Leash: "cannot sample solutions with zero probability") and
  no gradient (our own §5 wide-answer-space stall). The window "pass@1 low, pass@k
  high" may be *narrow or empty* for a 447M model: tasks are either one-shot
  (pass@1≈pass@k≈1, no gap) or never-solved (pass@k≈0, no gradient), with little
  in between.
- **What we should pre-commit to measuring.** The pass@k instrument (R3) makes
  the failure *diagnostic, not null*. Three distinguishable outcomes:
  1. **Gap exists and RL closes it** → the intended positive result.
  2. **Gap exists, RL does NOT close it (or collapses pass@k)** → an *algorithm*
     result: Dr.GRPO under-exploits on a tiny actor; motivates R6 /
     pass@k-aware objectives. Still publishable.
  3. **No gap exists at any reachable difficulty** (everything is one-shot or
     unsolved) → a *scale* result: at 447M the exploration gap RL needs doesn't
     open; **SFT-coverage is the right lever and RL is structurally moot here.**
     This *confirms and sharpens* the project's cross-experiment rule with a
     mechanism ("no pass@1<pass@k window at this scale"), and is a clean,
     honest, decision-relevant finding for marin: *don't spend RL compute on
     sub-1B coders; spend it on SFT coverage.*

In all three cases the experiment is worth running because the **pass@k
measurement converts our current ambiguous "RL marginal" into a mechanistic
statement**. The cheapest first move (R3 on the existing ladder) already pays for
itself by quantifying outcome (3)'s "no gap" baseline before we build any new tier.

---

## Sources

- Yue et al., *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* — https://arxiv.org/abs/2504.13837 ; project page https://limit-of-rlvr.github.io/
- Chu et al., *SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training* (ICML 2025) — https://arxiv.org/abs/2501.17161
- Wu et al., *The Invisible Leash: Why RLVR May or May Not Escape Its Origin* — https://arxiv.org/abs/2507.14843
- Liu et al., *ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models* — https://arxiv.org/abs/2505.24864
- *MURPHY: Multi-Turn GRPO for Self Correcting Code Generation* — https://arxiv.org/abs/2511.07833
- *Pass@k Training for Adaptively Balancing Exploration and Exploitation of Large Reasoning Models* — https://arxiv.org/abs/2508.10751
- *Beyond Pass@1: Self-Play with Variational Problem Synthesis Sustains RLVR* (SvS) — https://arxiv.org/abs/2508.14029
- *Through the Valley: Path to Effective Long CoT Training for Small Language Models* — https://arxiv.org/abs/2506.07712
- *HardTests: Synthesizing High-Quality Test Cases for LLM Coding* — https://arxiv.org/abs/2505.24098
- *Is the Cure Still Worse Than the Disease? Test Overfitting by LLMs in Automated Program Repair* — https://arxiv.org/abs/2511.16858
- *LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking* — https://arxiv.org/abs/2604.15149
- *Exploration vs Exploitation: Rethinking RLVR through Clipping, Entropy, and Spurious Reward* — https://arxiv.org/abs/2512.16912
- Spurious Rewards: Rethinking Training Signals in RLVR — https://unknown-nlp.github.io/blog/2025/spurious-rewards-rethinking-training-signals-in-rlvr/
- *The Landscape of Agentic Reinforcement Learning for LLMs: A Survey* — https://arxiv.org/abs/2509.02547
- *Agentic Reinforcement Learning for Real-World Code Repair* — https://arxiv.org/abs/2510.22075
- *AceReason-Nemotron: Advancing Math and Code Reasoning through Reinforcement Learning* — https://arxiv.org/abs/2505.16400
