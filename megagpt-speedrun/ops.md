# megagpt-speedrun — ops runbook

Operational notes for running the H100x8 jobs on `cw-us-east-02a`. Keep this current
when something bites us.

## The `job` command

The interactive `job` alias is:

```sh
KUBECONFIG=~/.kube/coreweave-iris-gpu uv run iris --cluster=cw-us-east-02a job <subcmd>
```

- Listing is `job list` (NOT `job ls` — that prints nothing).
- `timeout` cannot wrap it (it's a function, not a binary) → `timeout: No such file or directory`.
- `job kill` is **prefix-match** — always pass the full `/power/<name>` so you don't take out
  a sibling (`megagpt-24h-pre` is a prefix of `megagpt-24h-pre-cool`, etc.).
- Worker log timestamps are **UTC**; your laptop may be UTC-4. Don't try to reconcile the
  data-loader "N seconds" timer against submit time — it counts from when the loader thread
  started (after process init / during compile), not from step 0.

## MONITOR FOR SILENT DATA-LOADING HANGS  ← the one that bit us

**Symptom.** The log fills with, every 10s, a climbing counter and *no* `train …it` progress:

```
W… root Data loading is taking a long time: 14480.0 seconds. Waiting for 256 items.
```

**What it is.** A stuck data fetch (a dead Ray data-fetch actor, or an R2 object read that
hangs with no timeout). The loader prefetches ahead, so the job *steps for a while* (we got to
~step 284) off the buffer, then blocks forever on the one batch that never arrives.

**Why it's dangerous.** The job stays in state `running` and **nothing crashes**, so
`--max-retries` does NOT recover it. It will sit there burning all 8 H100s indefinitely. A
status check that only looks at job state (or only greps for `Fatal`/tracebacks) will report
"healthy" — this is exactly why our first watcher missed it and we lost ~4h of node time.

**Detection (do this, not just job-state checks).** Track the **step number** over time:
- First `train …it` line should appear within ~20 min (compile). If no step after ~30 min → startup hang.
- Once stepping, the step advances ~2.4 it/s. If the **max step is frozen for >~15 min** while
  state is still `running` → HANG. (Checkpoint saves only pause stepping for ~1-2 min every
  30 min, so a 15-min threshold won't false-positive on those.)
- The climbing `Data loading is taking a long time` line is the confirming fingerprint.

The watcher script `watch_pretrain.sh` (in the session scratchpad) implements this: it exits
(notifying the operator) on completion / crash / step-stall-hang. Always run a **step-progress**
watcher for long runs, not just a crash watcher.

**Recovery.**
1. `job kill /power/<full-name>` (full path — prefix match).
2. Confirm via `job list` it shows `killed` (the node must release before relaunch, or two jobs collide).
3. Relaunch the *identical* command. Same config → same `run_id` → resumes from the last 30-min
   R2 checkpoint. (If the hang happened before the first checkpoint, ~step 4200, it restarts at 0 — cheap.)
4. If it hangs again at ~the same step → suspect a specific bad/missing cache shard (deterministic
   shuffle order). Then investigate the nemotron cache integrity / change the data seed. A transient
   Ray/R2 stall will NOT recur at the same step.

## Other things to watch

- **OOM.** b16 is nondeterministic-OOM at this geometry (15B / E256 / seq4096); **b8** is the
  reliable batch (`SP_FIT_BATCH 8`). `SP_FAST_QB=1` pushes b16 over the edge — leave it off.
  OOM shows as a real crash (`RESOURCE_EXHAUSTED` / NCCL rendezvous hang then exit) → `--max-retries`
  resumes it, but if it OOMs every attempt the retries just churn — drop the batch instead.
- **Loss divergence.** Constant-LR (WSD stable phase) should descend then **plateau** — a plateau is
  expected, not divergence. Divergence = loss climbing / NaN (NaN crashes → caught as a crash). If
  loss climbs without NaN, kill and relaunch with a lower SP_TOKENS (lower peak LR).
- **`RAGGED_DOT_IMPL=triton` is mandatory** (baked into `launch_cw.sh`). Without it the MoE grouped
  matmul falls back to the XLA dense path and OOMs (100s of GiB).
- **CoreWeave is R2-only** (`s3://marin-na`, cannot read `gs://`). R2 creds are auto-injected on the
  worker — do NOT forward them in `-e`.

## Quick health check (one-liner)

```sh
export KUBECONFIG=~/.kube/coreweave-iris-gpu
uv run iris --cluster=cw-us-east-02a job logs /power/megagpt-24h-pre 2>/dev/null | tail -30 \
  | grep -E "train [0-9]+it|loss=|Data loading is taking|Error|Traceback"
```

If you see only `Data loading is taking a long time` and no recent `train …it` → it's hung (see above).
