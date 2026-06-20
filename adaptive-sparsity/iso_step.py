# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iso-step quality<->sparsity table.

collect_results.py reports each run's *final* summary value, but runs whose wandb stream
dropped early (state `crashed`) freeze their summary at a smaller step than runs that
finished, so the final values are not comparable across arms. This pulls full history and
reports train/loss (and realized active fraction) at a common reference step, plus each
run's max logged step, so the frontier is iso-step.

Usage:
    uv run python iso_step.py --group adaptive-sparsity-kmin1 --step 4000
"""

import argparse
import os
import re

import wandb

PROJECT = "marin_moe"
_NAME_RE = re.compile(
    r"sparsity-(?P<mode>fixed|adapt)-d(?P<d>\d+)-E(?P<E>\d+)-k(?P<k>\d+)"
    r"(?:-min(?P<min>\d+)-c(?P<coef>[0-9.]+)-t(?P<temp>[0-9.]+))?"
)
LOSS = "train/loss"
CE = "train/cross_entropy_loss"
ACT = "train/router/sparsity/realized_active_frac"


def _at_step(hist, step, key):
    """Value of `key` at the history row nearest to `step` (<= step preferred)."""
    if hist is None or key not in hist:
        return None, None
    sub = hist[["_step", key]].dropna()
    if not len(sub):
        return None, None
    le = sub[sub["_step"] <= step]
    row = le.iloc[-1] if len(le) else sub.iloc[0]
    return float(row[key]), int(row["_step"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--step", type=int, default=4000)
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    args = ap.parse_args()

    api = wandb.Api()
    entity = args.entity or api.default_entity
    runs = api.runs(f"{entity}/{PROJECT}", filters={"group": args.group})

    rows = []
    for run in runs:
        m = _NAME_RE.search(run.name or "")
        E = int(m["E"]) if m else 0
        k = int(m["k"]) if m else 0
        kmin = int(m["min"]) if (m and m["min"]) else (0 if (m and m["mode"] == "adapt") else k)
        coef = float(m["coef"]) if (m and m["coef"]) else 0.0
        mode = "adaptive" if (m and m["mode"] == "adapt") else "fixed"
        hist = run.history(keys=[LOSS, CE, ACT], pandas=True, samples=20000)
        loss, at = _at_step(hist, args.step, LOSS)
        ce, _ = _at_step(hist, args.step, CE)
        act, _ = _at_step(hist, args.step, ACT)
        if act is None:
            act = (k / E) if E else None
        maxstep = int(hist["_step"].max()) if (hist is not None and len(hist)) else None
        rows.append(dict(mode=mode, E=E, k=k, kmin=kmin, coef=coef, loss=loss, ce=ce,
                         act=act, at=at, maxstep=maxstep))

    rows = [r for r in rows if r["loss"] is not None]
    rows.sort(key=lambda r: (r["act"] if r["act"] is not None else 0))
    print(f"\n### {args.group}  —  train/loss at step ~{args.step} ({len(rows)} runs)\n")
    print("| mode | K_max | K_min | λ | active frac | loss@step | CE@step | (step) | max step |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        act = f"{r['act']:.4%}" if r["act"] is not None else "—"
        ce = f"{r['ce']:.4f}" if r["ce"] is not None else "—"
        print(f"| {r['mode']} | {r['k']} | {r['kmin']} | {r['coef']:g} | {act} | "
              f"{r['loss']:.4f} | {ce} | {r['at']} | {r['maxstep']} |")
    print()


if __name__ == "__main__":
    main()
