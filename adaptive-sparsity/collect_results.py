# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect final sparsity-sweep metrics from wandb into a markdown table.

Pulls every run in a wandb group, extracts the final train loss, cross-entropy, and
realized active-expert fraction, and prints a table sorted by active fraction — the
quality↔sparsity frontier used in the milestone updates and the report.

Usage:
    uv run python collect_results.py --group adaptive-sparsity-baseline
    WANDB_ENTITY=<entity> uv run python collect_results.py --group adaptive-sparsity-aggressive
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


def _final(run, key):
    """Final value of a metric: prefer summary, fall back to the last history row."""
    val = run.summary.get(key)
    if val is not None:
        return val
    try:
        hist = run.history(keys=[key], pandas=True)
        if hist is not None and len(hist) and key in hist:
            series = hist[key].dropna()
            if len(series):
                return float(series.iloc[-1])
    except Exception:
        pass
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--project", default=PROJECT)
    args = ap.parse_args()

    api = wandb.Api()
    entity = args.entity or api.default_entity
    runs = api.runs(f"{entity}/{args.project}", filters={"group": args.group})

    rows = []
    for run in runs:
        m = _NAME_RE.search(run.name or "")
        cfg = run.config or {}
        # Pull geometry from the name first (robust), then fall back to logged config.
        if m:
            mode = "adaptive" if m["mode"] == "adapt" else "fixed"
            E = int(m["E"])
            k = int(m["k"])
            coef = float(m["coef"]) if m["coef"] else 0.0
        else:
            mode = "adaptive" if cfg.get("model", {}).get("adaptive_routing") else "fixed"
            E = int(cfg.get("model", {}).get("num_experts", 0) or 0)
            k = int(cfg.get("model", {}).get("num_experts_per_token", 0) or 0)
            coef = float(cfg.get("model", {}).get("sparsity_loss_coef", 0.0) or 0.0)

        realized = _final(run, "train/router/sparsity/realized_active_frac")
        nominal = (k / E) if E else None
        active = realized if realized is not None else nominal
        rows.append(
            dict(
                name=run.name,
                state=run.state,
                mode=mode,
                E=E,
                k=k,
                coef=coef,
                active_frac=active,
                loss=_final(run, "train/loss"),
                ce=_final(run, "train/cross_entropy_loss"),
                steps=run.summary.get("_step"),
            )
        )

    rows = [r for r in rows if r["active_frac"] is not None]
    rows.sort(key=lambda r: r["active_frac"])

    print(f"\n### group: {args.group}  ({len(rows)} runs)\n")
    print("| arm | mode | E | K | λ | active frac | train loss | cross-entropy | state | step |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        loss = f"{r['loss']:.4f}" if isinstance(r["loss"], (int, float)) else "—"
        ce = f"{r['ce']:.4f}" if isinstance(r["ce"], (int, float)) else "—"
        print(
            f"| {r['name']} | {r['mode']} | {r['E']} | {r['k']} | {r['coef']:g} | "
            f"{r['active_frac']:.4%} | {loss} | {ce} | {r['state']} | {r['steps']} |"
        )
    print()


if __name__ == "__main__":
    main()
