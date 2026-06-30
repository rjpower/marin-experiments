#!/usr/bin/env python
"""Throughput-sweep monitor using the iris Python client directly (one persistent controller
tunnel; no CLI subprocess shelling, no bash poll loops).

Primary signal = the per-step ``[THRUPUT]`` lines emitted by train.py's stdout hook (levanter's own
tokens_per_second + mfu from the real step_duration and the device's true peak FLOPs). We report the
MEDIAN over the steady window (step >= --warmup) -- robust and compile-independent.

Usage:
  uv run python monitor.py <prefix> [--gpus N] [--warmup S] [--watch SECONDS] [--cluster NAME]
    e.g.  uv run python monitor.py d3 --gpus 8 --warmup 8
          uv run python monitor.py sw3 --watch 120        # re-poll every 120s until all report
"""
import argparse
import os
import re
import statistics
import sys
import time

os.environ.setdefault("KUBECONFIG", os.path.expanduser("~/.kube/coreweave-iris-gpu"))

from pathlib import Path  # noqa: E402

from iris.client import IrisClient  # noqa: E402
from iris.cluster.composer import provider_bundle  # noqa: E402
from iris.cluster.config import load_config  # noqa: E402
from iris.cluster.types import JobName  # noqa: E402
from iris.cli.connect import IRIS_CLUSTER_CONFIG_DIRS, iap_config  # noqa: E402
from iris.cli.main import client_credentials, resolve_cluster_name  # noqa: E402
from rigging.config_discovery import resolve_cluster_config  # noqa: E402
import iris.rpc.job_pb2 as job_pb2  # noqa: E402

TOK = 2e9  # SP_TOKENS used to form the [arm] budget
H100 = 989.5e12  # per-GPU bf16 peak (only for the ~tqdm fallback MFU)

ARM = re.compile(r"\[arm\].*?D=(\d+).*?seq=(\d+).*?E=(\d+) k=(\d+).*?batch=(\d+).*?budget=([\d.eE+]+)")
THRU = re.compile(r"\[THRUPUT\] step=(\d+) dt=([\d.]+)s tok_s=(\w+) mfu=(\w[\w.]*) bs=(\d+)")
TQDM = re.compile(r"on:train ([\d.]+)it/[\d.]+it rate:([\d.]+)(it/s|s/it)")
BAD = re.compile(r"RESOURCE_EXHAUSTED|out of memory|Segmentation fault|different incarnation|Fatal Python error")


def connect(cluster):
    """Return (client, close_fn) for the named cluster, holding the controller tunnel open."""
    cfg = load_config(resolve_cluster_config(cluster, dirs=IRIS_CLUSTER_CONFIG_DIRS))
    name = resolve_cluster_name(cfg, None, cluster)
    creds = client_credentials(cfg, name)
    iap = iap_config(cfg)
    if iap is not None:
        return IrisClient.remote(iap.url, workspace=None, credentials=creds), (lambda: None)
    bundle = provider_bundle(cfg)
    addr = cfg.controller_address() or bundle.controller.discover_controller(cfg.controller)
    cm = bundle.controller.tunnel(address=addr)
    url = cm.__enter__()
    return IrisClient.remote(url, workspace=None, credentials=creds), (lambda: cm.__exit__(None, None, None))


def fetch_log_text(client, job_id):
    entries = client.fetch_task_logs(JobName.from_wire(job_id), max_lines=4000, tail=True)
    return "\n".join(e.data for e in entries)


def analyze(text, ngpu, warmup):
    arm = None
    for m in ARM.finditer(text):
        D, seq, E, k, b, bud = m.groups()
        arm = dict(D=int(D), seq=int(seq), E=int(E), k=int(k), b=int(b), fpt=float(bud) / (3 * TOK))
    bad = next((m.group(0) for m in BAD.finditer(text)), "")
    tps, mfus = [], []
    for m in THRU.finditer(text):
        if float(m.group(1)) >= warmup:
            if m.group(3) != "na":
                tps.append(float(m.group(3)))
            if m.group(4) != "na":
                mfus.append(float(m.group(4)))
    if tps:
        return arm, statistics.median(tps), (statistics.median(mfus) if mfus else None), "hook", bad
    last = None
    for m in TQDM.finditer(text):
        last = m
    if last and arm:
        step, val, unit = float(last.group(1)), float(last.group(2)), last.group(3)
        itps = val if unit == "it/s" else 1.0 / val
        t = itps * arm["b"] * arm["seq"]
        mfu = (3 * arm["fpt"] * arm["seq"] * arm["b"] * itps) / (H100 * ngpu) * 100
        return arm, t, mfu, "~tqdm", (bad or f"@step{step:.0f}")
    return arm, None, None, "-", (bad or "warming")


def report(client, prefix, ngpu, warmup):
    statuses = client.list_jobs(prefix=f"/power/{prefix}-")
    rows = []
    for st in statuses:
        jid = st.job_id
        state = job_pb2.JobState.Name(st.state).replace("JOB_STATE_", "").lower()
        arm, tps, mfu, src, note = analyze(fetch_log_text(client, jid), ngpu, warmup)
        rows.append((jid.split("/")[-1], state, arm, tps, mfu, src, note))
    rows.sort(key=lambda r: -(r[3] if r[3] else -1))
    print(f"\n{'job':18s} {'state':9s} {'E':>4}{'k':>3}{'seq':>6}{'b':>3} "
          f"{'tok/s':>9} {'MFU%':>6} {'tok/24h':>8} {'src':>6}  note")
    print("-" * 100)
    n_hook = 0
    for job, state, a, tps, mfu, src, note in rows:
        if src == "hook":
            n_hook += 1
        E = a["E"] if a else "?"
        k = a["k"] if a else ""
        seq = a["seq"] if a else ""
        b = a["b"] if a else ""
        if tps:
            t24 = tps * 86400 * 0.85 / 1e9
            ms = f"{mfu:.2f}" if mfu is not None else "na"
            print(f"{job:18s} {state:9s} {E:>4}{k:>3}{seq:>6}{b:>3} "
                  f"{tps:>9,.0f} {ms:>6} {t24:>7.1f}B {src:>6}  {note[:34]}")
        else:
            print(f"{job:18s} {state:9s} {E:>4}{k:>3}{seq:>6}{b:>3} "
                  f"{'':>9} {'':>6} {'':>8} {src:>6}  {note[:34]}")
    print(f"\n({len(rows)} jobs; {n_hook} with hook data; median step>={warmup:.0f}; {ngpu}-GPU; tok/24h@0.85)")
    return rows, n_hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--warmup", type=float, default=8)
    ap.add_argument("--watch", type=int, default=0, help="re-poll every N seconds until all report")
    ap.add_argument("--cluster", default="cw-us-east-02a")
    ap.add_argument("--max-passes", type=int, default=15)
    ap.add_argument("--logs", default="", help="dump recent logs for the single job /power/<prefix> (exact suffix)")
    ap.add_argument("--grep", default="", help="with --logs: only lines matching this regex")
    a = ap.parse_args()
    client, close = connect(a.cluster)
    try:
        if a.logs:
            txt = fetch_log_text(client, f"/power/{a.logs}")
            pat = re.compile(a.grep) if a.grep else None
            for line in txt.splitlines():
                if "WatchTasksAsync" in line or "spmd_partitioner" in line:
                    continue
                if pat is None or pat.search(line):
                    print(line)
            return
        if not a.watch:
            report(client, a.prefix, a.gpus, a.warmup)
            return
        for p in range(1, a.max_passes + 1):
            print(f"\n######## pass {p} @ {time.strftime('%H:%M:%SZ', time.gmtime())} ########")
            rows, n_hook = report(client, a.prefix, a.gpus, a.warmup)
            done = sum(1 for r in rows if r[1] in ("succeeded", "failed"))
            if rows and (n_hook >= len(rows) or done >= len(rows)):
                print("### all jobs reported or finished ###")
                return
            sys.stdout.flush()
            time.sleep(a.watch)
    finally:
        close()


if __name__ == "__main__":
    main()
