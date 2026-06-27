#!/usr/bin/env python
"""Analyze a throughput sweep. PRIMARY signal = the per-step ``[THRUPUT]`` lines emitted by the
custom stdout hook in train.py (levanter's own per-step tokens_per_second + mfu, computed from the
real step_duration and the device's true peak FLOPs). We take the MEDIAN over the steady window
(step >= warmup) -- robust and compile-independent.

FALLBACK (labelled ~tqdm) for runs predating the hook: the final tqdm ``rate:`` line. The smoothed
tqdm rate at the LAST step (~60) is ~steady-state (compile washed out), but it is a display artifact,
so it is only a rough cross-check -- relaunch with the hook for authoritative numbers.

Usage: uv run python analyze_sweep.py <prefix> [n_gpus] [warmup_step]
"""
import re
import statistics
import subprocess
import sys

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "sw3"
NGPU = int(sys.argv[2]) if len(sys.argv) > 2 else 1
WARMUP = float(sys.argv[3]) if len(sys.argv) > 3 else 8
TOK = 2e9
H100 = 989.5e12  # per-GPU bf16 peak (only used for the ~tqdm fallback MFU)

KCFG = "KUBECONFIG=/home/power/.kube/coreweave-iris-gpu"
CLU = "--cluster=cw-us-east-02a"

ARM = re.compile(r"\[arm\].*?D=(\d+).*?seq=(\d+).*?E=(\d+) k=(\d+).*?batch=(\d+).*?budget=([\d.eE+]+)")
THRU = re.compile(r"\[THRUPUT\] step=(\d+) dt=([\d.]+)s tok_s=(\w+) mfu=(\w[\w.]*) bs=(\d+)")
TQDM = re.compile(r"on:train ([\d.]+)it/[\d.]+it rate:([\d.]+)(it/s|s/it)")
BAD = re.compile(r"RESOURCE_EXHAUSTED|out of memory|Segmentation fault|different incarnation|Fatal Python error")


def jobs():
    out = subprocess.run(f"{KCFG} uv run iris {CLU} job list", shell=True,
                         capture_output=True, text=True).stdout
    ns = set(re.findall(rf"/power/({re.escape(PREFIX)}-\S+)", out))
    return sorted(ns)


def analyze(job):
    out = subprocess.run(f"{KCFG} uv run iris {CLU} job logs /power/{job}", shell=True,
                         capture_output=True, text=True).stdout
    L = [x for x in out.splitlines() if "WatchTasksAsync" not in x and "grpc_status" not in x]
    arm = None
    for x in L:
        m = ARM.search(x)
        if m:
            D, seq, E, k, b, bud = m.groups()
            arm = dict(D=int(D), seq=int(seq), E=int(E), k=int(k), b=int(b), fpt=float(bud) / (3 * TOK))
    bad = next((x.split("| ", 1)[-1][:55] for x in L if BAD.search(x)), "")
    # primary: [THRUPUT]
    tps, mfus = [], []
    for x in L:
        m = THRU.search(x)
        if m and float(m.group(1)) >= WARMUP:
            if m.group(3) != "na":
                tps.append(float(m.group(3)))
            if m.group(4) != "na":
                mfus.append(float(m.group(4)))
    if tps:
        return job, arm, statistics.median(tps), (statistics.median(mfus) if mfus else None), "hook", bad
    # fallback: final tqdm rate
    last = None
    for x in L:
        m = TQDM.search(x)
        if m:
            last = m
    if last and arm:
        step, val, unit = float(last.group(1)), float(last.group(2)), last.group(3)
        itps = val if unit == "it/s" else 1.0 / val
        t = itps * arm["b"] * arm["seq"]
        mfu = (3 * arm["fpt"] * arm["seq"] * arm["b"] * itps) / (H100 * NGPU) * 100
        note = bad or (f"~tqdm@{step:.0f}" if step >= 40 else f"~tqdm@{step:.0f}(early!)")
        return job, arm, t, mfu, "~tqdm", note
    return job, arm, None, None, "-", (bad or "warming")


def main():
    rows = [analyze(j) for j in jobs()]
    rows.sort(key=lambda r: -(r[2] if r[2] else -1))
    print(f"\n{'job':20s} {'E':>4}{'k':>3}{'seq':>6}{'b':>3} {'tok/s':>9} {'MFU%':>6} {'tok/24h':>9} {'src':>6}  note")
    print("-" * 92)
    for job, a, tps, mfu, src, note in rows:
        if not a:
            print(f"{job:20s} {'?':>4}{'':>3}{'':>6}{'':>3} {'':>9} {'':>6} {'':>9} {src:>6}  {note}")
            continue
        if tps:
            t24 = tps * 86400 * 0.85 / 1e9
            ms = f"{mfu:.2f}" if mfu is not None else "na"
            print(f"{job:20s} {a['E']:>4}{a['k']:>3}{a['seq']:>6}{a['b']:>3} "
                  f"{tps:>9,.0f} {ms:>6} {t24:>8.1f}B {src:>6}  {note}")
        else:
            print(f"{job:20s} {a['E']:>4}{a['k']:>3}{a['seq']:>6}{a['b']:>3} "
                  f"{'':>9} {'':>6} {'':>9} {src:>6}  {note}")
    print(f"\n(median over step>={WARMUP:.0f}; {NGPU}-GPU; tok/24h@0.85; 'hook'=authoritative, '~tqdm'=rough)")


if __name__ == "__main__":
    main()
