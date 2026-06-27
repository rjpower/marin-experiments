#!/usr/bin/env python
"""Download a grug xprof trace from R2 and produce a device-side op-time breakdown that explains MFU.

The profile run (SP_PROFILE=1) writes a TensorBoard/xprof trace; launch.py uploads it (and levanter
may also write it directly) to `<output_path>/profiler/**` on R2. xprof emits a Perfetto/Chrome
`*.trace.json.gz` (self-describing JSON) alongside `*.xplane.pb`. We parse the trace.json.gz: keep the
GPU/device "X" (complete-duration) events, aggregate total GPU-time by op name, and roll names up into
components (attention / MoE ragged_dot / lm_head+CE / EP all-to-all / router top-k sort / optimizer /
copy). That rollup is what tells us where the ~8% MFU goes.

Usage:
  uv run python analyze_profile.py <run_id_substr>   # e.g. st40-w8   (finds the matching R2 dir)
"""
import gzip
import json
import os
import re
import sys
from collections import defaultdict

import s3fs

R2 = "https://74981a43be0de7712369306c7b19133d.r2.cloudflarestorage.com"
BASE = "marin-na/marin/grug/sparsity"

# name-substring -> component. First match wins (order matters). Lowercased compare.
CATEGORIES = [
    ("nccl", "EP/collective (all-to-all/all-reduce)"),
    ("all-to-all", "EP/collective (all-to-all/all-reduce)"),
    ("alltoall", "EP/collective (all-to-all/all-reduce)"),
    ("allreduce", "EP/collective (all-to-all/all-reduce)"),
    ("reducescatter", "EP/collective (all-to-all/all-reduce)"),
    ("allgather", "EP/collective (all-to-all/all-reduce)"),
    ("flash", "attention (FA4)"),
    ("cute", "attention (FA4)"),
    ("fa4", "attention (FA4)"),
    ("attention", "attention (FA4)"),
    ("ragged", "MoE grouped matmul (ragged_dot)"),
    ("triton", "MoE grouped matmul (ragged_dot)"),
    ("group", "MoE grouped matmul (ragged_dot)"),
    ("radixsort", "router top-k sort"),
    ("sort", "router top-k sort"),
    ("topk", "router top-k sort"),
    ("top_k", "router top-k sort"),
    ("approx", "router top-k sort"),
    ("cross_entropy", "lm_head + cross-entropy"),
    ("softmax", "lm_head + cross-entropy"),
    ("logits", "lm_head + cross-entropy"),
    ("lm_head", "lm_head + cross-entropy"),
    ("gemm", "dense matmul (gemm: lm_head/proj/attn-qkvo)"),
    ("cutlass", "dense matmul (gemm: lm_head/proj/attn-qkvo)"),
    ("sm90", "dense matmul (gemm: lm_head/proj/attn-qkvo)"),
    ("dot", "dense matmul (gemm: lm_head/proj/attn-qkvo)"),
    ("fusion", "elementwise/optimizer fusions"),
    ("loop", "elementwise/optimizer fusions"),
    ("copy", "copy/transpose/reshape"),
    ("transpose", "copy/transpose/reshape"),
    ("memcpy", "copy/transpose/reshape"),
    ("convert", "copy/transpose/reshape"),
]


def categorize(name):
    n = name.lower()
    for sub, cat in CATEGORIES:
        if sub in n:
            return cat
    return "other"


def fs_r2():
    return s3fs.S3FileSystem(client_kwargs={"endpoint_url": R2},
                             key=os.environ.get("R2_ACCESS_KEY_ID"),
                             secret=os.environ.get("R2_SECRET_ACCESS_KEY"))


def find_run_dir(fs, substr):
    cands = [d for d in fs.ls(BASE) if substr in d.split("/")[-1]]
    if not cands:
        sys.exit(f"no run dir under {BASE} matching {substr!r}")
    # prefer one that has a profiler/ dir
    withprof = [d for d in cands if fs.exists(f"{d}/profiler")]
    chosen = (withprof or cands)
    if len(chosen) > 1:
        print(f"[warn] multiple matches, using newest-sorted: {[c.split('/')[-1] for c in chosen]}")
    return sorted(chosen)[-1]


def load_trace_events(fs, run_dir):
    prof = f"{run_dir}/profiler"
    if not fs.exists(prof):
        sys.exit(f"no profiler/ under {run_dir} (trace not uploaded yet?)")
    allf = [f for f in fs.find(prof)]
    traces = [f for f in allf if f.endswith(".trace.json.gz") or f.endswith(".trace.json")]
    print(f"profiler files ({len(allf)}): " + ", ".join(sorted({f.split('/')[-1] for f in allf})))
    if not traces:
        sys.exit("no *.trace.json[.gz] found; only xplane.pb present — parse with tensorboard_plugin_profile")
    tf = sorted(traces)[-1]
    print(f"parsing {tf.split('/')[-1]}")
    raw = fs.cat_file(tf)
    if tf.endswith(".gz"):
        raw = gzip.decompress(raw)
    return json.loads(raw)


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: analyze_profile.py <run_id_substr>")
    substr = sys.argv[1]
    fs = fs_r2()
    run_dir = find_run_dir(fs, substr)
    print(f"run dir: {run_dir.split('/')[-1]}\n")
    data = load_trace_events(fs, run_dir)
    events = data.get("traceEvents", data) if isinstance(data, dict) else data

    # identify device/GPU process+thread ids from metadata; device tracks have "/device:GPU" or "stream"
    device_pids = set()
    for e in events:
        if e.get("ph") == "M" and e.get("name") in ("process_name", "thread_name"):
            nm = str(e.get("args", {}).get("name", "")).lower()
            if "gpu" in nm or "stream" in nm or "device" in nm:
                device_pids.add(e.get("pid"))

    by_name = defaultdict(float)
    by_cat = defaultdict(float)
    n_dur = 0
    total = 0.0
    for e in events:
        if e.get("ph") != "X":
            continue
        if device_pids and e.get("pid") not in device_pids:
            continue
        dur = float(e.get("dur", 0))
        if dur <= 0:
            continue
        name = e.get("name", "?")
        # strip the trailing "#hlo_op=..." / arg noise for grouping
        key = re.split(r"[#(]", name)[0].strip()[:60]
        by_name[key] += dur
        by_cat[categorize(name)] += dur
        total += dur
        n_dur += 1

    if total == 0:
        sys.exit("no device-duration events parsed (pid filter may be wrong); inspect trace manually")

    print(f"device duration events: {n_dur}; total device-busy us: {total:,.0f}\n")
    print("=== COMPONENT ROLLUP (share of device-busy time) ===")
    for cat, us in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f"  {us/total*100:6.2f}%  {us:>14,.0f} us  {cat}")
    print("\n=== TOP 30 OPS ===")
    for name, us in sorted(by_name.items(), key=lambda x: -x[1])[:30]:
        print(f"  {us/total*100:6.2f}%  {us:>14,.0f} us  [{categorize(name)[:28]:28s}] {name}")


if __name__ == "__main__":
    main()
