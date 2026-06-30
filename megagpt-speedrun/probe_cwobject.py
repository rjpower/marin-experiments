"""Worker-side probe: is CoreWeave cluster-local object storage (cwobject) faster than
remote R2 for reading tokenized data, and is store_8ac06c74 loadable?

Run as an iris job ON a worker (cluster-local to cwobject). Compares sustained read
throughput cwobject-vs-R2 and dumps the store's tensorstore/zarr layout.
"""
import os, time, sys

def s3fs_for(key, secret, endpoint):
    import s3fs
    return s3fs.S3FileSystem(
        key=key, secret=secret,
        client_kwargs={"endpoint_url": endpoint},
        config_kwargs={"s3": {"addressing_style": "virtual"}},
    )

def bench(fs, files, label):
    total = 0; t0 = time.time(); n = 0; lat = []
    for f in files:
        try:
            ts = time.time()
            b = fs.cat_file(f)
            lat.append(time.time() - ts)
            total += len(b); n += 1
        except Exception as e:
            print(f"  [{label}] read err {f}: {str(e)[:120]}")
    dt = time.time() - t0
    if n:
        import statistics
        print(f"[{label}] {n} objs, {total/1e6:.1f}MB in {dt:.2f}s = {total/1e6/dt:.1f} MB/s "
              f"| per-obj latency med={statistics.median(lat)*1000:.0f}ms max={max(lat)*1000:.0f}ms")
    else:
        print(f"[{label}] no objects read")

def collect(fs, root, want=40):
    # find data chunk files under a prefix
    out = []
    try:
        for p in fs.find(root):
            if p.rstrip("/").endswith(("zarr.json", "shard_ledger.json")):
                continue
            out.append(p)
            if len(out) >= want:
                break
    except Exception as e:
        print(f"  find err {root}: {str(e)[:160]}")
    return out

print("=== CWOBJECT vs R2 read probe (on worker) ===", flush=True)

# --- cwobject (cluster-local) ---
ck, cs = os.environ.get("CW_KEY_ID"), os.environ.get("CW_KEY_SECRET")
print(f"CW creds present: {bool(ck and cs)}", flush=True)
if ck and cs:
    try:
        cw = s3fs_for(ck, cs, "https://cwobject.com")
        # cluster=5 had ~125M tokens -> real data chunks
        root = "marin-us-east-02a/datakit/store_8ac06c74/cluster=5/quality=0/"
        parts = cw.ls(root)[:6]
        files = []
        for pt in parts:
            files += collect(cw, pt, want=12)
        print(f"cwobject: {len(files)} data objects under {len(parts)} parts", flush=True)
        bench(cw, files[:60], "CWOBJECT")
        # dump one part's layout + ledger
        if parts:
            lay = cw.find(parts[0])
            print("sample part layout:", [p.split(parts[0])[-1] for p in lay][:8])
    except Exception as e:
        print("cwobject probe ERR:", str(e)[:200], flush=True)

# --- R2 (remote, what we read today) ---
ep = os.environ.get("AWS_ENDPOINT_URL", "")
print(f"\nR2 endpoint env: {ep[:60]}", flush=True)
try:
    import s3fs
    r2 = s3fs.S3FileSystem(client_kwargs={"endpoint_url": ep} if ep else {})
    # a real tokenized cache we train from
    r2root = "marin-na/marin/tokenized/proofpile_2-5ba7ac/train/"
    rfiles = collect(r2, r2root, want=60)
    print(f"R2: {len(rfiles)} data objects under {r2root}", flush=True)
    bench(r2, rfiles[:60], "R2")
except Exception as e:
    print("R2 probe ERR:", str(e)[:200], flush=True)

print("=== probe done ===", flush=True)
