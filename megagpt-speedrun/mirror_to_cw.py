"""Mirror R2 levanter `tokenized/` TreeCaches -> CoreWeave cwobject (cluster-local).

Runs as a worker iris job: R2 read creds are auto-injected (AWS_*); CW write creds come from
CW_KEY_ID/CW_KEY_SECRET (passed via -e). Smart-ish protocol:
  * size every component's train/ prefix, copy SMALLEST-FIRST so experiments can bootstrap on
    the first completed cache;
  * massively parallel object copy (latency-bound: caches are millions of ~1MB chunk files);
  * resumable: skip objects already present on cwobject with identical size;
  * write a MANIFEST after each component completes so the data path knows what's ready.

Dest layout mirrors the source key: R2 marin-na/marin/tokenized/<rel>/... ->
cwobject marin-us-east-02a/marin/tokenized/<rel>/...  (read back with bucket:"" +
endpoint=https://marin-us-east-02a.cwobject.com).
"""
import os, sys, time, json, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import botocore.session
from botocore.config import Config

SRC_BUCKET = "marin-na"
DST_BUCKET = "marin-us-east-02a"
KEY_PREFIX = "marin/tokenized"            # under both buckets
DST_ENDPOINT = "https://cwobject.com"

# rel path (under marin/tokenized) -> the 7 production components. /train holds the cache.
COMPONENTS = {
    "nemotron_cc/hq_actual":   "nemotron_cc/hq_actual-5af4cc",
    "nemotron_cc/medium_high": "nemotron_cc/medium_high-d21701",
    "nemotron_cc/medium_low":  "nemotron_cc/medium_low-5b94a4",
    "nemotron_cc/low_actual":  "nemotron_cc/low_actual-cb3f2c",
    "nemotron_cc/low_synth":   "nemotron_cc/low_synth-3c57b3",
    "starcoderdata":           "starcoderdata-12f018",
    "proofpile_2":             "proofpile_2-5ba7ac",
}
# optional subset / cap via env
ONLY = [c.strip() for c in os.environ.get("MIRROR_ONLY", "").split(",") if c.strip()]
MAX_GB = float(os.environ.get("MIRROR_MAX_GB", "0"))      # 0 = no cap
WORKERS = int(os.environ.get("MIRROR_WORKERS", "96"))
SPLIT = os.environ.get("MIRROR_SPLIT", "train")

_sess = botocore.session.get_session()
cfg = Config(max_pool_connections=WORKERS + 16, retries={"max_attempts": 6, "mode": "adaptive"},
             read_timeout=120, connect_timeout=20)
src = _sess.create_client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL"), config=cfg)
dst = _sess.create_client("s3", endpoint_url=DST_ENDPOINT,
                   aws_access_key_id=os.environ["CW_KEY_ID"],
                   aws_secret_access_key=os.environ["CW_KEY_SECRET"],
                   region_name="us-east-1",
                   config=Config(max_pool_connections=WORKERS + 16, s3={"addressing_style": "virtual"},
                                 signature_version="s3v4", retries={"max_attempts": 6, "mode": "adaptive"},
                                 read_timeout=120, connect_timeout=20))

def list_objs(prefix):
    """yield (key, size) under SRC prefix."""
    tok = None
    while True:
        kw = dict(Bucket=SRC_BUCKET, Prefix=prefix, MaxKeys=1000)
        if tok: kw["ContinuationToken"] = tok
        r = src.list_objects_v2(**kw)
        for o in r.get("Contents", []):
            yield o["Key"], o["Size"]
        if not r.get("IsTruncated"): break
        tok = r["NextContinuationToken"]

def dst_size(key):
    try:
        return dst.head_object(Bucket=DST_BUCKET, Key=key)["ContentLength"]
    except Exception:
        return None

def copy_one(key, size):
    if dst_size(key) == size:                       # resumable: already mirrored
        return ("skip", size)
    body = src.get_object(Bucket=SRC_BUCKET, Key=key)["Body"].read()
    dst.put_object(Bucket=DST_BUCKET, Key=key, Body=body)
    return ("copy", len(body))

def mirror_component(name, rel):
    prefix = f"{KEY_PREFIX}/{rel}/{SPLIT}"
    print(f"\n[{name}] listing {prefix} ...", flush=True)
    objs = list(list_objs(prefix))
    total = sum(s for _, s in objs)
    print(f"[{name}] {len(objs)} objects, {total/1e9:.1f} GB", flush=True)
    done_b = [0]; copied_b = [0]; n_done = [0]; n_copy = [0]
    lock = threading.Lock(); t0 = time.time()
    def work(arg):
        kind, b = copy_one(*arg)
        with lock:
            done_b[0] += arg[1]; n_done[0] += 1
            if kind == "copy": copied_b[0] += b; n_copy[0] += 1
            if n_done[0] % 2000 == 0 or n_done[0] == len(objs):
                dt = time.time() - t0
                print(f"[{name}] {n_done[0]}/{len(objs)} objs  {done_b[0]/1e9:.1f}/{total/1e9:.1f} GB "
                      f"(copied {copied_b[0]/1e9:.1f} GB) {copied_b[0]/1e6/max(dt,1):.0f} MB/s", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for f in as_completed(ex.submit(work, a) for a in objs):
            f.result()
    print(f"[{name}] *** COMPONENT COMPLETE *** {total/1e9:.1f} GB, copied {n_copy[0]} new objs "
          f"in {time.time()-t0:.0f}s", flush=True)
    return total

def main():
    comps = {k: v for k, v in COMPONENTS.items() if not ONLY or k in ONLY}
    print(f"=== sizing {len(comps)} components (smallest-first) ===", flush=True)
    sizes = {}
    for name, rel in comps.items():
        prefix = f"{KEY_PREFIX}/{rel}/{SPLIT}"
        s = 0
        for _, sz in list_objs(prefix):
            s += sz
        sizes[name] = s
        print(f"  {name:28s} {s/1e9:8.1f} GB", flush=True)
    order = sorted(sizes, key=sizes.get)
    print(f"order: {order}", flush=True)
    manifest, cum = [], 0.0
    for name in order:
        if MAX_GB and cum/1e9 >= MAX_GB:
            print(f"=== MAX_GB cap {MAX_GB} reached, stopping before {name} ===", flush=True); break
        cum += mirror_component(name, comps[name])
        manifest.append({"name": name, "rel": comps[name], "bytes": sizes[name]})
        dst.put_object(Bucket=DST_BUCKET, Key=f"{KEY_PREFIX}/_mirror_manifest.json",
                       Body=json.dumps({"complete": manifest}, indent=2).encode())
        print(f"=== manifest updated: {[m['name'] for m in manifest]} ===", flush=True)
    print("\n=== MIRROR DONE ===", flush=True)

if __name__ == "__main__":
    main()
