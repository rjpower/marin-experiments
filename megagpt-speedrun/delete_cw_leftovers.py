"""Delete the megagpt-speedrun leftovers from the CoreWeave cluster-local **cwobject** store.

Background (marin issue #6688): to unblock the megagpt-speedrun 8xH100 runs on
``cw-us-east-02a`` before native CW data existed, ``mirror_to_cw.py`` byte-for-byte copied the
R2 ``marin/tokenized/`` nemotron TreeCaches onto the cluster-local cwobject store
(``s3://marin-us-east-02a/marin/tokenized/``). Some cwobject-data test runs
(``launch_cwdata.sh`` sets ``MARIN_PREFIX=s3://marin-us-east-02a/marin``) also left temp
checkpoints and smoke artifacts behind. This is pure scaffolding; the production run used R2.
This script deletes that leftover cwobject data.

cwobject specifics (same as mirror_to_cw.py / cw_patch.py):
  * endpoint https://cwobject.com, bucket ``marin-us-east-02a``
  * VIRTUAL-HOSTED addressing only (path-style is rejected) -> boto s3.addressing_style=virtual
  * write/delete creds come from env CW_KEY_ID / CW_KEY_SECRET (NOT the AWS_* R2 creds)
  * cwobject is cluster-local but its endpoint is reachable off-cluster too, so this can run
    from a laptop with the CW creds exported. (It is I/O-latency bound, not bandwidth bound:
    deletes are batched 1000 keys/call and parallelized.)

SAFETY: dry-run by default (lists count + size, deletes NOTHING). Pass --execute to delete.
The bucket is hard-coded; prefixes are validated against a minimum specificity so a typo can
never wipe the bucket. Deletion is idempotent -- re-run to mop up anything left.

Usage:
  # 1) inspect what WOULD be deleted (the nemotron cache copy -- the default target):
  CW_KEY_ID=... CW_KEY_SECRET=... uv run python delete_cw_leftovers.py

  # 2) actually delete the nemotron cache copy:
  CW_KEY_ID=... CW_KEY_SECRET=... uv run python delete_cw_leftovers.py --execute

  # other scopes (see GROUPS below); combine with commas:
  ... delete_cw_leftovers.py --groups nemotron-cache,run-artifacts --execute
  ... delete_cw_leftovers.py --prefix marin/some/other/prefix/ --execute   # ad-hoc target
"""
import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import botocore.session
from botocore.config import Config

ENDPOINT = "https://cwobject.com"
BUCKET = "marin-us-east-02a"
REGION = "us-east-1"

# Named deletion scopes. Each maps to a list of key prefixes under BUCKET.
#   nemotron-cache : the mirror_to_cw.py copy of the R2 tokenized/ nemotron caches -- THE
#                    "temporary nemotron cache copy" this cleanup is about (~2.25 TB).
#   run-artifacts  : temp checkpoints + benchmark/smoke stores + tokenizer copy that the
#                    cwobject-data test runs (launch_cwdata.sh, MARIN_PREFIX=cwobject) left
#                    behind. Pure scaffolding; the production run wrote to R2 (marin-na).
#   datakit        : datakit tokenization store copies. **GUARDED** -- verify these are not the
#                    #6036 native-CW-data effort's store before deleting (see --allow-datakit).
GROUPS = {
    "nemotron-cache": [
        "marin/tokenized/",
    ],
    "run-artifacts": [
        "marin/tmp/",
        "marin/grug/",
        "marin/tokenizers/",
        "marin/_cwtest/",
        "marin/experiments/",
    ],
    "datakit": [
        "marin/datakit/store_8ac06c74/",
        "datakit/store_8ac06c74/",
    ],
}
DEFAULT_GROUPS = ["nemotron-cache"]

# A prefix must have at least this many "/"-separated non-empty segments to be accepted, so a
# bare "" / "/" / "marin/" (which would match huge swaths of the shared cluster store) is
# rejected. Every real target above clears this.
MIN_PREFIX_SEGMENTS = 2


def make_client():
    key = os.environ.get("CW_KEY_ID")
    secret = os.environ.get("CW_KEY_SECRET")
    if not (key and secret):
        sys.exit("ERROR: set CW_KEY_ID and CW_KEY_SECRET (the cwobject creds) in the environment.")
    sess = botocore.session.get_session()
    return sess.create_client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        region_name=REGION,
        config=Config(
            max_pool_connections=128,
            s3={"addressing_style": "virtual"},
            signature_version="s3v4",
            retries={"max_attempts": 6, "mode": "adaptive"},
            read_timeout=120,
            connect_timeout=20,
        ),
    )


def validate_prefix(prefix: str):
    segs = [s for s in prefix.split("/") if s]
    if len(segs) < MIN_PREFIX_SEGMENTS:
        sys.exit(
            f"ERROR: refusing prefix {prefix!r} -- too broad (< {MIN_PREFIX_SEGMENTS} path "
            f"segments). This guard prevents wiping the shared cluster bucket by mistake."
        )


def iter_keys(client, prefix):
    """Yield (key, size) for every object under prefix."""
    token = None
    while True:
        kw = dict(Bucket=BUCKET, Prefix=prefix, MaxKeys=1000)
        if token:
            kw["ContinuationToken"] = token
        r = client.list_objects_v2(**kw)
        for o in r.get("Contents", []):
            yield o["Key"], o["Size"]
        if not r.get("IsTruncated"):
            break
        token = r["NextContinuationToken"]


def survey(client, prefix):
    n = 0
    b = 0
    for _, size in iter_keys(client, prefix):
        n += 1
        b += size
    return n, b


def delete_prefix(client, prefix, workers=32):
    """Batch-delete every object under prefix. Returns (deleted, errors)."""
    deleted = [0]
    errors = [0]
    lock = threading.Lock()
    t0 = time.time()

    def delete_batch(batch):
        resp = client.delete_objects(
            Bucket=BUCKET,
            Delete={"Objects": [{"Key": k} for k in batch], "Quiet": True},
        )
        errs = resp.get("Errors", [])
        with lock:
            deleted[0] += len(batch) - len(errs)
            errors[0] += len(errs)
            if errs:
                for e in errs[:5]:
                    print(f"    delete error {e.get('Key')}: {e.get('Code')} {e.get('Message')}")
            if deleted[0] % 20000 < 1000 or errs:
                print(
                    f"  [{prefix}] deleted {deleted[0]:,} ({errors[0]} errors) "
                    f"{deleted[0]/max(time.time()-t0,1):.0f} obj/s",
                    flush=True,
                )

    batch = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = []
        for key, _ in iter_keys(client, prefix):
            batch.append(key)
            if len(batch) == 1000:
                futures.append(ex.submit(delete_batch, batch))
                batch = []
        if batch:
            futures.append(ex.submit(delete_batch, batch))
        for f in as_completed(futures):
            f.result()
    return deleted[0], errors[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--groups",
        default=",".join(DEFAULT_GROUPS),
        help=f"comma-separated scopes to delete. Available: {', '.join(GROUPS)}. "
        f"Default: {','.join(DEFAULT_GROUPS)}",
    )
    ap.add_argument("--prefix", action="append", default=[], help="ad-hoc extra key prefix to delete (repeatable)")
    ap.add_argument("--execute", action="store_true", help="actually delete (default: dry-run survey only)")
    ap.add_argument("--yes", action="store_true", help="skip the interactive typed confirmation (for scripts)")
    ap.add_argument(
        "--allow-datakit",
        action="store_true",
        help="required to include the 'datakit' group -- confirm the store is NOT the #6036 native-CW-data effort first",
    )
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    group_names = [g.strip() for g in args.groups.split(",") if g.strip()]
    prefixes = []
    for g in group_names:
        if g not in GROUPS:
            sys.exit(f"ERROR: unknown group {g!r}. Available: {', '.join(GROUPS)}")
        if g == "datakit" and not args.allow_datakit:
            sys.exit(
                "ERROR: the 'datakit' group is guarded. The store_8ac06c74 copies may belong to the "
                "#6036 native-CW-data effort. Verify ownership, then pass --allow-datakit to proceed."
            )
        prefixes.extend(GROUPS[g])
    prefixes.extend(args.prefix)
    # de-dup, preserve order
    prefixes = list(dict.fromkeys(prefixes))
    if not prefixes:
        sys.exit("ERROR: nothing to do (no groups/prefixes selected).")
    for p in prefixes:
        validate_prefix(p)

    client = make_client()

    print(f"Target store: {ENDPOINT} bucket={BUCKET} (virtual-hosted)")
    print(f"Mode: {'EXECUTE (will DELETE)' if args.execute else 'DRY-RUN (survey only, deletes nothing)'}\n")
    print("Surveying prefixes ...")
    total_n = 0
    total_b = 0
    for p in prefixes:
        n, b = survey(client, p)
        total_n += n
        total_b += b
        print(f"  {p:45s} {n:>10,d} objs  {b/1e9:>10.2f} GB")
    print(f"  {'TOTAL':45s} {total_n:>10,d} objs  {total_b/1e9:>10.2f} GB\n")

    if not args.execute:
        print("Dry-run only. Re-run with --execute to delete the above.")
        return
    if total_n == 0:
        print("Nothing to delete.")
        return

    if not args.yes:
        print(f"About to PERMANENTLY DELETE {total_n:,} objects ({total_b/1e9:.2f} GB) from {BUCKET}.")
        resp = input("Type 'DELETE' to proceed: ").strip()
        if resp != "DELETE":
            sys.exit("Aborted.")

    grand_del = 0
    grand_err = 0
    for p in prefixes:
        print(f"\nDeleting {p} ...", flush=True)
        d, e = delete_prefix(client, p, workers=args.workers)
        grand_del += d
        grand_err += e
        print(f"  done {p}: deleted {d:,}, errors {e}")
    print(f"\n=== COMPLETE: deleted {grand_del:,} objects, {grand_err} errors ===")
    if grand_err:
        sys.exit(1)


if __name__ == "__main__":
    main()
