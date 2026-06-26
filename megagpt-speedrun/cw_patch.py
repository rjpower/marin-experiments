"""Runtime patch: virtual-hosted S3 addressing for CoreWeave cwobject.

levanter's ``build_kvstore_spec`` emits a path-style tensorstore s3 spec
(``{endpoint}/{bucket}/{key}``). CoreWeave cwobject REJECTS path-style
("PathStyleRequestNotAllowed"); it requires virtual-hosted addressing
(``{bucket}.{host}/{key}``). tensorstore's s3 driver only does virtual-hosted with an
EMPTY bucket field + the bucket folded into the endpoint host -- supported from
tensorstore 0.1.84 (google/tensorstore#285).

When ``LEVANTER_S3_VIRTUAL_HOSTED`` is truthy AND ``AWS_ENDPOINT_URL`` is set, we rewrite
the spec to: ``bucket=""``, ``endpoint=https://<bucket>.<endpoint-host>``. Otherwise we
defer to the original (R2 path-style is unchanged). Import this module BEFORE any cache is
opened (i.e. at the top of launch.py). The upstream version of this lives in a marin PR;
this keeps us unblocked until that lands + is repinned.
"""
import os
import urllib.parse


def _virtual_hosted_enabled() -> bool:
    return os.environ.get("LEVANTER_S3_VIRTUAL_HOSTED", "").strip().lower() in ("1", "true", "yes")


def _build_kvstore_spec(path: str) -> dict:
    parsed = urllib.parse.urlparse(path)
    if parsed.scheme == "s3":
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        endpoint = os.environ.get("AWS_ENDPOINT_URL")
        if endpoint and _virtual_hosted_enabled():
            ep = urllib.parse.urlparse(endpoint)
            host = ep.netloc or ep.path  # tolerate "cwobject.com" with no scheme
            scheme = ep.scheme or "https"
            spec: dict = {"driver": "s3", "bucket": "", "path": key,
                          "endpoint": f"{scheme}://{bucket}.{host}"}
        else:
            spec = {"driver": "s3", "bucket": bucket, "path": key}
            if endpoint:
                spec["endpoint"] = endpoint
        region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")
        if region:
            spec["aws_region"] = region
        elif endpoint:
            spec["aws_region"] = "us-east-1"
        if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
            spec["aws_credentials"] = {"type": "environment"}
        return spec
    elif parsed.scheme == "gs":
        return {"driver": "gcs", "bucket": parsed.netloc, "path": parsed.path.lstrip("/")}
    elif parsed.scheme in ("", "file"):
        return {"driver": "file", "path": os.path.abspath(path)}
    else:
        raise ValueError(f"Unsupported URI scheme for tensorstore: {parsed.scheme!r} in {path!r}")


def _override_s3_env_in_process() -> None:
    """iris injects AWS_ENDPOINT_URL (the cluster's R2 endpoint) at a layer that OVERRIDES the
    job's ``-e`` flags, so we cannot point the process at cwobject from the launcher. Override it
    IN-PROCESS here (cw_patch is imported first in launch.py, before any s3 client / the marin
    executor runs). The intended endpoint comes from ``CW_S3_ENDPOINT`` (a non-AWS_-named var
    iris leaves alone); creds come from ``CW_KEY_ID``/``CW_KEY_SECRET``. After this, the WHOLE
    process (marin executor + checkpoints + data, via env-based s3fs/tensorstore) talks cwobject.
    iris's own R2 state lives in a separate agent process and is unaffected."""
    ep = os.environ.get("CW_S3_ENDPOINT", "https://cwobject.com")
    os.environ["AWS_ENDPOINT_URL"] = ep
    if os.environ.get("CW_KEY_ID") and os.environ.get("CW_KEY_SECRET"):
        os.environ["AWS_ACCESS_KEY_ID"] = os.environ["CW_KEY_ID"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["CW_KEY_SECRET"]
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")


def _patch_s3fs_makedirs() -> None:
    """s3fs.makedirs (called by BOTH levanter's fsspec_utils AND fsspec-core's open_files) tries
    to ``create_bucket``. cwobject's bucket already exists and CreateBucket may be rejected; s3
    has no real directories anyway (writes create keys directly). Make makedirs a tolerant no-op
    on errors so cache opens / executor writes don't crash."""
    try:
        import s3fs
    except Exception:
        return
    _orig = s3fs.S3FileSystem.makedirs

    def _safe(self, path, exist_ok=True):
        try:
            return _orig(self, path, exist_ok=exist_ok)
        except Exception:
            return None  # bucket exists / CreateBucket unsupported -> keys are written directly

    s3fs.S3FileSystem.makedirs = _safe


def _configure_fsspec_virtual_hosted() -> None:
    """levanter also uses fsspec/s3fs (NOT tensorstore) for cache metadata + mkdirs. s3fs/botocore
    default to PATH-STYLE against a custom endpoint, which cwobject rejects. botocore reads the
    addressing style from the shared AWS config file, so write one forcing virtual-hosted and
    point AWS_CONFIG_FILE at it. (boto/s3fs/aiobotocore all honor it; tensorstore does not, but
    that side is handled by the empty-bucket spec above.)"""
    cfg_path = os.path.join(os.environ.get("TMPDIR", "/tmp"), "cw_aws_config")
    region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION") or "us-east-1"
    try:
        with open(cfg_path, "w") as f:
            f.write(f"[default]\nregion = {region}\ns3 =\n    addressing_style = virtual\n")
        os.environ["AWS_CONFIG_FILE"] = cfg_path
    except Exception as e:  # best-effort
        print(f"[cw_patch] could not write AWS config: {e}", flush=True)


def _patch_mkdirs() -> None:
    """s3 has no real directories; on a virtual-host-only store (cwobject) the bucket already
    exists and ``makedirs`` may try (and fail) to ``create_bucket``. Reads/writes create keys
    directly, so swallow bucket/region create errors instead of crashing the cache open."""
    try:
        import levanter.utils.fsspec_utils as _fsu
    except Exception:
        return
    _orig = _fsu.mkdirs

    def _safe_mkdirs(path):
        try:
            _orig(path)
        except Exception as e:
            msg = str(e)
            if str(path).startswith(("s3://", "s3a://")) or any(
                t in msg for t in ("Bucket", "bucket", "Region", "PathStyle", "CreateBucket")
            ):
                return
            raise

    _fsu.mkdirs = _safe_mkdirs


def apply() -> None:
    """Monkeypatch every import site of build_kvstore_spec + configure fsspec. Idempotent."""
    import levanter.tensorstore_serialization as _tss
    _tss.build_kvstore_spec = _build_kvstore_spec
    # jagged_array did `from levanter.tensorstore_serialization import build_kvstore_spec`,
    # so it holds its own name binding that must be patched too.
    try:
        import levanter.store.jagged_array as _ja
        _ja.build_kvstore_spec = _build_kvstore_spec
    except Exception:
        pass
    if _virtual_hosted_enabled():
        _override_s3_env_in_process()
        _configure_fsspec_virtual_hosted()
        _patch_mkdirs()
        _patch_s3fs_makedirs()
        klen = len(os.environ.get("AWS_ACCESS_KEY_ID", ""))
        print(f"[cw_patch] virtual-hosted S3 ENABLED (endpoint={os.environ.get('AWS_ENDPOINT_URL')}, "
              f"key_len={klen}, MARIN_PREFIX={os.environ.get('MARIN_PREFIX')}, "
              f"AWS_CONFIG_FILE={os.environ.get('AWS_CONFIG_FILE')})", flush=True)


# apply on import for convenience
apply()
