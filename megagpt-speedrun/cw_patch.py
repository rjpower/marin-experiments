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
    # botocore endpoint precedence: AWS_ENDPOINT_URL_S3 (service-specific) > AWS_ENDPOINT_URL
    # (global) > config-file endpoint. iris sets the service-specific one to R2, so we MUST set
    # it too (setting only the global one is silently ignored).
    os.environ["AWS_ENDPOINT_URL"] = ep
    os.environ["AWS_ENDPOINT_URL_S3"] = ep
    if os.environ.get("CW_KEY_ID") and os.environ.get("CW_KEY_SECRET"):
        os.environ["AWS_ACCESS_KEY_ID"] = os.environ["CW_KEY_ID"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["CW_KEY_SECRET"]
    # drop any iris/profile creds file pointer so a stale [default] R2 profile can't shadow our
    # env creds/endpoint (env creds already win, but AWS_PROFILE can change resolution order).
    os.environ.pop("AWS_PROFILE", None)
    # iris sets AWS_DEFAULT_REGION/AWS_REGION="auto" (R2's value); cwobject SigV4 needs us-east-1,
    # so OVERWRITE (not setdefault -- the keys already exist, so setdefault is a silent no-op).
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    os.environ["AWS_REGION"] = "us-east-1"


def _override_fsspec_conf() -> None:
    """iris seeds ``fsspec.config.conf['s3']`` (from ``FSSPEC_S3_*`` env) with the R2 endpoint;
    fsspec injects that into EVERY ``S3FileSystem`` as explicit kwargs (beating our env override),
    so writes leak to R2 (``access key length 16, should be 32``). Replace that protocol-config
    block with cwobject so every fsspec S3 filesystem -- however it's constructed -- targets
    cwobject with virtual-hosted addressing."""
    try:
        import fsspec
    except Exception:
        return
    ep = os.environ.get("AWS_ENDPOINT_URL")
    key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    region = os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    s3conf = dict(fsspec.config.conf.get("s3") or {})
    if ep:
        s3conf["endpoint_url"] = ep
    if key:
        s3conf["key"] = key
    if secret:
        s3conf["secret"] = secret
    ck = dict(s3conf.get("client_kwargs") or {})
    ck.pop("endpoint_url", None)  # endpoint_url is top-level only (avoid create_client dup)
    ck["region_name"] = region
    s3conf["client_kwargs"] = ck
    cfgk = dict(s3conf.get("config_kwargs") or {})
    cfgk.setdefault("signature_version", "s3v4")
    s3cfg = dict(cfgk.get("s3") or {})
    s3cfg.setdefault("addressing_style", "virtual")
    cfgk["s3"] = s3cfg
    s3conf["config_kwargs"] = cfgk
    fsspec.config.conf["s3"] = s3conf


def _patch_s3fs_client() -> None:
    """Force EVERY ``s3fs.S3FileSystem`` to use the exact recipe the mirror proved works against
    cwobject: explicit ``endpoint_url`` (generic ``https://cwobject.com``), CW key/secret,
    ``region_name=us-east-1``, virtual-hosted addressing, SigV4. Relying on env-based endpoint
    resolution leaked writes to R2 on the worker (``PutObject`` -> ``InvalidArgument: Credential
    access key has length 16, should be 32`` -- that's R2's 32-hex-char rule; cwobject happily
    takes our 16-char key). Injecting ``client_kwargs``/``config_kwargs`` removes all
    endpoint-resolution ambiguity. ``setdefault`` everywhere so an explicit caller (e.g. a
    different bucket) still wins."""
    try:
        import s3fs
    except Exception:
        return
    ep = os.environ.get("AWS_ENDPOINT_URL")          # generic endpoint; botocore prepends bucket
    key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    region = os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    _orig_init = s3fs.S3FileSystem.__init__

    def _init(self, *args, **kwargs):
        # FORCE (not setdefault) endpoint_url/key/secret: fsspec merges fsspec.config.conf["s3"]
        # (iris pre-seeds it with the R2 endpoint from FSSPEC_S3_* env) into kwargs as EXPLICIT
        # values BEFORE __init__, so setdefault is beaten and writes leak to R2. We own this whole
        # process (everything talks cwobject), so overriding is safe.
        # endpoint_url MUST be top-level only (s3fs builds create_client(**init_kwargs, **client_kwargs)
        # with endpoint_url in init_kwargs); a stray endpoint_url in client_kwargs -> "multiple values".
        if key:
            kwargs["key"] = key
        if secret:
            kwargs["secret"] = secret
        if ep:
            kwargs["endpoint_url"] = ep
        ck = dict(kwargs.get("client_kwargs") or {})
        ck.pop("endpoint_url", None)
        ck["region_name"] = region
        kwargs["client_kwargs"] = ck
        cfgk = dict(kwargs.get("config_kwargs") or {})
        cfgk.setdefault("signature_version", "s3v4")
        s3cfg = dict(cfgk.get("s3") or {})
        s3cfg.setdefault("addressing_style", "virtual")
        cfgk["s3"] = s3cfg
        kwargs["config_kwargs"] = cfgk
        return _orig_init(self, *args, **kwargs)

    s3fs.S3FileSystem.__init__ = _init
    # nuke any instance cached BEFORE the patch (e.g. an R2-pointed default fs created at import).
    try:
        s3fs.S3FileSystem.clear_instance_cache()
    except Exception:
        pass


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
    """Monkeypatch every import site of build_kvstore_spec + configure fsspec. Idempotent.

    Order matters: set the S3 env + patch s3fs FIRST (before importing any levanter module), so a
    session/filesystem created at levanter import time already points at cwobject; THEN patch
    build_kvstore_spec on the (now-imported) levanter modules."""
    if _virtual_hosted_enabled():
        _override_s3_env_in_process()
        _configure_fsspec_virtual_hosted()
        _override_fsspec_conf()
        _patch_s3fs_client()
        _patch_s3fs_makedirs()
        klen = len(os.environ.get("AWS_ACCESS_KEY_ID", ""))
        print(f"[cw_patch] virtual-hosted S3 ENABLED (endpoint={os.environ.get('AWS_ENDPOINT_URL')}, "
              f"region={os.environ.get('AWS_DEFAULT_REGION')}, key_len={klen}, "
              f"MARIN_PREFIX={os.environ.get('MARIN_PREFIX')}, "
              f"AWS_CONFIG_FILE={os.environ.get('AWS_CONFIG_FILE')})", flush=True)
    import levanter.tensorstore_serialization as _tss
    _tss.build_kvstore_spec = _build_kvstore_spec
    # jagged_array did `from levanter.tensorstore_serialization import build_kvstore_spec`,
    # so it holds its own name binding that must be patched too.
    try:
        import levanter.store.jagged_array as _ja
        _ja.build_kvstore_spec = _build_kvstore_spec
    except Exception:
        pass
    # mkdirs lives in levanter.utils.fsspec_utils -> patch after levanter is importable.
    if _virtual_hosted_enabled():
        _patch_mkdirs()


# apply on import for convenience
apply()
