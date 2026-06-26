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


def apply() -> None:
    """Monkeypatch every import site of build_kvstore_spec. Idempotent."""
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
        print(f"[cw_patch] virtual-hosted S3 ENABLED (endpoint={os.environ.get('AWS_ENDPOINT_URL')})", flush=True)


# apply on import for convenience
apply()
