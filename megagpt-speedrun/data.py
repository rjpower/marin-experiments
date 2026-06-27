# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Training data for the adaptive-sparsity experiment.

Two mixtures are available:

- ``build_nemotron_mix`` — the **real marin grug-MoE training mixture** (``nemotron_mix``
  in ``marin-community/marin``: 7 Nemotron-CC quality splits + StarCoder + proof-pile-2,
  TiB-proportional weights). This is what the large (10B-100B token) runs use. The
  ``experiments.*`` package that defines it is not published in the marin wheels, so we
  reconstruct it here from the already-tokenized GCS caches: each component is a
  ``TokenizeConfig`` pointed at its existing ``gs://marin-<region>/tokenized/...`` cache
  (``include_raw_paths=False`` -> we read the cache directly and never re-tokenize).
  These caches are byte-for-byte replicated across regions (verified present and complete
  in us-east5, us-central2, us-east1, eu-west4, us-central1), so any region with a TPU
  slice works; we pin the data ``region`` to match the run's region for same-region reads.

- ``build_fineweb_edu_mix`` — the smaller HF-backed FineWeb-Edu cache used by the earlier
  ~1B-token sweep and for fast smoke checks. Unlike nemotron (pre-built in GCS), this one
  is downloaded per-region on first use.

All caches are llama3-tokenized (vocab 128_256); ``marin-community/marin-tokenizer`` is
llama3-equivalent and not gated, so we use it for every component (mixtures assert a single
shared tokenizer).
"""

import os

from levanter.data.text import (
    ChatLmDatasetFormat,
    DatasetComponent,
    HfDatasetSourceConfig,
    LmDataConfig,
    TextLmDatasetFormat,
)
from marin.execution.types import InputName
from marin.processing.tokenize import lm_mixture_data_config
from marin.processing.tokenize.download_pretokenized import download_pretokenized_cache
from marin.processing.tokenize.tokenize import TokenizeConfig

# llama3-equivalent, ungated; the nemotron + FineWeb-Edu caches were tokenized with it.
MARIN_TOKENIZER = "marin-community/marin-tokenizer"

_FINEWEB_EDU_10B_REPO = "marin-community/fineweb-edu-pretokenized-10B"
_FINEWEB_EDU_10M_REPO = "marin-community/fineweb-edu-pretokenized-10M"

# nemotron_mix: component -> (relative tokenized-cache dir, TiB-proportional weight).
# Cache hashes are the pinned ones from marin's experiments/pretraining_datasets/nemotron.py
# (and dclm.py for the two code datasets); weights are NEMOTRON_WEIGHTS + the dclm code
# weights. Verified to exist under gs://marin-<region>/tokenized/ in all major regions.
_NEMOTRON_COMPONENTS: dict[str, tuple[str, float]] = {
    "nemotron_cc/hq_actual": ("tokenized/nemotron_cc/hq_actual-5af4cc", 0.91351),
    "nemotron_cc/hq_synth": ("tokenized/nemotron_cc/hq_synth-3525e2", 2.72),
    "nemotron_cc/medium_high": ("tokenized/nemotron_cc/medium_high-d21701", 0.82471),
    "nemotron_cc/medium": ("tokenized/nemotron_cc/medium-d86506", 3.38),
    "nemotron_cc/medium_low": ("tokenized/nemotron_cc/medium_low-0fdb07", 1.54),
    "nemotron_cc/low_actual": ("tokenized/nemotron_cc/low_actual-cb3f2c", 0.70123),
    "nemotron_cc/low_synth": ("tokenized/nemotron_cc/low_synth-3c57b3", 0.62771),
    "starcoderdata": ("tokenized/starcoderdata-12f018", 0.25),
    "proofpile_2": ("tokenized/proofpile_2-4a35c7", 0.055),
}


def _cache_component(cache_path: str, tags: list[str]) -> TokenizeConfig:
    """A read-only reference to an already-built tokenized cache (no re-tokenization).

    ``TokenizeConfig`` requires a non-empty ``train_paths``, but ``include_raw_paths=False``
    in the mixture drops the raw paths from the Levanter source config, so the placeholder
    below is never read — only ``cache_path`` (the GCS cache dir) is.
    """
    return TokenizeConfig(
        train_paths=[cache_path],  # placeholder; dropped by include_raw_paths=False
        validation_paths=[],
        cache_path=cache_path,
        tokenizer=MARIN_TOKENIZER,
        tags=tags,
    )


def build_nemotron_mix(region: str = "us-east5"):
    """The real marin grug-MoE training mixture, read from region-local GCS caches.

    ``region`` selects which ``gs://marin-<region>/tokenized/...`` bucket to read; it should
    match the region the TPU job runs in so reads are same-region. The convergence metric for
    this experiment is ``train/loss`` (matched-token across sparsity arms), so no validation
    sets are attached.
    """
    components = {
        name: _cache_component(f"gs://marin-{region}/{rel}", tags=name.split("/"))
        for name, (rel, _) in _NEMOTRON_COMPONENTS.items()
    }
    weights = {name: w for name, (_, w) in _NEMOTRON_COMPONENTS.items()}
    return lm_mixture_data_config(
        components=components,
        weights=weights,
        include_raw_paths=False,
    )


def build_fineweb_edu_mix(smoke: bool = False):
    """Single-source FineWeb-Edu mixture (HF-backed, per-region download). For smoke checks."""
    repo = _FINEWEB_EDU_10M_REPO if smoke else _FINEWEB_EDU_10B_REPO
    name = "fineweb-edu-10M" if smoke else "fineweb-edu-10B"
    cache = download_pretokenized_cache(name, repo, MARIN_TOKENIZER)
    return lm_mixture_data_config(
        components={"fineweb_edu": cache},
        weights={"fineweb_edu": 1.0},
        include_raw_paths=False,
    )


# ---------------------------------------------------------------------------
# Nemotron mixture on R2 (the real CoreWeave pretraining data).
#
# CoreWeave (cw-us-east-02a) is R2-only and cannot read the gs:// nemotron caches.
# BUT the standard marin ``tokenized/`` levanter caches (the SAME ones build_nemotron_mix
# references via gs://) are mirrored byte-for-byte to R2 under
# ``s3://marin-na/marin/tokenized/<rel>/`` -- each is a real levanter TreeCache (with
# ``train/shard_ledger.json``), llama3-tokenized (vocab 128256). VERIFIED loadable via
# load_lm_dataset_cache: hq_actual alone is 537.6B tokens. No re-tokenize, no store build;
# we reuse _NEMOTRON_COMPONENTS (correct cache hashes + TiB-proportional marin weights) and
# just swap the gs:// prefix for the R2 one.
#
# (The datakit/tokenize/ *raw parquet* on R2 is NOT a loadable cache -- it has no ledger --
# and the datakit store_b9f9b109 mirror is missing its ledgers too; both are dead ends. The
# tokenized/ caches below are the working path.)
# ---------------------------------------------------------------------------
_R2_TOK_PREFIX = "s3://marin-na/marin"  # rel paths hang off this on R2

# Components VERIFIED to have a loadable train TreeCache (train/shard_ledger.json) on R2.
# NB: nemotron_cc hq_synth and medium (the two largest natural splits) are NOT mirrored to
# R2 with a ledger under any hash -- only the tiers below load. Weights are a deliberate
# quality tilt (best-available HQ up-weighted) rather than the natural TiB proportions,
# since dropping hq_synth/medium would otherwise skew the mix toward medium/low quality.
# lm_mixture normalizes, so these are relative. hq_actual alone is 537.6B tokens; the whole
# set is many T tokens -- far more than a 24h run consumes.
#   name (also the tag path) -> (relative tokenized/ cache root, weight)
_NEMOTRON_R2_COMPONENTS: dict[str, tuple[str, float]] = {
    "nemotron_cc/hq_actual": ("tokenized/nemotron_cc/hq_actual-5af4cc", 2.0),  # 537.6B, best available
    "nemotron_cc/medium_high": ("tokenized/nemotron_cc/medium_high-d21701", 1.5),
    "nemotron_cc/medium_low": ("tokenized/nemotron_cc/medium_low-5b94a4", 1.0),
    "nemotron_cc/low_actual": ("tokenized/nemotron_cc/low_actual-cb3f2c", 0.5),
    "nemotron_cc/low_synth": ("tokenized/nemotron_cc/low_synth-3c57b3", 0.4),
    "starcoderdata": ("tokenized/starcoderdata-12f018", 0.5),  # code -> structure/reasoning
    "proofpile_2": ("tokenized/proofpile_2-5ba7ac", 0.2),  # math/proofs
}

# A tiny single-split mixture for validating the R2 cache path quickly.
_NEMOTRON_R2_SMOKE = {"nemotron_cc/hq_actual": ("tokenized/nemotron_cc/hq_actual-5af4cc", 1.0)}


def _r2_cache_component(rel: str, tags: list[str], prefix: str = _R2_TOK_PREFIX) -> DatasetComponent:
    """A Levanter component over a ``tokenized/`` TreeCache root (split appended -> /train).

    ``prefix`` selects the store: ``_R2_TOK_PREFIX`` (Cloudflare R2, default) or
    ``_CW_TOK_PREFIX`` (CoreWeave cwobject mirror -- read with the cwobject env + the
    LEVANTER_S3_VIRTUAL_HOSTED virtual-hosted patch).
    """
    return DatasetComponent(
        source=None,
        cache_dir=InputName.hardcoded(f"{prefix}/{rel}"),
        format=TextLmDatasetFormat(),
        tags=tags,
        flat_cache=False,  # cache_root/<split>; the train split has the shard_ledger.json
    )


# cwobject mirror prefix: mirror_to_cw.py copies R2 ``marin/tokenized/<rel>`` ->
# ``marin-us-east-02a/marin/tokenized/<rel>`` (same rel paths / cache hashes as R2).
_CW_TOK_PREFIX = "s3://marin-us-east-02a/marin"


def build_nemotron_cw_mix(smoke: bool = False, only: list[str] | None = None):
    """Nemotron mixture read from the cwobject mirror (cluster-local). Identical caches/weights
    to ``build_nemotron_datakit_mix``; only the storage prefix differs. ``only`` restricts to a
    subset of component names (use the components that have finished mirroring)."""
    table = _NEMOTRON_R2_SMOKE if smoke else _NEMOTRON_R2_COMPONENTS
    if only:
        table = {k: v for k, v in table.items() if k in only}
        if not table:
            raise ValueError(f"build_nemotron_cw_mix: none of {only} in {list(_NEMOTRON_R2_COMPONENTS)}")
    components = {
        name: _r2_cache_component(rel, name.split("/"), prefix=_CW_TOK_PREFIX)
        for name, (rel, _) in table.items()
    }
    weights = {name: w for name, (_, w) in table.items()}
    return LmDataConfig(
        tokenizer=MARIN_TOKENIZER,
        cache_dir=None,
        components=components,
        train_weights=[(0, weights)],
        auto_build_caches=False,
    )


def build_nemotron_datakit_mix(smoke: bool = False):
    """The real R2 nemotron pretraining mixture (standard ``tokenized/`` caches, vocab 128256)."""
    table = _NEMOTRON_R2_SMOKE if smoke else _NEMOTRON_R2_COMPONENTS
    components = {name: _r2_cache_component(rel, name.split("/")) for name, (rel, _) in table.items()}
    weights = {name: w for name, (_, w) in table.items()}
    return LmDataConfig(
        tokenizer=MARIN_TOKENIZER,
        cache_dir=None,
        components=components,
        train_weights=[(0, weights)],
        auto_build_caches=False,
    )


def build_nemotron_datakit_eval_mix(smoke: bool = False):
    """Datakit mix for POST-HOC bpb eval (held-out slice of the pretrain distribution).

    The standard datakit components are ``source=None`` hardcoded ``/train`` TreeCaches with NO
    separate ``/validation`` split, so the validation machinery's ``build_caches("validation")``
    FileNotFound-fails on them. Here each component is ``flat_cache=True`` pointing AT the existing
    ``/train`` ledger dir: flat caches are SKIPPED for the validation split (datasets.py: flat +
    split!=train -> None), so no missing cache is read, and ``num_validation_sequences`` (set in
    launch.py under SP_EVAL=1) slices the held-out eval sequences from the loaded train dataset.
    Effectively held-out because the 24h run consumed only ~0.19% of the 2.84T-token corpus.
    """
    table = _NEMOTRON_R2_SMOKE if smoke else _NEMOTRON_R2_COMPONENTS
    components = {
        name: DatasetComponent(
            source=None,
            cache_dir=InputName.hardcoded(f"{_R2_TOK_PREFIX}/{rel}/train"),
            format=TextLmDatasetFormat(),
            tags=name.split("/"),
            flat_cache=True,  # cache_root IS the train ledger dir; validation split is skipped
        )
        for name, (rel, _) in table.items()
    }
    weights = {name: w for name, (_, w) in table.items()}
    return LmDataConfig(
        tokenizer=MARIN_TOKENIZER,
        cache_dir=None,
        components=components,
        train_weights=[(0, weights)],
        auto_build_caches=False,
    )


# ---------------------------------------------------------------------------
# SFT cooldown mixture (chat datasets, assistant-only loss via the chat template's
# {% generation %} region). tulu-3 + smoltalk + OpenThoughts-Agent. The grug train step
# already feeds GrugLmExample.loss_weight into the fused-CE `weight` arg, so a
# ChatLmDatasetFormat component (mask_user_turns=True) yields assistant-only SFT loss with NO
# train-step change. The marin tokenizer's chat template carries the {% generation %} marker.
#
# ROBUST READ PATH -- post-mortem of cool2-sft's SILENT ~4.5h data-loader hang at ~step 9.8k
# ("Data loading is taking a long time ... Waiting for 1024 items", climbing to 16,310s, job
# stays state=running, --max-retries cannot recover):
#   * The hang was NOT the inline tokenization. All 3 component caches built & consolidated
#     fine (tulu3/smoltalk/ot_agent finished ~05:23/05:34/05:42) ~20 min BEFORE the stall began
#     (~06:01). It was a READ-side stall: cool2-sft was launched via iris_jobs.py, whose BASE
#     env LACKS ``LEVANTER_TS_CACHE_LIMIT`` (launch_cw.sh sets it to 32GB). With the default 1GB
#     tensorstore read cache the block-shuffle window's ~2GB working set is evicted and
#     re-fetched from R2 ("re-fetch thrash"), and eventually one R2 GET hangs with NO timeout
#     -> the loader's get_batch() blocks forever (tensorstore s3 driver has no read timeout).
# FIX (defense in depth):
#   (1) auto_build_caches=False + source=None: read the STATIC pre-built TreeCache directly --
#       NO inline zephyr cache-build sub-job inside the training process at all. The cache is
#       already on R2 (all 3 components consolidated); (re)build it with build_sft_cache_build_mix().
#   (2) By default LOCALIZE the (tiny, 3.3GB) cache to the worker's LOCAL DISK at startup and
#       read it via the tensorstore "file" driver -> ZERO R2 reads during training -> a hung R2
#       GET is structurally impossible. SP_SFT_LOCAL=0 disables this (reads the R2 cache directly,
#       then relying on LEVANTER_TS_CACHE_LIMIT to hold the whole working set in RAM).
#   (3) iris_jobs.py `cool` sweep also exports LEVANTER_TS_CACHE_LIMIT (belt + braces for (2) off).
# NB marin-levanter>=0.2.28 ``ChatLmDatasetFormat.chat_template_kwargs`` is a COLUMN-NAME str|None
# (per-example kwargs lookup), NOT a static dict -- a dict made the processor do ``dict in example``
# -> ``TypeError: unhashable type: 'dict'``; we leave it None.
_SFT_R2_CACHE = "s3://marin-na/marin/tokenized/megagpt_sft_v2"  # per-component: /<name>/train (consolidated)

# name -> (hf id, hf config name, revision, messages column, mixture weight)
_SFT_COMPONENTS: dict[str, tuple[str, str | None, str | None, str, float]] = {
    # broad, high-quality instruction following -- the backbone of the chat behaviour
    "tulu3": ("allenai/tulu-3-sft-mixture", None, "55e9fd6", "messages", 0.5),
    # diverse multi-turn chat (format + everyday conversation)
    "smoltalk": ("HuggingFaceTB/smoltalk", "all", None, "messages", 0.4),
    # the user's pick: agentic / tool-use capability flavour (small weight -- it is hard for an
    # undertrained base, and uses a `conversations` column with the same {role,content} schema)
    "ot_agent": ("open-thoughts/OpenThoughts-Agent-v1-SFT", None, None, "conversations", 0.1),
}


def _sft_format(messages_field: str) -> ChatLmDatasetFormat:
    return ChatLmDatasetFormat(
        messages_field=messages_field,
        mask_user_turns=True,  # assistant-only loss (needs the template's {% generation %} marker)
        pack=True,  # pack short conversations to fill seq -- big throughput win for SFT
        chat_template_kwargs=None,  # COLUMN-NAME str|None in levanter>=0.2.28, NOT a static dict
    )


def _localize_sft_cache(names: list[str], s3_prefix: str = _SFT_R2_CACHE, local_root: str | None = None) -> str:
    """Mirror each component's consolidated TreeCache (shard_ledger.json + input_ids/ +
    assistant_masks/) from R2 to local disk, then return the local root so training reads it via
    the tensorstore "file" driver -- no R2 reads during training => the cool2-sft hung-GET hang
    is structurally impossible. Uses fsspec (which, unlike tensorstore's raw s3 driver, has
    request timeouts/retries), so a download problem CRASHES loudly + is --max-retries-recoverable
    rather than hanging silently.

    Idempotent: a component whose local ``.localized`` marker exists is skipped (instant on a
    --max-retries resume that kept the pod; ~30s-2min cold for the ~3.3GB set on a fresh pod).
    """
    import fsspec

    local_root = local_root or os.environ.get("SP_SFT_LOCAL_DIR", "/tmp/megagpt_sft_v2")
    fs = fsspec.filesystem(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL"),
        key=os.environ.get("AWS_ACCESS_KEY_ID"),
        secret=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )
    s3_base = s3_prefix.replace("s3://", "")  # fsspec s3 paths are bucket/key (no scheme)
    for name in names:
        dst = os.path.join(local_root, name, "train")
        marker = os.path.join(dst, ".localized")
        if os.path.exists(marker):
            continue
        os.makedirs(dst, exist_ok=True)
        src = f"{s3_base}/{name}/train"
        fs.get(f"{src}/shard_ledger.json", os.path.join(dst, "shard_ledger.json"))
        # consolidated layout: the reader (TreeStore.open) only needs the flat field dirs + ledger
        for field in ("input_ids", "assistant_masks"):
            fs.get(f"{src}/{field}", os.path.join(dst, field), recursive=True)
        with open(marker, "w") as fh:
            fh.write("ok\n")
    return local_root


def _sft_static_component(name: str, messages_field: str, prefix: str) -> DatasetComponent:
    """Read-only component over a pre-built ChatLmDatasetFormat TreeCache (NO inline build).

    ``prefix`` is an absolute local dir (file driver) or ``s3://...`` (R2). ``flat_cache=False``
    appends ``/train``; the format must match the build's (ChatProcessor exemplar input_ids +
    assistant_masks). A metadata mismatch only WARNS (CacheLedger.load), never raises.
    """
    fmt = _sft_format(messages_field)
    return DatasetComponent(
        source=None,  # static cache: no inline tokenization / no zephyr build sub-job
        cache_dir=InputName.hardcoded(f"{prefix}/{name}"),  # abs path -> passed through unchanged
        format=fmt,
        pack=True,
        tags=[name],
        split="train",
        flat_cache=False,  # cache_root/train holds the shard_ledger.json
    )


def build_sft_mix() -> LmDataConfig:
    """SFT cooldown mixture, reading the STATIC pre-built cache (``auto_build_caches=False``).

    Default: localize the cache to local disk (``SP_SFT_LOCAL`` != 0) and read it via the file
    driver. Used by ``SP_DATA=sft``. Pair with ``SP_INIT_FROM`` (resume the pretrain ckpt) +
    ``SP_SCHEDULE=linear SP_MIN_LR=0`` (decay the WSD cooldown peak->0).
    """
    names = list(_SFT_COMPONENTS)
    prefix = _SFT_R2_CACHE
    if os.environ.get("SP_SFT_LOCAL", "1").strip().lower() not in ("0", "false", "no"):
        try:
            prefix = _localize_sft_cache(names)
        except Exception as e:  # noqa: BLE001 -- auto-degrade: never crash the cooldown on localize
            # Fall back to reading the static R2 cache directly; LEVANTER_TS_CACHE_LIMIT (32GB in
            # iris_jobs BASE) still holds the whole 3.3GB working set in RAM -> no re-fetch thrash.
            import logging

            logging.getLogger("data").warning(
                "SFT cache localize to local disk failed (%s); reading R2 static cache directly "
                "(relying on LEVANTER_TS_CACHE_LIMIT).", e
            )
    components = {
        name: _sft_static_component(name, field, prefix)
        for name, (_, _, _, field, _) in _SFT_COMPONENTS.items()
    }
    weights = {name: w for name, (_, _, _, _, w) in _SFT_COMPONENTS.items()}
    return LmDataConfig(
        tokenizer=MARIN_TOKENIZER,
        cache_dir=None,
        components=components,
        train_weights=[(0, weights)],
        auto_build_caches=False,
    )


def build_sft_cache_build_mix() -> LmDataConfig:
    """INLINE-BUILD path: tokenize tulu-3 + smoltalk + OpenThoughts-Agent from HF and write the
    TreeCaches to ``_SFT_R2_CACHE``. NOT used for training (that read-thrashes; see build_sft_mix).
    Run it ONCE to (re)build the static cache when the mixture changes, then build_sft_mix() reads
    it. Verify each ``<name>/train/shard_ledger.json`` has ``is_finished: true`` afterward.
    """
    def _src_component(hf_id, cfg, rev, field, name):
        fmt = _sft_format(field)
        src = HfDatasetSourceConfig(id=hf_id, name=cfg, revision=rev, format=fmt, splits=["train"], stream=True)
        return DatasetComponent(
            source=src, cache_dir=f"{_SFT_R2_CACHE}/{name}", format=fmt, pack=True, tags=[name], split="train"
        )

    components = {
        name: _src_component(hf_id, cfg, rev, field, name)
        for name, (hf_id, cfg, rev, field, _) in _SFT_COMPONENTS.items()
    }
    weights = {name: w for name, (_, _, _, _, w) in _SFT_COMPONENTS.items()}
    return LmDataConfig(
        tokenizer=MARIN_TOKENIZER,
        cache_dir=None,
        components=components,
        train_weights=[(0, weights)],
        auto_build_caches=True,
    )
