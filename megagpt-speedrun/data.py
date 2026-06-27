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
# {% generation %} region). These are tokenized INLINE on the worker at cooldown start
# (auto_build_caches=True) from HF and the TreeCache is written to R2 so a retry reuses it.
# The grug train step already feeds GrugLmExample.loss_weight into the fused-CE `weight` arg,
# so a ChatLmDatasetFormat component (mask_user_turns=True) yields assistant-only SFT loss with
# NO train-step change. The marin tokenizer's chat template carries the {% generation %} marker
# (required by mask_user_turns). NB: in marin-levanter>=0.2.28 ``ChatLmDatasetFormat.chat_template_kwargs``
# is a COLUMN-NAME str|None (per-example kwargs lookup), NOT a static dict -- passing a dict made the
# processor do ``dict in example`` -> ``TypeError: unhashable type: 'dict'`` and crashed the cache build.
# We leave it None (no per-example kwargs); enable_thinking is a no-op on the llama3-based marin template.
_SFT_R2_CACHE = "s3://marin-na/marin/tokenized/megagpt_sft_v2"  # per-component: /<name>/train (v2 = post-fix, clean rebuild)

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


def _sft_component(hf_id: str, name: str | None, revision: str | None, messages_field: str, tag: str):
    fmt = ChatLmDatasetFormat(
        messages_field=messages_field,
        mask_user_turns=True,  # assistant-only loss (needs the template's {% generation %} marker)
        pack=True,  # pack short conversations to fill seq -- big throughput win for SFT
        chat_template_kwargs=None,  # Jun-26 levanter: COLUMN-NAME str|None (per-example kwargs), NOT a static dict
    )
    src = HfDatasetSourceConfig(
        id=hf_id,
        name=name,
        revision=revision,
        format=fmt,
        splits=["train"],
        stream=True,
    )
    return DatasetComponent(
        source=src,
        cache_dir=f"{_SFT_R2_CACHE}/{tag}",
        format=fmt,
        pack=True,
        tags=[tag],
        split="train",
    )


def build_sft_mix():
    """SFT cooldown mixture (tulu-3 + smoltalk + OpenThoughts-Agent), chat-format, assistant-only loss.

    auto_build_caches=True -> tokenized inline from HF on the worker at cooldown start, written to
    R2 under ``megagpt_sft/``. Consumed by the cooldown run (resume-from-pretrain + LR decay->0).
    """
    components = {
        name: _sft_component(hf_id, cfg, rev, field, name)
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
