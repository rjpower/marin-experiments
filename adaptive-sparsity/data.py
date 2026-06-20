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
