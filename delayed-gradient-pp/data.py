# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned Nemotron-CC training mixture for the delayed-gradient PP experiment.

The experiment does not re-tokenize anything: it consumes the existing
production Nemotron-CC + code caches by pinning each component's output path to
its known GCS location (via ``with_output_path``). The executor verifies the
cache exists and skips the tokenize step, so the heavy
``experiments.defaults``/``pretraining_datasets`` dependency web is not needed.

All caches live under ``$MARIN_PREFIX`` (``gs://marin-eu-west4`` for this
experiment); the paths below are relative and resolved against it. Raw source
paths are intentionally dropped (``include_raw_paths=False``) — we only ever read
the materialized caches, never re-tokenize, so the launcher carries no
dependency on the raw corpora.
"""

from marin.execution.types import ExecutorStep, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, lm_mixture_data_config, tokenize
from marin.processing.tokenize.data_configs import TokenizerStep

# Must match the tokenizer that produced the pinned caches (and the training run).
LLAMA3_TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"

# Never read: caches are pinned by output path and raw paths are excluded from
# the data config, so this placeholder only satisfies TokenizeConfig validation.
_PLACEHOLDER_TRAIN_PATH = "placeholder"

# name -> (pinned cache path relative to MARIN_PREFIX, mixture weight in TiB).
# Nemotron split weights and code weights mirror experiments.pretraining_datasets
# (nemotron.py / dclm.py); the cache hashes are the llama3 overrides pinned there.
_COMPONENTS: dict[str, tuple[str, float]] = {
    "nemotron_cc/hq_actual": ("tokenized/nemotron_cc/hq_actual-5af4cc", 0.91351),
    "nemotron_cc/hq_synth": ("tokenized/nemotron_cc/hq_synth-3525e2", 2.72),
    "nemotron_cc/medium_high": ("tokenized/nemotron_cc/medium_high-d21701", 0.82471),
    "nemotron_cc/medium": ("tokenized/nemotron_cc/medium-d86506", 3.38),
    "nemotron_cc/medium_low": ("tokenized/nemotron_cc/medium_low-0fdb07", 1.54),
    "nemotron_cc/low_actual": ("tokenized/nemotron_cc/low_actual-cb3f2c", 0.70123),
    "nemotron_cc/low_synth": ("tokenized/nemotron_cc/low_synth-3c57b3", 0.62771),
    "starcoderdata": ("tokenized/starcoderdata-12f018/", 0.25),
    "proofpile_2": ("tokenized/proofpile_2-4a35c7/", 0.055),
}


def _pinned(name: str, cache_path: str) -> TokenizerStep:
    """A tokenize step pinned to an already-materialized cache (no re-tokenize)."""
    step = ExecutorStep(
        name=f"tokenized/{name}",
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=[_PLACEHOLDER_TRAIN_PATH],
            validation_paths=versioned([]),
            cache_path=this_output_path(),
            tokenizer=versioned(LLAMA3_TOKENIZER),
        ),
    )
    return step.with_output_path(cache_path)


def build_nemotron_mix():
    """Build the pinned Nemotron-CC + code training mixture (no validation sets).

    The convergence metric for this experiment is ``train/loss``, so no
    validation sets are attached.
    """
    components = {name: _pinned(name, cache) for name, (cache, _weight) in _COMPONENTS.items()}
    weights = {name: weight for name, (_cache, weight) in _COMPONENTS.items()}
    return lm_mixture_data_config(components, weights, include_raw_paths=False)
