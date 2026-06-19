# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned Nemotron-CC training mixture + Paloma validation for the re-entrant experiment.

The experiment does not re-tokenize anything. It consumes the existing production
Nemotron-CC + code caches (for training) and the Paloma + uncheatable-eval caches
(for validation) by pinning each component's output path to its known GCS location
(via ``with_output_path``). The executor verifies the cache exists and skips the
tokenize step, so the heavy ``experiments.defaults`` / ``experiments.pretraining_datasets``
dependency web is not needed (those modules are not shipped in the ``marin-core`` wheel).

The training half is byte-identical to ``delayed-gradient-pp/data.py`` (same llama3
tokenizer, so the same content-addressed cache hashes). The validation half is what
this experiment adds: the headline metric is the **Paloma macro loss**, which requires
the validation sets to be attached, whereas the delayed-gradient-pp template trained on
``train/loss`` only.

All caches live under ``$MARIN_PREFIX`` (``gs://marin-us-central1`` for this experiment,
to be region-local to the v5p-8 pool); the paths below are relative and resolved against
it. Validation cache hashes were resolved with marin's own
``compute_output_path(default_validation_sets(llama3))`` and verified against the
materialized ``gs://marin-us-central1/tokenized/{paloma,uncheatable_eval}`` caches.
"""

from levanter.data.text import LmDataConfig
from marin.execution.types import ExecutorStep, this_output_path, versioned
from marin.processing.tokenize import (
    TokenizeConfig,
    add_validation_sets_to_mixture,
    lm_mixture_data_config,
    tokenize,
)
from marin.processing.tokenize.data_configs import TokenizerStep

# Must match the tokenizer that produced the pinned caches (and the training run).
LLAMA3_TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"

# Never read: caches are pinned by output path, so this placeholder only satisfies
# TokenizeConfig validation. When the executor sees the pinned cache exists it skips
# tokenization, and at eval time Levanter reads the tokenized cache, never the raw path.
_PLACEHOLDER_PATH = "placeholder"

# Training mixture. name -> (pinned cache path relative to MARIN_PREFIX, mixture weight in TiB).
# Identical to delayed-gradient-pp: the Nemotron split + code weights mirror
# experiments.pretraining_datasets (nemotron.py / dclm.py); the cache hashes are the
# llama3 overrides pinned there.
_TRAIN_COMPONENTS: dict[str, tuple[str, float]] = {
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

# Validation sets (Paloma + uncheatable-eval), keyed by the SAME names that
# ``experiments.defaults.default_validation_sets(llama3)`` produces, so the tagged
# evaluator derives the identical "paloma" parent tag that the headline macro loss reads.
# These are weight-0 in the mixture (eval-only).
_VALIDATION_COMPONENTS: dict[str, str] = {
    "paloma/4chan": "tokenized/paloma/4chan-496ad5",
    "paloma/c4_100_domains": "tokenized/paloma/c4_100_domains-2b6db7",
    "paloma/c4_en": "tokenized/paloma/c4_en-cf1f79",
    "paloma/dolma-v1_5": "tokenized/paloma/dolma-v1_5-d3bed7",
    "paloma/dolma_100_programing_languages": "tokenized/paloma/dolma_100_programing_languages-369132",
    "paloma/dolma_100_subreddits": "tokenized/paloma/dolma_100_subreddits-f25f70",
    "paloma/falcon-refinedweb": "tokenized/paloma/falcon-refinedweb-75d43b",
    "paloma/gab": "tokenized/paloma/gab-ccaced",
    "paloma/m2d2_s2orc_unsplit": "tokenized/paloma/m2d2_s2orc_unsplit-7dbcc1",
    "paloma/m2d2_wikipedia_unsplit": "tokenized/paloma/m2d2_wikipedia_unsplit-b33d23",
    "paloma/manosphere_meta_sep": "tokenized/paloma/manosphere_meta_sep-a07891",
    "paloma/mc4": "tokenized/paloma/mc4-ea36a2",
    "paloma/ptb": "tokenized/paloma/ptb-628036",
    "paloma/redpajama": "tokenized/paloma/redpajama-9d4ddd",
    "paloma/twitterAAE_HELM_fixed": "tokenized/paloma/twitterAAE_HELM_fixed-2e17c1",
    "paloma/wikitext_103": "tokenized/paloma/wikitext_103-1f5636",
    "uncheatable_eval/ao3_english": "tokenized/uncheatable_eval/ao3_english-bb5666",
    "uncheatable_eval/arxiv_computer_science": "tokenized/uncheatable_eval/arxiv_computer_science-2b4f07",
    "uncheatable_eval/arxiv_physics": "tokenized/uncheatable_eval/arxiv_physics-f4ad8c",
    "uncheatable_eval/bbc_news": "tokenized/uncheatable_eval/bbc_news-4df59f",
    "uncheatable_eval/github_cpp": "tokenized/uncheatable_eval/github_cpp-a9de07",
    "uncheatable_eval/github_python": "tokenized/uncheatable_eval/github_python-baab41",
    "uncheatable_eval/wikipedia_english": "tokenized/uncheatable_eval/wikipedia_english-6330df",
}


def _pinned(name: str, cache_path: str, *, is_validation: bool) -> TokenizerStep:
    """A tokenize step pinned to an already-materialized cache (no re-tokenize).

    ``is_validation`` puts the placeholder raw path on ``validation_paths`` (vs
    ``train_paths``) so the step matches the shape of the original validation steps;
    either way the cache is pinned by output path and the raw path is never read.
    """
    step = ExecutorStep(
        name=f"tokenized/{name}",
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=[] if is_validation else [_PLACEHOLDER_PATH],
            validation_paths=versioned([_PLACEHOLDER_PATH] if is_validation else []),
            cache_path=this_output_path(),
            tokenizer=versioned(LLAMA3_TOKENIZER),
        ),
    )
    return step.with_output_path(cache_path)


def build_nemotron_mix() -> LmDataConfig:
    """Build the pinned Nemotron-CC + code training mixture (no validation sets)."""
    components = {name: _pinned(name, cache, is_validation=False) for name, (cache, _w) in _TRAIN_COMPONENTS.items()}
    weights = {name: weight for name, (_cache, weight) in _TRAIN_COMPONENTS.items()}
    return lm_mixture_data_config(components, weights, include_raw_paths=False)


def build_validation_sets() -> dict[str, TokenizerStep]:
    """Build the pinned Paloma + uncheatable-eval validation steps (eval-only)."""
    return {name: _pinned(name, cache, is_validation=True) for name, cache in _VALIDATION_COMPONENTS.items()}


def build_nemotron_mix_with_validation() -> LmDataConfig:
    """Training mixture with the Paloma + uncheatable validation sets attached (weight 0).

    The depth-scaling eval reads the Paloma parent-tag macro loss off these sets; it is
    the headline metric for the whole study.
    """
    return add_validation_sets_to_mixture(build_nemotron_mix(), build_validation_sets())
