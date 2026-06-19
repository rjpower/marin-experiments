# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Region-agnostic FineWeb-Edu training data for the adaptive-sparsity experiment.

Unlike ``delayed-gradient-pp`` (which pins already-tokenized Nemotron-CC caches to
their ``gs://marin-eu-west4`` GCS paths and is therefore locked to europe-west4),
this experiment uses the **HuggingFace-backed** pre-tokenized FineWeb-Edu cache.
Its source of truth is the HF Hub, not one GCS region: the executor re-downloads it
into whichever region's ``$MARIN_PREFIX`` bucket the TPU job runs in, so the run is
not region-locked and we can take v6e capacity wherever it is (us-east5, us-east1,
europe-west4) without tripping marin's cross-region read guard.

- ``marin-community/fineweb-edu-pretokenized-10B`` — ~10B tokens, single source.
- ``...-10M`` — a 10M-token subset for fast smoke checks.

The cache is tokenized with the marin tokenizer, which is llama3-equivalent (it is
registered in marin's ``_EQUIVALENT_TOKENIZERS`` with ``meta-llama/Meta-Llama-3.1-8B``),
so it is compatible with the llama3-sized grug model (vocab 128_256).
"""

from marin.processing.tokenize import lm_mixture_data_config
from marin.processing.tokenize.download_pretokenized import download_pretokenized_cache

# llama3-equivalent; the FineWeb-Edu caches below were tokenized with it.
MARIN_TOKENIZER = "marin-community/marin-tokenizer"

_FINEWEB_EDU_10B_REPO = "marin-community/fineweb-edu-pretokenized-10B"
_FINEWEB_EDU_10M_REPO = "marin-community/fineweb-edu-pretokenized-10M"


def build_fineweb_edu_mix(smoke: bool = False):
    """Build a single-source FineWeb-Edu training mixture (no validation sets).

    The convergence metric for this experiment is ``train/loss`` (matched-token loss
    across sparsity arms), so no validation sets are attached. Set ``smoke=True`` to
    use the 10M-token subset for a quick cluster smoke check.
    """
    repo = _FINEWEB_EDU_10M_REPO if smoke else _FINEWEB_EDU_10B_REPO
    name = "fineweb-edu-10M" if smoke else "fineweb-edu-10B"
    cache = download_pretokenized_cache(name, repo, MARIN_TOKENIZER)
    return lm_mixture_data_config(
        components={"fineweb_edu": cache},
        weights={"fineweb_edu": 1.0},
        include_raw_paths=False,
    )
