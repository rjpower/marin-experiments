"""Synthetic in-memory dataset for THROUGHPUT benchmarking (enabled by SP_SYNTH_DATA=1).

Yields random-token ``GrugLmExample``s with NO storage / dataloader cold-start, so a benchmark
measures PURE COMPUTE throughput (tokens/sec, MFU) instead of data I/O -- essential when pushing
into the high-tok/s regime where a real loader would bottleneck the measurement. Loss is
meaningless here (random tokens); use real data (SP_DATA=cw/datakit) for convergence runs.

Wired in ``train.build_train_dataset``: when SP_SYNTH_DATA is truthy it returns a single-component
``MixtureDataset`` wrapping ``SyntheticGrugDataset`` (the mixture wrapper keeps the unconditional
mixture-stage callback in train.py happy).
"""
from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np

from levanter.data.dataset import AsyncDataset
from levanter.data.text.examples import GrugLmExample

# Effectively-infinite finite length: large enough that a benchmark never exhausts it, but
# is_finite=True so MixtureDataset's block sampler treats it as a normal (huge) component.
_BIG = 1 << 31


class SyntheticGrugDataset(AsyncDataset[GrugLmExample]):
    """Random-token examples; index -> deterministic random sequence (so retries are stable)."""

    def __init__(self, seq_len: int, vocab_size: int, doc_len: int = 1024):
        super().__init__()
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        # Packed-document segment ids (the FA4 CuTe kernel REQUIRES packed segment_ids). Mimic real
        # data: ~doc_len-token documents -> several segments per sequence (block-diagonal attn).
        chunk = max(1, min(int(doc_len), self.seq_len))
        self._segment_ids = jnp.asarray((np.arange(self.seq_len) // chunk).astype(np.int32))
        self._max_segments = int((self.seq_len + chunk - 1) // chunk)

    async def async_len(self) -> int:
        return _BIG

    def is_finite(self) -> bool:
        return True

    async def get_batch(self, indices: Sequence[int]) -> Sequence[GrugLmExample]:
        out = []
        for idx in indices:
            rng = np.random.default_rng(int(idx) & 0x7FFFFFFF)
            # tokens in [1, vocab) -- avoid id 0 (often pad); identical compute to real data.
            toks = jnp.asarray(rng.integers(1, self.vocab_size, size=self.seq_len, dtype=np.int32))
            out.append(
                GrugLmExample.causal(
                    tokens=toks, segment_ids=self._segment_ids, max_segments=self._max_segments
                )
            )
        return out


def make_synthetic_mixture(*, seq_len, vocab_size, key, stop_strategy, block_size):
    """Single-component MixtureDataset wrapping the synthetic dataset (keeps train.py's mixture
    stage callback working)."""
    from levanter.data.mixture import MixtureDataset

    ds = SyntheticGrugDataset(seq_len=seq_len, vocab_size=vocab_size)
    return MixtureDataset(
        datasets={"synthetic": ds},
        weights={"synthetic": 1.0},
        stop_strategy=stop_strategy,
        key=key,
        block_size=block_size,
    )
