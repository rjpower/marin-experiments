# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Make the fused cross-entropy pallas kernel usable for large vocab on H100.

**Problem.** The installed ``marin-levanter`` ships a tuned block-size table whose
large-batch buckets return oversized weight tiles for a 128256-vocab LM head
(e.g. ``h_block=256, v_block=2048`` = 1 MiB). An NVIDIA H100's per-kernel shared
memory budget is only ~99 KiB, so the pallas GPU fused-CE kernel rejects that tile.
Worse, the lookup reports ``has_tuned_match=True``, so the autotune that *would*
find a fitting ``v_block`` is skipped -- the oversized tile fails the kernel's
shared-memory guard and the fused CE silently falls back to the **XLA** path, which
materializes the full ``[tokens, vocab]`` logits. For a 15B MoE at batch 512 that is
~34 GiB of logits per device (plus grads), and training OOMs.

**Fix.** Wrap ``infer_block_sizes_with_tuned_match`` and, on NVIDIA devices, clamp
the returned tile (reduce ``v_block`` first, then ``h_block``; both powers of two,
>=16) so ``h_block * v_block * itemsize(w)`` fits under the device budget. The
streaming pallas kernel is then selected and the full logits are never materialized.
No-op on TPU / non-NVIDIA. Import this module before any fused-CE call (model.py does).
"""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from levanter.kernels.pallas.fused_cross_entropy_loss import api as _api
from levanter.kernels.pallas.fused_cross_entropy_loss import tuned_block_sizes as _tbs
from levanter.kernels.pallas.fused_cross_entropy_loss.pallas_gpu import (
    _max_weight_tile_bytes_for_device,
)

_orig_infer = _tbs.infer_block_sizes_with_tuned_match
_SAFETY = 0.95  # leave headroom below the hard limit for the kernel's other smem use


def _floor_pow2(n: int) -> int:
    n = int(n)
    return 1 << (n.bit_length() - 1) if n >= 1 else 1


def _clamp_tile(bs, w_bytes: int, limit: int):
    budget = int(limit * _SAFETY)
    h, v = bs.h_block_size, bs.v_block_size
    if h * v * w_bytes <= budget:
        return bs
    # Reduce v_block first (preserves the h tile for matmul efficiency).
    v = max(16, min(v, _floor_pow2(budget // max(1, h * w_bytes))))
    if h * v * w_bytes > budget:  # v already at the floor -> shrink h too
        h = max(16, _floor_pow2(budget // max(1, v * w_bytes)))
    return dataclasses.replace(bs, h_block_size=h, v_block_size=v)


def _patched_infer(b, h, v, *, dtype, x_dtype=None, w_dtype=None, device_kind=None):
    bs, matched = _orig_infer(b, h, v, dtype=dtype, x_dtype=x_dtype, w_dtype=w_dtype, device_kind=device_kind)
    dk = device_kind
    if dk is None:
        try:
            devs = jax.devices()
            dk = devs[0].device_kind.lower() if devs else ""
        except Exception:
            dk = ""
    limit = _max_weight_tile_bytes_for_device((dk or "").lower())
    if limit is None:  # non-NVIDIA (TPU/CPU): leave the tuned result untouched
        return bs, matched
    w_bytes = jnp.dtype(w_dtype or dtype or jnp.bfloat16).itemsize
    clamped = _clamp_tile(bs, w_bytes, limit)
    # Force ``has_tuned_match=True`` on NVIDIA. The newer marin-levanter CE flow
    # (api.py) only uses our clamped block when ``has_tuned_match`` is True; when it's
    # False it runs an autotune sweep that, for vocab 128256 at large token counts
    # (seq>=4096), finds no viable candidate and silently falls back to the XLA path,
    # which materializes the full [tokens, vocab] logits (a ~500 GiB single alloc → OOM).
    # Returning True makes the CE use our smem-fitting tile directly and stream over
    # vocab, never materializing logits. Trades autotune block-size optimality for not
    # OOMing; throughput can be tuned later via the clamp heuristic.
    return clamped, True


_patched_infer.__wrapped__ = _orig_infer  # type: ignore[attr-defined]
# Patch both the defining module and the name already imported into `api`.
_tbs.infer_block_sizes_with_tuned_match = _patched_infer
_api.infer_block_sizes_with_tuned_match = _patched_infer
