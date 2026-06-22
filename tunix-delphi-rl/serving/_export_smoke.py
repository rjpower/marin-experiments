# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TPU smoke for the HF export path on the REAL Qwen3-1.7B (no training).

Loads Qwen3-1.7B-Base into tunix on a sharded mesh exactly as the trainer does,
exports it to a ``gs://`` HF checkpoint, reloads that checkpoint, and reports
forward-logit parity. This de-risks the one thing the CPU roundtrip cannot:
``jax.device_get`` gathering *sharded* actor params to host during export, on
the real model dims and the real worker's GCS-write path.

Pass/fail gate = ``load_qwen3`` reloading the exported dir WITHOUT raising (it
hard-asserts key coverage + all-params-concrete); parity stats are reported.

  SMOKE_SAVE_PATH=gs://marin-us-east5/rl-checkpoints/export-smoke python serving/_export_smoke.py
"""

from __future__ import annotations

import os
import tempfile

import jax
import jax.numpy as jnp

from huggingface_hub import snapshot_download
from models.registry import get_model_spec
from models.qwen3_loader import load_qwen3
from serving.export_hf import save_qwen3_to_hf
from training.train_multiturn import _build_mesh


def main() -> None:
  save_path = os.environ["SMOKE_SAVE_PATH"].rstrip("/")
  save_dtype = os.environ.get("SMOKE_SAVE_DTYPE", "bfloat16")
  spec = get_model_spec("qwen3")
  model_dir = os.environ.get("CURRIC_MODEL_DIR") or f"./{spec.name}"
  if not os.path.exists(os.path.join(model_dir, "config.json")):
    snapshot_download(repo_id=spec.repo, local_dir=model_dir)

  print(f"[smoke] jax {jax.__version__} devices={jax.devices()}", flush=True)
  mesh = _build_mesh()
  model = spec.load_model(model_dir, dtype=jnp.float32, mesh=mesh)
  # The model shards activations over the fsdp axis, so the batch dim must be
  # divisible by the device count (the Sampler handles this for real serving;
  # here we replicate one prompt across all devices).
  nb = jax.device_count()
  row = jnp.arange(16, dtype=jnp.int32)
  toks = jnp.broadcast_to(row, (nb, 16))
  pos = jnp.broadcast_to(row, (nb, 16))
  with mesh:
    logits0, _ = model(toks, pos, None, None)
  logits0 = jax.device_get(logits0)

  print(f"[smoke] exporting real 1.7B actor -> {save_path} ({save_dtype})", flush=True)
  save_qwen3_to_hf(model, save_path, hf_config_dir=model_dir, save_dtype=save_dtype)

  import gcsfs

  fs = gcsfs.GCSFileSystem()
  remote = sorted(p.rsplit("/", 1)[-1] for p in fs.ls(save_path))
  print(f"[smoke] uploaded files: {remote}", flush=True)
  with tempfile.TemporaryDirectory() as dl:
    for p in fs.ls(save_path):
      fs.get_file(p, os.path.join(dl, p.rsplit("/", 1)[-1]))
    # Hard gate: reload must pass load_qwen3's key-coverage + concreteness asserts.
    # Load onto the same mesh so the sharded forward below matches the actor.
    m2 = load_qwen3(dl, mesh=mesh, dtype=jnp.float32)
    with mesh:
      logits1, _ = m2(toks, pos, None, None)
    logits1 = jax.device_get(logits1)

  import numpy as np

  amax = float(np.max(np.abs(logits0 - logits1)))
  argmatch = float(np.mean(np.argmax(logits0, -1) == np.argmax(logits1, -1)))
  print(
      f"[smoke] RELOAD OK. parity: max_abs_logit_diff={amax:.3e} "
      f"argmax_agree={argmatch:.3f} (bf16 export => small diffs expected)",
      flush=True,
  )
  print("[smoke] EXPORT SMOKE PASSED", flush=True)


if __name__ == "__main__":
  main()
