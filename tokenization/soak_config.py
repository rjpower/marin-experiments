# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Soak-run configuration shared by the training launcher and the bake-off driver.

Pure dataclasses with no training/grug imports, so the analysis pipeline (``bakeoff.py``) can read
the soak's default shape without pulling in ``train_model``'s grug launcher dependency.
"""

import dataclasses

STEPS_PER_EVAL = 1000


@dataclasses.dataclass(frozen=True)
class NgramSpec:
    """The hashed multi-gram input embedding (Over-Encoding / LongCat) config for a soak arm.

    Adds input-side capacity at zero extra serving FLOPs; ``ratio`` scales each n-gram term's
    init to the base embedding norm. The default is the proxy study's winning 64k config.
    """

    orders: tuple[int, ...] = (2, 3, 4)
    num_hashes: int = 2
    hash_buckets: int = 4_000_037
    rank: int = 128
    combine: str = "mean"
    ratio: float = 0.25


@dataclasses.dataclass(frozen=True)
class SoakParams:
    """Mesh, batch, and schedule knobs for one soak run (the ~10B/500M SCALE_* shape is inherited).

    ``ngram`` attaches the hashed n-gram embedding; ``None`` (the default) is the plain model.
    """

    replicas: int = 8
    expert_axis: int = 8
    replica_axis: int = 1
    batch_size: int = 512
    steps: int = 40_000
    processes_per_task: int = 1
    ram: str = "512g"
    steps_per_eval: int = STEPS_PER_EVAL
    mp: str = "params=float32,compute=bfloat16,output=bfloat16"
    ngram: NgramSpec | None = None
