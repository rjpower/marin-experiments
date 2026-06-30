# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable
from typing import TypeVar

from fray.cluster import ResourceConfig
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, create_environment
from marin.training.run_environment import extras_for_resources
from marin.training.training import resolve_training_env

logger = logging.getLogger(__name__)

ConfigT = TypeVar("ConfigT")

# Runtime-tuning env vars forwarded from the dispatcher to the train tasks.
# Iris tasks don't inherit the submitter's shell, so anything the launcher was
# given (e.g. `iris job run -e XLA_FLAGS ...`) must be re-exported explicitly.
# JAX_PLATFORMS is excluded: the dispatcher runs CPU-only and its value must
# not leak onto accelerator tasks.
# RAGGED_DOT_IMPL is read at *runtime* inside the train task by
# haliax.nn.ragged_dot._preferred_implementations (via os.environ), so it must
# reach the worker. Without it the MoE grouped matmul stays on "auto", whose
# triton kernel raises at large token counts and silently falls back to the XLA
# dense ragged_dot_general -- a per-device [M, G, N] materialization that OOMs
# (hundreds of GiB) at our deeply-sparse 15B geometry. Forcing "triton" keeps it
# streamed/bounded.
_FORWARDED_ENV_PREFIXES = ("XLA_FLAGS", "LIBTPU_INIT_ARGS", "NCCL_", "JAX_", "RAGGED_DOT_IMPL")
_FORWARDED_ENV_EXCLUDE = ("JAX_PLATFORMS",)


def _forwarded_env_vars() -> dict[str, str]:
    return {
        k: v for k, v in os.environ.items() if k.startswith(_FORWARDED_ENV_PREFIXES) and k not in _FORWARDED_ENV_EXCLUDE
    }


def _safe_job_suffix(run_id: str) -> str:
    """Sanitize run IDs into Fray/Iris-safe job-name suffixes."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", run_id)


def dispatch_grug_training_run(
    *,
    run_id: str,
    config: ConfigT,
    local_entrypoint: Callable[[ConfigT], None],
    resources: ResourceConfig,
    max_retries_failure: int = 3,
) -> None:
    """Submit a grug train entrypoint through Fray and wait for completion."""
    safe_run_id = _safe_job_suffix(run_id)
    env_vars = resolve_training_env(base_env=_forwarded_env_vars(), resources=resources)
    request = JobRequest(
        name=f"grug-train-{safe_run_id}",
        entrypoint=Entrypoint.from_callable(local_entrypoint, args=[config]),
        resources=resources,
        environment=create_environment(env_vars=env_vars, extras=extras_for_resources(resources)),
        max_retries_failure=max_retries_failure,
    )
    logger.info("Dispatching grug training via Fray: %s", request.name)
    job = current_client().submit(request)
    job.wait(raise_on_failure=True)
