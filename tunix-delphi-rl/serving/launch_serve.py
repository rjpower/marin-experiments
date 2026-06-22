# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit :mod:`serving.serve` to Iris on a v6e-4 with a named ``http`` port.

A 1.7B-class model inference fits a small v6e-4 slice -- far cheaper than the
v6e-8 trainer. We submit ``python serving/serve.py`` with ``ports=["http"]`` so
the worker gets an allocated host port, and a generous preemption-retry count so
the registered endpoint reappears if the task is preempted (the endpoint vanishes
when the task dies).

PRIMARY PATH -- the Python ``IrisClient.submit`` API.
======================================================
The ``iris job run`` CLI does NOT expose a named-port flag (verified: the
``job run`` command in ``iris/cli/job.py`` accepts ``--tpu/--gpu/--cpu/--memory/
--disk/--extra/--region/--zone/--max-retries/...`` but no ``--port``/``--ports``;
``_submit_and_wait_job`` calls ``client.submit(...)`` with no ``ports=``). Named
ports are ONLY reachable through ``IrisClient.submit(ports=[...])``. So this
launcher is the canonical way to start the server.

    from pathlib import Path
    from iris.client import IrisClient
    from iris.cluster.types import (
        Entrypoint, ResourceSpec, EnvironmentSpec, tpu_device)
    from iris.cluster.constraints import region_constraint
    from rigging.timing import Duration

    client = IrisClient.remote(controller_url, workspace=Path("."))
    job = client.submit(
        entrypoint=Entrypoint.from_command("python", "serving/serve.py"),
        name="delphi-rl-serve",
        resources=ResourceSpec(
            cpu=8, memory="64GB", disk="60GB", device=tpu_device("v6e-4")),
        environment=EnvironmentSpec(
            env_vars={"SERVE_CKPT": "gs://.../ckpt", "SERVE_MODEL": "qwen3"},
            extras=("tpu",)),
        ports=["http"],
        constraints=[region_constraint(["europe-west4"])],
        max_retries_preemption=1000,
    )

NOTE: there is no equivalent ``iris job run`` CLI command, because the CLI cannot
request a named port. Use this launcher (or call ``IrisClient.submit`` directly).

Do NOT run this against a real controller as part of a smoke test -- it spends a
TPU. ``--dry-run`` (the default here) prints the resolved plan and exits.

Config via flags or env (flags win):
  * ``--controller``  (``IRIS_CONTROLLER``)   -- pre-tunneled controller URL.
  * ``--cluster``     (``IRIS_CLUSTER``)       -- named cluster (e.g. ``marin``); we
        resolve its YAML config and open the SSH tunnel ourselves (like the
        ``iris`` CLI) so no pre-tunneled URL is needed. Ignored if ``--controller``
        is set; only one is required to submit.
  * ``--ckpt``        (``SERVE_CKPT``)         -- checkpoint dir (local or gs://).
  * ``--model``       (``SERVE_MODEL``)        -- registry model name (qwen3).
  * ``--endpoint-name`` (``SERVE_ENDPOINT_NAME``) -- Iris endpoint name (delphi-rl).
  * ``--tpu``         (``SERVE_TPU``)          -- TPU variant (v6e-4).
  * ``--region``      (``SERVE_REGION``)       -- pin region(s) (comma-separated).
  * ``--max-prompt`` / ``--max-new``           -- cache budget (1024 / 512).
  * ``--name``                                 -- Iris job name (delphi-rl-serve).
  * ``--max-retries-preemption``               -- preemption retries (1000).
  * ``--dry-run / --no-dry-run``               -- print plan vs actually submit.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
from pathlib import Path
from typing import ContextManager, Optional, Union

logger = logging.getLogger(__name__)


def _build_env_vars(args: argparse.Namespace) -> dict:
  """Builds the worker ``SERVE_*`` env from the resolved args."""
  env = {
      "SERVE_CKPT": args.ckpt,
      "SERVE_MODEL": args.model,
      "SERVE_ENDPOINT_NAME": args.endpoint_name,
      "SERVE_PORT_NAME": args.port_name,
      "SERVE_MAX_PROMPT": str(args.max_prompt),
      "SERVE_MAX_NEW": str(args.max_new),
      "SERVE_TASK": args.task,
      "SERVE_CALC_STAGE": args.calc_stage,
  }
  # Only the dashboard host needs the compare list; omit when empty so single-mode
  # serves render their own one-model dashboard.
  if args.compare_endpoints:
    env["SERVE_COMPARE_ENDPOINTS"] = args.compare_endpoints
  return env


def build_submit_kwargs(args: argparse.Namespace) -> dict:
  """Builds the kwargs dict passed to ``IrisClient.submit`` (no client needed).

  Importing the iris types lazily keeps ``--dry-run`` plan-printing usable even
  where the iris cluster types are import-heavy, and lets unit tests construct
  the plan offline.

  Args:
    args: parsed CLI args.

  Returns:
    A kwargs dict ready to splat into ``client.submit(**kwargs)``.
  """
  from iris.cluster.constraints import region_constraint
  from iris.cluster.types import (
      EnvironmentSpec,
      Entrypoint,
      ResourceSpec,
      tpu_device,
  )

  kwargs: dict = {
      "entrypoint": Entrypoint.from_command("python", "serving/serve.py"),
      "name": args.name,
      "resources": ResourceSpec(
          cpu=args.cpu,
          memory=args.memory,
          disk=args.disk,
          device=tpu_device(args.tpu),
      ),
      "environment": EnvironmentSpec(
          env_vars=_build_env_vars(args),
          extras=tuple(args.extra),
      ),
      "ports": [args.port_name],
      "max_retries_preemption": args.max_retries_preemption,
      "max_retries_failure": args.max_retries_failure,
  }
  if args.region:
    regions = [r.strip() for r in args.region.split(",") if r.strip()]
    if regions:
      kwargs["constraints"] = [region_constraint(regions)]
  return kwargs


def _describe(kwargs: dict) -> str:
  """A human-readable one-block summary of a submit plan (for ``--dry-run``)."""
  res = kwargs["resources"]
  envspec = kwargs["environment"]
  lines = [
      "[launch-serve] PLAN (IrisClient.submit):",
      f"  name              = {kwargs['name']}",
      f"  entrypoint        = {kwargs['entrypoint'].command}",
      f"  resources.cpu     = {res.cpu}",
      f"  resources.memory  = {res.memory}",
      f"  resources.disk    = {res.disk}",
      f"  resources.device  = {res.device}",
      f"  ports             = {kwargs['ports']}",
      f"  extras            = {envspec.extras}",
      f"  max_retries_preempt = {kwargs['max_retries_preemption']}",
      f"  constraints       = {kwargs.get('constraints')}",
      "  env_vars:",
  ]
  for k, v in (envspec.env_vars or {}).items():
    lines.append(f"    {k} = {v}")
  return "\n".join(lines)


def resolve_controller_url(
    args: argparse.Namespace,
) -> Union[str, ContextManager[str]]:
  """Resolves how to reach the controller from ``--controller`` or ``--cluster``.

  Two modes, mirroring the ``iris`` CLI:

  * ``--controller`` set: returns the URL string as-is (it is assumed already
    reachable, e.g. an explicit URL or a tunnel the caller established).
  * else ``--cluster`` set: resolves the named cluster's YAML config from the
    iris config search paths, builds the provider bundle, discovers the
    controller address, and returns the *tunnel context manager* (NOT yet
    entered). Entering it opens the SSH tunnel and yields a
    ``http://127.0.0.1:PORT`` URL; exiting tears the tunnel down. This is the
    same sequence ``iris.cli.connect.require_controller_url`` performs.

  The tunnel is intentionally returned un-entered so the caller can keep it
  alive across ``client.submit(...)`` via a ``with`` block and tear it down
  afterwards.

  Args:
    args: parsed CLI args (uses ``args.controller`` and ``args.cluster``).

  Returns:
    Either a plain controller URL ``str`` (``--controller`` mode) or an
    un-entered context manager yielding the tunneled URL (``--cluster`` mode).

  Raises:
    SystemExit: if neither ``--controller`` nor ``--cluster`` is set, or the
      named cluster cannot be resolved.
  """
  if args.controller:
    return args.controller

  if not args.cluster:
    raise SystemExit(
        "Either --controller (or IRIS_CONTROLLER) or --cluster (or IRIS_CLUSTER) "
        "is required to submit."
    )

  # Resolve the named cluster to its YAML config exactly like the iris CLI:
  # iris/cli/main.py resolves the name via rigging.config_discovery against the
  # iris config search dirs, then loads it into an IrisConfig.
  from rigging.config_discovery import resolve_cluster_config

  from iris.cli.connect import IRIS_CLUSTER_CONFIG_DIRS
  from iris.cluster.backends.local.cluster import LocalCluster
  from iris.cluster.config import IrisConfig

  try:
    resolved = resolve_cluster_config(args.cluster, dirs=IRIS_CLUSTER_CONFIG_DIRS)
  except FileNotFoundError as exc:
    raise SystemExit(
        f"Unknown cluster {args.cluster!r}. Run `iris cluster list` to see "
        "available clusters."
    ) from exc
  logger.info("Resolved cluster %r to config: %s", args.cluster, resolved)
  print(f"[launch-serve] Resolved cluster {args.cluster!r} -> {resolved}", flush=True)

  iris_config = IrisConfig.load(str(resolved))
  bundle = iris_config.provider_bundle()

  # Discover the controller address, then hand back the tunnel context manager
  # un-entered (see require_controller_url in iris/cli/connect.py:94-117).
  if iris_config.proto.controller.WhichOneof("controller") == "local":
    # Local clusters have no remote controller to tunnel to; start it and wrap
    # the resulting address in a nullcontext so the caller's ``with`` still works.
    cluster = LocalCluster(iris_config.proto)
    controller_address = cluster.start()
    return contextlib.nullcontext(controller_address)

  controller_address = iris_config.controller_address()
  if not controller_address:
    controller_address = bundle.controller.discover_controller(
        iris_config.proto.controller
    )

  print(
      f"[launch-serve] Establishing SSH tunnel to controller {controller_address} ...",
      flush=True,
  )
  return bundle.controller.tunnel(address=controller_address)


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
  """Parses CLI args, defaulting from the environment."""
  p = argparse.ArgumentParser(description="Submit serving/serve.py to Iris.")
  p.add_argument("--controller", default=os.environ.get("IRIS_CONTROLLER"))
  p.add_argument(
      "--cluster",
      default=os.environ.get("IRIS_CLUSTER"),
      help="Named iris cluster (e.g. 'marin'); resolved + tunneled like the CLI. "
      "Used only when --controller is not set.",
  )
  p.add_argument("--ckpt", default=os.environ.get("SERVE_CKPT"))
  p.add_argument("--model", default=os.environ.get("SERVE_MODEL", "qwen3"))
  p.add_argument(
      "--task",
      default=os.environ.get("SERVE_TASK", "coding"),
      help="Serving task: 'coding' (single-turn /generate) or 'calc' (CALC tool agent).",
  )
  p.add_argument(
      "--calc-stage",
      default=os.environ.get("SERVE_CALC_STAGE", "t1"),
      help="CALC stage t0/t1/t2 (sizing comes from the stage); calc task only.",
  )
  p.add_argument(
      "--compare-endpoints",
      default=os.environ.get("SERVE_COMPARE_ENDPOINTS", ""),
      help="Comma-sep proxy-encoded endpoint names (e.g. tunix.delphi-calc-base,"
      "tunix.delphi-calc-rl) for the side-by-side compare dashboard.",
  )
  p.add_argument(
      "--endpoint-name",
      default=os.environ.get("SERVE_ENDPOINT_NAME", "delphi-rl"),
  )
  p.add_argument("--port-name", default=os.environ.get("SERVE_PORT_NAME", "http"))
  p.add_argument("--name", default=os.environ.get("SERVE_JOB_NAME", "delphi-rl-serve"))
  p.add_argument("--tpu", default=os.environ.get("SERVE_TPU", "v6e-4"))
  p.add_argument("--region", default=os.environ.get("SERVE_REGION", "europe-west4"))
  p.add_argument("--cpu", type=float, default=float(os.environ.get("SERVE_CPU", "8")))
  p.add_argument("--memory", default=os.environ.get("SERVE_MEMORY", "64GB"))
  p.add_argument("--disk", default=os.environ.get("SERVE_DISK", "60GB"))
  p.add_argument(
      "--extra",
      action="append",
      default=None,
      help="UV extra(s) to install on the worker (repeatable; default: tpu).",
  )
  p.add_argument("--max-prompt", type=int, default=int(os.environ.get("SERVE_MAX_PROMPT", "1024")))
  p.add_argument("--max-new", type=int, default=int(os.environ.get("SERVE_MAX_NEW", "512")))
  p.add_argument(
      "--max-retries-preemption",
      type=int,
      default=int(os.environ.get("SERVE_MAX_RETRIES_PREEMPTION", "1000")),
  )
  p.add_argument(
      "--max-retries-failure",
      type=int,
      default=int(os.environ.get("SERVE_MAX_RETRIES_FAILURE", "0")),
  )
  p.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
  p.add_argument("--no-dry-run", dest="dry_run", action="store_false")
  args = p.parse_args(argv)
  # ``action="append"`` with a non-None default would prepend; resolve here.
  if args.extra is None:
    args.extra = ["tpu"]
  return args


def main(argv: Optional[list] = None) -> None:
  """Resolves args, prints the plan, and (unless ``--dry-run``) submits."""
  args = parse_args(argv)
  if not args.ckpt:
    raise SystemExit("--ckpt (or SERVE_CKPT) is required.")

  kwargs = build_submit_kwargs(args)
  print(_describe(kwargs), flush=True)

  if args.dry_run:
    print("[launch-serve] --dry-run: not submitting. Pass --no-dry-run to submit.", flush=True)
    return

  # Resolve the controller URL (plain string for --controller, or a tunnel
  # context manager for --cluster). NOTE: this is only reached under
  # --no-dry-run, so a dry run never opens a tunnel or hits the network.
  resolved = resolve_controller_url(args)

  from iris.client import IrisClient

  # ``contextlib.nullcontext`` makes the plain-URL case use the same ``with``
  # block; the tunnel context manager stays alive across ``client.submit(...)``
  # and is torn down on exit.
  cm = resolved if hasattr(resolved, "__enter__") else contextlib.nullcontext(resolved)
  with cm as controller_url:
    print(f"[launch-serve] Using controller URL: {controller_url}", flush=True)
    client = IrisClient.remote(controller_url, workspace=Path("."))
    job = client.submit(**kwargs)
    print(f"[launch-serve] SUBMITTED job={job}", flush=True)
    print(
        "[launch-serve] Once running, discover the full endpoint name and query it "
        "with serving/query.py (see its docstring).",
        flush=True,
    )


if __name__ == "__main__":
  main()
