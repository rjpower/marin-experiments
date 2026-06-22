# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Query a :mod:`serving.serve` endpoint through the Iris controller proxy.

Runs anywhere that has ``marin-iris`` installed. Resolves the registered endpoint
to its controller proxy base URL with :class:`iris.actor.resolver.ProxyResolver`
and POSTs to ``/generate``.

How resolution works (verified against the iris source):
  ``ProxyResolver(controller_url).resolve(name).first().url`` returns
  ``{controller_url}/proxy/{encoded}`` where ``encoded`` is ``name`` with its
  leading slash stripped and remaining slashes replaced by dots. The controller's
  proxy route is ``/proxy/{endpoint_name}/{sub_path}``, so appending ``/generate``
  to that base URL forwards the request to the worker's registered
  ``http://<advertise_host>:<port>/generate``.

The FULL endpoint name a caller must pass to ``--endpoint``:
  ``serve.py`` calls ``ctx.registry.register("delphi-rl", ...)``. The registry
  auto-prefixes the job NAMESPACE (derived from the job id), so the resolvable
  wire name is ``/<namespace>/delphi-rl`` (e.g. ``/<user>/delphi-rl``). Pass that
  FULL slash-prefixed name here; the resolver dot-encodes it for the proxy path.

Discovering the full name:
  there is no dedicated ``iris endpoint list`` CLI, but the controller exposes a
  SQL query command over its endpoint store:

      iris --cluster=marin query "SELECT name, address FROM endpoints"

  (find the row whose ``name`` ends in ``/delphi-rl``). The same names are what
  ``iris actor call <full-name> ...`` expects (its help shows the
  ``/user/job/.../actor-0`` form).

CLI:
  python serving/query.py --controller <url> --endpoint /<ns>/delphi-rl \
      --prompt "def solve():" --max-tokens 128 --temperature 0.0

Equivalent raw curl (the proxy URL is ``/proxy/<encoded-name>/generate``; the
slashes in the full name become dots):
  # full name /alice/delphi-rl  -> encoded alice.delphi-rl
  curl -sS -X POST \
    "$CONTROLLER/proxy/alice.delphi-rl/generate" \
    -H 'content-type: application/json' \
    -d '{"prompt": "def solve():", "max_tokens": 128, "temperature": 0.0, "top_p": 1.0}'
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from typing import Optional


def resolve_base_url(controller_url: str, endpoint: str) -> str:
  """Resolves ``endpoint`` to its controller-proxy base URL.

  Args:
    controller_url: the controller URL (e.g. ``http://controller:8080``).
    endpoint: the FULL registered endpoint name (e.g. ``/alice/delphi-rl``).

  Returns:
    ``{controller_url}/proxy/{encoded}`` -- POST ``/generate`` onto this.
  """
  from iris.actor.resolver import ProxyResolver

  return ProxyResolver(controller_url).resolve(endpoint).first().url


def generate(
    controller_url: str,
    endpoint: str,
    *,
    prompt: Optional[str] = None,
    prompts: Optional[list] = None,
    max_tokens: Optional[int] = None,
    temperature: float = 0.8,
    top_p: float = 1.0,
    seed: int = 0,
    stop: Optional[list] = None,
    timeout_s: float = 120.0,
) -> dict:
  """POSTs a ``/generate`` request through the proxy and returns the JSON body.

  Args:
    controller_url: controller URL.
    endpoint: full registered endpoint name (slash-prefixed).
    prompt: a single prompt (mutually exclusive with ``prompts``).
    prompts: a batch of prompts.
    max_tokens: requested new tokens (server clamps to its cache budget).
    temperature: sampling temperature (0.0 => greedy).
    top_p: nucleus top_p (always sent; the server requires it).
    seed: sampling seed.
    stop: optional extra stop strings.
    timeout_s: HTTP timeout.

  Returns:
    The parsed JSON response (``{"text": ...}`` or ``{"texts": [...]}``).
  """
  import httpx

  base = resolve_base_url(controller_url, endpoint)
  url = base.rstrip("/") + "/generate"
  body: dict = {"temperature": temperature, "top_p": top_p, "seed": seed}
  if prompts is not None:
    body["prompts"] = prompts
  else:
    body["prompt"] = prompt
  if max_tokens is not None:
    body["max_tokens"] = max_tokens
  if stop is not None:
    body["stop"] = stop
  resp = httpx.post(url, json=body, timeout=timeout_s)
  resp.raise_for_status()
  return resp.json()


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
  """Parses the query CLI args."""
  p = argparse.ArgumentParser(description="Query a serve.py endpoint via the Iris proxy.")
  p.add_argument(
      "--controller",
      default=os.environ.get("IRIS_CONTROLLER"),
      help="Controller URL (e.g. http://127.0.0.1:PORT). Omit to use --cluster.",
  )
  p.add_argument(
      "--cluster",
      default=os.environ.get("IRIS_CLUSTER"),
      help="Cluster name (e.g. marin); establishes the controller tunnel like the CLI.",
  )
  p.add_argument(
      "--endpoint",
      required=True,
      help="FULL registered endpoint name, slash-prefixed (e.g. /alice/delphi-rl).",
  )
  p.add_argument("--prompt", help="A single prompt.")
  p.add_argument("--prompts", help="JSON list of prompts (overrides --prompt).")
  p.add_argument("--max-tokens", type=int, default=None)
  p.add_argument("--temperature", type=float, default=0.8)
  p.add_argument("--top-p", type=float, default=1.0)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--stop", help="JSON list of extra stop strings.")
  p.add_argument("--timeout", type=float, default=120.0)
  return p.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
  """Resolves the endpoint, issues one ``/generate``, prints the JSON result."""
  args = parse_args(argv)
  prompts = json.loads(args.prompts) if args.prompts else None
  stop = json.loads(args.stop) if args.stop else None
  if prompts is None and args.prompt is None:
    raise SystemExit("Pass --prompt or --prompts.")
  if not args.controller and not args.cluster:
    raise SystemExit("Pass --controller URL or --cluster NAME.")

  # When given --cluster (and no --controller), establish the controller tunnel
  # in-process -- exactly like serving/launch_serve.py -- and run inside it.
  if args.controller:
    ctx = contextlib.nullcontext(args.controller)
  else:
    from serving.launch_serve import resolve_controller_url

    resolved = resolve_controller_url(args)
    ctx = resolved if hasattr(resolved, "__enter__") else contextlib.nullcontext(resolved)

  with ctx as controller_url:
    base = resolve_base_url(controller_url, args.endpoint)
    print(f"[query] proxy base = {base}", flush=True)
    result = generate(
        controller_url,
        args.endpoint,
        prompt=args.prompt,
        prompts=prompts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        stop=stop,
        timeout_s=args.timeout,
    )
  print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
  main()
