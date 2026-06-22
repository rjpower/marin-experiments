# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve a trained Qwen3 (HF safetensors) over HTTP on an Iris TPU.

The worker entrypoint. Loads a Qwen3 checkpoint with the repo's tested loader
(:mod:`models.registry`), builds a tunix native :class:`Sampler`, and exposes a
tiny FastAPI server:

  * ``GET  /health``           -> ``{"status": "ok", "model": ...}`` once ready.
  * ``POST /generate``         -> run the sampler on one or many prompts.
  * ``POST /v1/completions``   -> a thin OpenAI-ish shim over ``/generate``.

When run inside an Iris job it reads its allocated named port from
``iris_ctx().get_port(SERVE_PORT_NAME)``, serves on ``0.0.0.0:<port>``, and
registers an Iris ENDPOINT at ``http://<advertise_host>:<port>`` so an external
client can reach it through the controller proxy (see :mod:`serving.query`). Off
cluster (no ``iris_ctx``) it falls back to ``SERVE_LOCAL_PORT`` and skips
registration, so the app is testable on a laptop.

Two task modes (``SERVE_TASK``):
  * ``coding`` (default) -- single-turn Qwen3 coding agent over ``/generate``.
  * ``calc``   -- a multi-turn Delphi CALC tool-use agent over ``/calc``: the
    model emits ``CALC(a * b)``, a server-side calculator runs, the result is
    injected back as ``Tool result: X`` and the loop repeats (mirrors the §8
    training rollout). Use ``SERVE_MODEL=delphi`` so the Delphi rope monkeypatch
    is installed.

Configuration (all env vars):
  * ``SERVE_CKPT``        -- checkpoint dir (local path, ``gs://...``, or an HF
    repo id like ``marin-community/delphi-3e18-447Mparams-1.2Btokens``). A
    ``gs://`` dir is downloaded via gcsfs; an HF repo via ``snapshot_download``.
  * ``SERVE_MODEL``       -- registry model name (default ``qwen3``; ``delphi``
    for calc).
  * ``SERVE_TASK``        -- ``coding`` (default) or ``calc``.
  * ``SERVE_CALC_STAGE``  -- calc curriculum stage ``t0``/``t1``/``t2``
    (default ``t1``); sets the system prompt, chain depth and cache sizing.
  * ``SERVE_COMPARE_ENDPOINTS`` -- comma-separated PROXY-ENCODED endpoint names
    (e.g. ``tunix.delphi-calc-base,tunix.delphi-calc-rl``) the calc dashboard
    renders side by side (empty => single-endpoint dashboard).
  * ``SERVE_MAX_PROMPT``  -- max prompt tokens the cache budgets (default 1024;
    coding only -- calc derives its sizing from the stage).
  * ``SERVE_MAX_NEW``     -- max generation tokens the cache budgets (default 512;
    coding only).
  * ``SERVE_ENDPOINT_NAME`` -- Iris endpoint name (default ``delphi-rl``).
  * ``SERVE_PORT_NAME``   -- Iris named port to bind (default ``http``).
  * ``SERVE_LOCAL_PORT``  -- off-cluster fallback port (default 8000).

CRITICAL: the sampler is ALWAYS called with an explicit ``top_p``. Without it the
tunix Sampler decodes GREEDILY and silently ignores temperature AND seed (a known
bug); every draw would be identical. For deterministic/greedy serving pass
``temperature=0.0``.

The tunix ``Sampler`` is NOT safe to call concurrently (the KV cache is shared),
so generation is serialized under a process-wide lock and uvicorn runs
single-process. One in-flight generation at a time is fine for "try it out".
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

logger = logging.getLogger("serving.serve")


# ---------------------------------------------------------------------------
# Request / response models (module-scope so FastAPI treats them as body models;
# pydantic models defined inside a function are not resolved as request bodies).
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
  """A ``/generate`` request: exactly one of ``prompt`` or ``prompts``."""

  prompt: Optional[str] = None
  prompts: Optional[List[str]] = None
  max_tokens: Optional[int] = None
  temperature: float = 0.8
  top_p: float = 1.0
  seed: int = 0
  stop: Optional[List[str]] = None

  def resolve_prompts(self) -> "tuple[List[str], bool]":
    """Returns ``(prompts, batched)``; raises ``ValueError`` if ill-formed."""
    if self.prompts is not None and self.prompt is not None:
      raise ValueError("Pass exactly one of `prompt` or `prompts`.")
    if self.prompts is not None:
      if not self.prompts:
        raise ValueError("`prompts` must be non-empty.")
      return list(self.prompts), True
    if self.prompt is not None:
      return [self.prompt], False
    raise ValueError("Pass `prompt` (str) or `prompts` (list[str]).")


class GenerateResponse(BaseModel):
  """A ``/generate`` response: ``text`` for one prompt, ``texts`` for a batch."""

  text: Optional[str] = None
  texts: Optional[List[str]] = None


class CompletionRequest(BaseModel):
  """An OpenAI-ish ``/v1/completions`` request (single prompt)."""

  prompt: str
  max_tokens: int = 256
  temperature: float = 0.8
  top_p: float = 1.0
  seed: int = 0
  stop: Optional[List[str]] = None


class CalcRequest(BaseModel):
  """A ``/calc`` request: a multiply problem in ``problem`` (or ``prompt``)."""

  problem: Optional[str] = None
  prompt: Optional[str] = None

  def resolve_problem(self) -> str:
    """Returns the problem text; raises ``ValueError`` if neither is given."""
    text = self.problem if self.problem is not None else self.prompt
    if text is None or not str(text).strip():
      raise ValueError("Pass a non-empty `problem` (or `prompt`).")
    return str(text)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ServeConfig:
  """Server configuration resolved from the environment."""

  ckpt: str
  model: str = "qwen3"
  task: str = "coding"
  calc_stage: str = "t1"
  compare_endpoints: str = ""
  max_prompt: int = 1024
  max_new: int = 512
  endpoint_name: str = "delphi-rl"
  port_name: str = "http"
  local_port: int = 8000
  # tunix's Sampler treats `temperature` as a STATIC (non-pytree) field and
  # branches on `top_p >= 1.0`, so every distinct (temperature, top_p) is a
  # separate XLA compile. A cold compile exceeds the controller proxy's 30s
  # timeout, so the server serves ONE warmed sampling preset (these values) and
  # only the per-request seed varies. temperature <= 0 always means greedy.
  #
  # top_p is pinned to 1.0 on purpose: at top_p>=1.0 the sampler does full
  # `jax.random.categorical` temperature sampling (with the seed) but SKIPS the
  # `jax.lax.top_k` sort over the whole 151936-vocab that any top_p<1.0 forces --
  # that sort is far slower to compile AND to run (it can blow the 30s proxy even
  # warm). top_p=1.0 is also what the report's rollout/eval used.
  sample_temperature: float = 0.7
  sample_top_p: float = 1.0

  @property
  def cache_size(self) -> int:
    """KV-cache size fixed at Sampler construction (prompt + new + slack)."""
    return self.max_prompt + self.max_new + 16

  @classmethod
  def from_env(cls) -> "ServeConfig":
    """Builds the config from ``SERVE_*`` env vars (``SERVE_CKPT`` required)."""
    ckpt = os.environ.get("SERVE_CKPT")
    if not ckpt:
      raise ValueError("SERVE_CKPT is required (a local dir or gs:// path).")
    return cls(
        ckpt=ckpt,
        model=os.environ.get("SERVE_MODEL", "qwen3"),
        task=os.environ.get("SERVE_TASK", "coding"),
        calc_stage=os.environ.get("SERVE_CALC_STAGE", "t1"),
        compare_endpoints=os.environ.get("SERVE_COMPARE_ENDPOINTS", ""),
        max_prompt=int(os.environ.get("SERVE_MAX_PROMPT", "1024")),
        max_new=int(os.environ.get("SERVE_MAX_NEW", "512")),
        endpoint_name=os.environ.get("SERVE_ENDPOINT_NAME", "delphi-rl"),
        port_name=os.environ.get("SERVE_PORT_NAME", "http"),
        local_port=int(os.environ.get("SERVE_LOCAL_PORT", "8000")),
        sample_temperature=float(os.environ.get("SERVE_TEMPERATURE", "0.7")),
        sample_top_p=float(os.environ.get("SERVE_TOP_P", "1.0")),
    )


# An HF repo id: ``<org-or-user>/<repo>`` (word chars, dots, dashes). Matched only
# AFTER ``gs://`` and an existing local path are ruled out, so a real local dir
# named like a repo still wins. Lets calc serve raw base Delphi straight from
# ``marin-community/delphi-3e18-447Mparams-1.2Btokens`` with no pre-staging.
_HF_REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")


def _maybe_download_ckpt(ckpt: str) -> str:
  """Returns a local dir for ``ckpt``; downloads a ``gs://`` dir or an HF repo.

  Resolution order: a ``gs://`` dir is downloaded via ``gcsfs`` (in the locked
  venv) rather than the ``gsutil``/``gcloud`` CLIs (absent on the iris worker
  image); an existing local path is returned as is; otherwise, if ``ckpt`` looks
  like an HF repo id (``org/repo``) it is fetched with
  ``huggingface_hub.snapshot_download``.

  Args:
    ckpt: a local path, a ``gs://bucket/path`` directory, or an HF repo id.

  Returns:
    A local directory containing ``config.json`` + ``*.safetensors`` + tokenizer.
  """
  if ckpt.startswith("gs://"):
    import gcsfs

    dst = tempfile.mkdtemp(prefix="serve_ckpt_")
    src = ckpt.rstrip("/")
    logger.info("Downloading checkpoint %s -> %s", src, dst)
    fs = gcsfs.GCSFileSystem()
    for remote in fs.ls(src):
      name = remote.rsplit("/", 1)[-1]
      if name:  # skip the directory placeholder, if any
        fs.get_file(remote, os.path.join(dst, name))
    return dst
  if os.path.exists(ckpt):
    return ckpt
  if _HF_REPO_RE.match(ckpt):
    from huggingface_hub import snapshot_download

    dst = tempfile.mkdtemp(prefix="serve_ckpt_")
    logger.info("Downloading HF repo %s -> %s", ckpt, dst)
    return snapshot_download(repo_id=ckpt, local_dir=dst)
  return ckpt


# ---------------------------------------------------------------------------
# The model "engine": load + sampler + a serialized generate()
# ---------------------------------------------------------------------------
class GenerationEngine:
  """Owns the loaded model, tokenizer, sampler, mesh, and a generation lock."""

  def __init__(self, cfg: ServeConfig):
    self._cfg = cfg
    self._lock = threading.Lock()
    self._ready = False
    self._model = None
    self._tokenizer = None
    self._sampler = None
    self._mesh = None
    self._eos_tokens: List[int] = []

  @property
  def ready(self) -> bool:
    return self._ready

  def load(self) -> None:
    """Loads the model + tokenizer + tunix Sampler (blocking; call once)."""
    import jax.numpy as jnp

    from environments.coding_agent_env import program_terminal_eos_tokens
    from models.registry import get_model_spec
    from tunix.generate import sampler as sampler_lib

    cfg = self._cfg
    local_dir = _maybe_download_ckpt(cfg.ckpt)
    mesh = _build_mesh_or_none()
    self._mesh = mesh

    spec = get_model_spec(cfg.model)
    logger.info("Loading model %s from %s (mesh=%s)", spec.name, local_dir, mesh)
    model = spec.load_model(local_dir, dtype=jnp.bfloat16, mesh=mesh)
    tokenizer = spec.load_tokenizer(local_dir)

    cache_config = sampler_lib.CacheConfig(
        cache_size=cfg.cache_size,
        num_layers=model.config.num_layers,
        num_kv_heads=model.config.num_kv_heads,
        head_dim=model.config.head_dim,
    )
    sampler = sampler_lib.Sampler(
        transformer=model, tokenizer=tokenizer, cache_config=cache_config
    )

    eos_tokens = sorted(
        set([tokenizer.eos_token_id]) | set(program_terminal_eos_tokens(tokenizer))
    )

    self._model = model
    self._tokenizer = tokenizer
    self._sampler = sampler
    self._eos_tokens = eos_tokens
    logger.info(
        "Loaded %s, cache_size=%d, %d eos tokens; warming up...",
        spec.name,
        cfg.cache_size,
        len(eos_tokens),
    )
    self.warmup()
    self._ready = True
    logger.info("Model ready: %s (warmed).", spec.name)

  def clamp_max_tokens(self, max_tokens: Optional[int]) -> int:
    """Clamps a request's ``max_tokens`` to the cache budget (>=1)."""
    budget = self._cfg.max_new
    if max_tokens is None or max_tokens <= 0:
      return budget
    return max(1, min(int(max_tokens), budget))

  def _stop_to_eos(self, stop: Optional[List[str]]) -> List[int]:
    """Maps an optional per-request ``stop`` string list to extra eos ids.

    Each stop string whose single-token decode (or whose tokenization to a lone
    id) matches is added to the default eos set. Strings that do not map to a
    single token are ignored for the token-level eos (the sampler only stops on
    eos token ids), so the default code-terminal eos set always applies.
    """
    eos = list(self._eos_tokens)
    if not stop:
      return eos
    extra: set[int] = set()
    for s in stop:
      try:
        ids = self._tokenizer.encode(s, add_special_tokens=False)
      except Exception:  # pragma: no cover - tokenizer-specific
        ids = []
      if len(ids) == 1:
        extra.add(int(ids[0]))
    return sorted(set(eos) | extra)

  def generate(
      self,
      prompts: List[str],
      *,
      max_tokens: Optional[int],
      temperature: float,
      top_p: float,
      seed: int,
      stop: Optional[List[str]],
  ) -> List[str]:
    """Runs the sampler on ``prompts`` (serialized under the generation lock).

    Args:
      prompts: the input strings (one or many).
      max_tokens: requested new tokens (clamped to the cache budget).
      temperature: sampling temperature (0.0 => greedy).
      top_p: nucleus top_p. ALWAYS forwarded explicitly (see module docstring).
      seed: sampling seed.
      stop: optional extra stop strings (single-token ones become eos ids).

    Returns:
      The decoded completions (echo stripped), one per prompt.
    """
    if not self._ready:
      raise RuntimeError("Engine not ready.")
    eos_tokens = self._stop_to_eos(stop)
    ctx = self._mesh if self._mesh is not None else contextlib.nullcontext()
    greedy = temperature is None or temperature <= 0.0
    with self._lock:
      with ctx:
        out = self._sample(
            prompts, greedy=greedy, seed=seed, eos_tokens=eos_tokens,
        )
    return list(out.text)

  def _sample(self, prompts, *, greedy, seed, eos_tokens):
    """Calls the tunix Sampler at the engine's FIXED, pre-warmed configuration.

    Everything that affects the XLA compile is held constant so a request never
    triggers a cold compile (which would exceed the controller proxy's 30s
    timeout):

    * Shape. ``max_generation_steps`` / ``max_prompt_length`` are constant
      (``cfg.max_new``); generation still stops early on eos, and per-request
      ``max_tokens`` is advisory.
    * Mode. ``greedy=True`` omits ``top_p`` entirely (tunix argmax -- passing
      ``top_p`` with ``temperature==0`` divides by zero => all-``!`` garbage).
      ``greedy=False`` uses the FIXED ``cfg.sample_temperature`` /
      ``cfg.sample_top_p`` (both compile-static in tunix), so only ``seed``
      varies between sampling requests.
    """
    cache_size = self._cfg.cache_size
    fixed_new = self._cfg.max_new
    kwargs = dict(
        input_strings=list(prompts),
        max_generation_steps=fixed_new,
        max_prompt_length=cache_size - fixed_new - 4,
        echo=False,
        eos_tokens=eos_tokens,
        seed=seed,
    )
    if not greedy:
      kwargs["temperature"] = self._cfg.sample_temperature
      kwargs["top_p"] = self._cfg.sample_top_p
    return self._sampler(**kwargs)

  def warmup(self) -> None:
    """Pre-compiles the two served graphs (greedy + the fixed sampling preset).

    Best-effort: a warmup failure is logged, not fatal -- the first matching
    request would then pay the compile. Called by :meth:`load` before the
    endpoint is registered so callers never hit a cold compile through the proxy.
    """
    ctx = self._mesh if self._mesh is not None else contextlib.nullcontext()
    for label, greedy in (("greedy", True), ("sampling", False)):
      try:
        with ctx:
          self._sample(["warmup"], greedy=greedy, seed=0, eos_tokens=self._eos_tokens)
        logger.info("Warmup (%s) compiled.", label)
      except Exception as exc:  # pragma: no cover - depends on accelerator
        logger.warning("Warmup (%s) failed (%s); first request will compile.", label, exc)


def _build_mesh_or_none():
  """Builds the training mesh, or returns None if there is no accelerator.

  Reuses :func:`training.train_multiturn._build_mesh`. On a CPU-only smoke host
  (or any failure constructing the device mesh) returns None so generation runs
  with a null context.
  """
  try:
    from training.train_multiturn import _build_mesh

    return _build_mesh()
  except Exception as exc:  # pragma: no cover - depends on accelerator
    logger.warning("No device mesh (%s); running with mesh=None.", exc)
    return None


# ---------------------------------------------------------------------------
# The CALC tool-use engine: the multi-turn rollout (mirrors §8 training).
# ---------------------------------------------------------------------------
# Per-stage constants, taken verbatim from training.train_agentic's T0/T1/T2
# configs: the few-shot system prompt, the episode step budget (env_max_steps),
# and the (max_prompt, max_new) the rollout sampler was sized for. The cache is
# sized for the WORST-CASE accumulated episode (system + every turn), so a deep
# chain's later turns -- whose prompt is system + all prior turns -- still fit.
_CALC_STAGES: Dict[str, Dict[str, int]] = {
    "t0": {"env_max_steps": 2, "max_prompt": 640, "max_new": 96},
    "t1": {"env_max_steps": 3, "max_prompt": 768, "max_new": 128},
    "t2": {"env_max_steps": 4, "max_prompt": 1024, "max_new": 192},
}


def _clean_final_answer(turn: str, tool_result: Optional[str]) -> str:
  """Extracts the verified product from a (possibly noisy) finish turn.

  The §8 reward only requires the gold to APPEAR in the final answer as a
  standalone integer, so a correct model often emits e.g. ``"420 * 7"`` -- the
  answer plus a trailing echo of the last multiplication -- and is still scored
  as solved. The verified product is the last executed tool result, so prefer it
  when it appears standalone in the turn; else fall back to the turn's leading
  integer; else the raw turn (e.g. a non-numeric finish). The raw turn is always
  preserved verbatim in the transcript, so this only cleans the headline answer.
  """
  if tool_result is not None and re.search(
      rf"(?<!\d){re.escape(str(tool_result))}(?!\d)", turn
  ):
    return str(tool_result)
  match = re.match(r"\s*(-?\d+)", turn)
  return match.group(1) if match else turn


class CalcAgentEngine:
  """Owns a Delphi model + a greedy tunix Sampler and runs the CALC tool loop.

  The loop mirrors the training rollout EXACTLY (see
  :mod:`environments.agentic_tools` / :mod:`training.agentic_common`):

    * The rollout prompt is the raw-text render of ``[system, user]`` with a
      trailing newline (``DelphiRawTextChatParser.parse(..., add_generation_prompt
      =True)`` with ``generation_suffix="\\n"``), i.e. ``f"{system}\\n{user}\\n"``.
    * Each turn is one greedy Sampler call stopping on a digit/``)``-terminated
      newline (Delphi never emits real EOS), parsed by :class:`CalcTextToolParser`.
      A parsed ``CALC(a * b)`` runs the stock :class:`CalculatorTool`; its result
      is injected back as ``Tool result: <result>`` and the loop continues. No
      parsed call => the model finished and the turn is the final answer.

  Greedy serve convention (see :class:`GenerationEngine`): ``temperature=0.0`` and
  ``top_p`` is OMITTED (passing it with ``temperature==0`` divides by zero in
  tunix => all-``!`` garbage). The single Sampler is not threadsafe, so a whole
  episode (its several Sampler calls) runs under one process-wide lock.
  """

  def __init__(self, cfg: ServeConfig):
    self._cfg = cfg
    self._stage = cfg.calc_stage.strip().lower()
    if self._stage not in _CALC_STAGES:
      raise ValueError(
          f"SERVE_CALC_STAGE={cfg.calc_stage!r} not in {sorted(_CALC_STAGES)}."
      )
    sizing = _CALC_STAGES[self._stage]
    self._env_max_steps = sizing["env_max_steps"]
    self._max_new = sizing["max_new"]
    # Cache sized for the worst-case accumulated episode: stage max_prompt covers
    # the system prompt (~177 tok) plus the accumulated turns, plus max_new plus
    # 32 tokens of headroom. Held constant at construction so no request compiles.
    self._cache_size = sizing["max_prompt"] + self._max_new + 32
    self._lock = threading.Lock()
    self._ready = False
    self._model = None
    self._tokenizer = None
    self._sampler = None
    self._mesh = None
    self._system = ""
    self._eos_tokens: List[int] = []

  @property
  def ready(self) -> bool:
    return self._ready

  def load(self) -> None:
    """Loads the Delphi model + tokenizer + greedy Sampler (blocking; call once)."""
    import jax.numpy as jnp

    from environments.agentic_tools import (
        T0_SYSTEM_PROMPT,
        T1_SYSTEM_PROMPT,
        T2_SYSTEM_PROMPT,
        newline_terminal_eos_tokens,
    )
    from models.delphi_qwen3 import DELPHI_EOS_ID
    from models.registry import get_model_spec
    from tunix.generate import sampler as sampler_lib

    cfg = self._cfg
    local_dir = _maybe_download_ckpt(cfg.ckpt)
    mesh = _build_mesh_or_none()
    self._mesh = mesh

    # MUST be the Delphi spec (load_delphi installs the rope monkeypatch); the
    # CALC surface and the eos rule are Delphi-specific.
    spec = get_model_spec(cfg.model)
    logger.info(
        "Loading calc model %s (stage=%s) from %s (mesh=%s)",
        spec.name, self._stage, local_dir, mesh,
    )
    model = spec.load_model(local_dir, dtype=jnp.bfloat16, mesh=mesh)
    tokenizer = spec.load_tokenizer(local_dir)

    cache_config = sampler_lib.CacheConfig(
        cache_size=self._cache_size,
        num_layers=model.config.num_layers,
        num_kv_heads=model.config.num_kv_heads,
        head_dim=model.config.head_dim,
    )
    sampler = sampler_lib.Sampler(
        transformer=model, tokenizer=tokenizer, cache_config=cache_config
    )

    self._system = {
        "t0": T0_SYSTEM_PROMPT, "t1": T1_SYSTEM_PROMPT, "t2": T2_SYSTEM_PROMPT,
    }[self._stage]
    # Delphi never emits real EOS; it stops on a digit/`)`-terminated newline.
    self._eos_tokens = sorted(
        set([DELPHI_EOS_ID]) | set(newline_terminal_eos_tokens(tokenizer))
    )

    self._model = model
    self._tokenizer = tokenizer
    self._sampler = sampler
    logger.info(
        "Loaded calc %s, cache_size=%d, %d eos tokens; warming up...",
        spec.name, self._cache_size, len(self._eos_tokens),
    )
    self.warmup()
    self._ready = True
    logger.info("Calc model ready: %s stage=%s (warmed).", spec.name, self._stage)

  def _sample_turn(self, context: str) -> str:
    """Runs ONE greedy Sampler turn on ``context`` (the FIXED, warmed shape)."""
    out = self._sampler(
        input_strings=[context],
        max_generation_steps=self._max_new,
        max_prompt_length=self._cache_size - self._max_new - 4,
        echo=False,
        eos_tokens=self._eos_tokens,
        temperature=0.0,  # greedy: OMIT top_p (top_p + temp==0 => /0 garbage)
        seed=0,
    )
    return out.text[0].strip()

  def run(self, user_problem: str) -> Dict[str, Any]:
    """Runs the CALC tool loop on one problem; returns the transcript + answer.

    Args:
      user_problem: the multiply problem, e.g. ``"47 * 53"`` or ``"Q: 47 * 53"``
        (a missing ``"Q: "`` prefix is added so it matches the few-shot demos).

    Returns:
      ``{"answer": str, "tool_result": str|None, "transcript": list, "steps":
      int}`` where ``transcript`` is a list of ``{"role", "content"}`` turns
      (``assistant`` turns and ``tool`` results, in order).
    """
    if not self._ready:
      raise RuntimeError("Engine not ready.")
    from environments.agentic_tools import (
        CalcTextToolParser,
        _extract_calculator_result,
    )
    from tunix.rl.agentic.tools.calculator_tool import CalculatorTool

    parser = CalcTextToolParser()
    calc = CalculatorTool(name="calculator", description="")
    if not user_problem.lstrip().startswith("Q:"):
      user_problem = "Q: " + user_problem
    # == DelphiRawTextChatParser.parse([system, user], add_generation_prompt=True)
    # with generation_suffix="\n" (system/user joined by \n, trailing \n).
    context = f"{self._system}\n{user_problem}\n"
    transcript: List[Dict[str, str]] = []
    last_result: Optional[str] = None

    ctx = self._mesh if self._mesh is not None else contextlib.nullcontext()
    with self._lock:
      with ctx:
        for _ in range(self._env_max_steps):
          turn = self._sample_turn(context)
          transcript.append({"role": "assistant", "content": turn})
          calls = parser.parse(turn)
          if not calls:
            # No tool call: the model finished -- this turn is the final answer.
            # Clean the headline (the raw turn stays verbatim in the transcript).
            return {
                "answer": _clean_final_answer(turn, last_result),
                "tool_result": last_result,
                "transcript": transcript,
                "steps": len(transcript),
            }
          args = calls[0].arguments
          result = _extract_calculator_result(
              calc.apply(a=args["a"], b=args["b"], op=args.get("op", "*"))
          )
          last_result = result
          transcript.append({"role": "tool", "content": result})
          # Next-turn prompt: append this assistant turn + the rendered tool
          # result, ending in a newline (the generation suffix) so the model
          # begins its next turn on a fresh line (matches the few-shot demos).
          context = f"{context}{turn}\nTool result: {result}\n"

    # Ran out of step budget mid-chain: the last produced content is the answer.
    return {
        "answer": transcript[-1]["content"] if transcript else "",
        "tool_result": last_result,
        "transcript": transcript,
        "steps": len(transcript),
    }

  def warmup(self) -> None:
    """Pre-compiles the greedy Sampler at the calc shape (one full episode).

    Best-effort: a warmup failure is logged, not fatal. Runs ``run("Q: 12 * 13")``
    once so the first real request -- which may make up to ``env_max_steps``
    Sampler calls -- is warm and stays under the controller proxy's 30s timeout.
    Called by :meth:`load` (with ``_ready`` still False, so it bypasses the guard
    by sampling directly).
    """
    try:
      with self._lock:
        ctx = self._mesh if self._mesh is not None else contextlib.nullcontext()
        with ctx:
          # One real turn compiles the fixed greedy shape that every turn reuses.
          self._sample_turn(f"{self._system}\nQ: 12 * 13\n")
      logger.info("Calc warmup compiled.")
    except Exception as exc:  # pragma: no cover - depends on accelerator
      logger.warning("Calc warmup failed (%s); first request will compile.", exc)


# ---------------------------------------------------------------------------
# Browser dashboard (served at "/" and "/dashboard"). Self-contained: no external
# assets. EVERY fetch uses `new URL(path, location.href)` so it resolves under the
# controller proxy's `/proxy/<encoded-name>/` prefix (the proxy does not rewrite
# HTML bodies, so absolute paths like `/generate` would escape the prefix).
# ---------------------------------------------------------------------------
DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>tunix · delphi-rl</title>
<style>
  :root { --bg:#0d1117; --panel:#161b22; --border:#30363d; --fg:#e6edf3;
          --muted:#8b949e; --accent:#2f81f7; --good:#3fb950; --err:#f85149; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  .wrap { max-width:900px; margin:0 auto; padding:26px 16px 64px; }
  header { display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; }
  h1 { font-size:20px; margin:0; font-weight:650; letter-spacing:-.01em; }
  .sub { color:var(--muted); font-size:13px; }
  .status { margin-left:auto; font-size:12px; color:var(--muted); white-space:nowrap; }
  .dot { display:inline-block; width:8px; height:8px; border-radius:50%;
         background:var(--muted); margin-right:6px; vertical-align:middle; }
  .dot.ok { background:var(--good); } .dot.bad { background:var(--err); }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:10px;
          padding:16px; margin-top:18px; }
  label { display:block; font-size:11px; color:var(--muted); margin-bottom:6px;
          text-transform:uppercase; letter-spacing:.05em; }
  textarea,input,select { font-family:inherit; color:var(--fg); background:#0b0f14;
          border:1px solid var(--border); border-radius:8px; }
  textarea { width:100%; min-height:120px; resize:vertical; padding:10px 12px;
          font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:13px; }
  .row { display:flex; gap:14px; flex-wrap:wrap; align-items:end; margin-top:14px; }
  .field { display:flex; flex-direction:column; }
  select,input[type=number] { padding:8px 10px; font-size:13px; }
  input[type=number] { width:110px; }
  button { margin-left:auto; background:var(--accent); border:none; border-radius:8px;
          padding:10px 22px; font-size:14px; font-weight:600; color:#fff; cursor:pointer; }
  button:disabled { opacity:.5; cursor:default; }
  .hint { color:var(--muted); font-size:12px; margin-top:12px; }
  .out { margin-top:16px; white-space:pre-wrap; word-break:break-word;
          font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:13px;
          background:#0b0f14; border:1px solid var(--border); border-radius:8px;
          padding:13px; min-height:64px; overflow-x:auto; }
  .out .prompt { color:var(--muted); } .out .gen { color:var(--good); }
  .out .err { color:var(--err); }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>tunix · <span id="model">delphi-rl</span></h1>
    <span class="sub">Qwen3-1.7B · Dr.GRPO coding agent · served on Iris TPU</span>
    <span class="status"><span id="dot" class="dot"></span><span id="statustext">connecting…</span></span>
  </header>
  <div class="card">
    <label for="prompt">Prompt</label>
    <textarea id="prompt">def is_prime(n):</textarea>
    <div class="row">
      <div class="field"><label for="mode">Decoding</label>
        <select id="mode"><option value="greedy">Greedy</option><option value="sample">Sample</option></select></div>
      <div class="field"><label for="seed">Seed</label><input id="seed" type="number" value="0"/></div>
      <button id="go">Generate</button>
    </div>
    <div class="hint">Cmd/Ctrl+Enter to run. Generation length is fixed server-side
      (compile stability); sampling uses the server's pinned temperature/top-p —
      vary the seed for different draws.</div>
    <div id="out" class="out"></div>
  </div>
</div>
<script>
const $ = id => document.getElementById(id);
const api = path => new URL(path, location.href).toString();
const esc = s => s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
async function health(){
  try {
    const r = await fetch(api('health'));
    const j = await r.json().catch(()=>({}));
    if (r.ok) { $('dot').className='dot ok'; $('statustext').textContent='ready';
                if (j.model) $('model').textContent=j.model; }
    else { $('dot').className='dot'; $('statustext').textContent=(j.detail||'loading…'); }
  } catch(e){ $('dot').className='dot bad'; $('statustext').textContent='unreachable'; }
}
async function generate(){
  const prompt=$('prompt').value, mode=$('mode').value;
  const body={ prompt, seed:parseInt($('seed').value||'0',10),
               temperature: mode==='greedy'?0:0.8, top_p:1.0 };
  $('go').disabled=true; $('out').innerHTML='<span class="prompt">generating…</span>';
  const t0=performance.now();
  try {
    const r=await fetch(api('generate'),{method:'POST',
            headers:{'content-type':'application/json'}, body:JSON.stringify(body)});
    const raw=await r.text();
    if(!r.ok){ $('out').innerHTML='<span class="err">'+r.status+' — '+esc(raw)+'</span>'; return; }
    const j=JSON.parse(raw);
    const gen=j.text!=null?j.text:JSON.stringify(j.texts,null,2);
    const dt=((performance.now()-t0)/1000).toFixed(1);
    $('out').innerHTML='<span class="prompt">'+esc(prompt)+'</span><span class="gen">'+esc(gen)+'</span>';
    $('statustext').textContent='ready · '+dt+'s';
  } catch(e){ $('out').innerHTML='<span class="err">'+esc(String(e))+'</span>'; }
  finally { $('go').disabled=false; }
}
$('go').addEventListener('click', generate);
$('prompt').addEventListener('keydown', e => { if((e.metaKey||e.ctrlKey)&&e.key==='Enter') generate(); });
health(); setInterval(health, 15000);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Calc dashboard (served at "/" and "/dashboard" in calc mode). ONE HTML with two
# render modes: if ``window.COMPARE`` is a non-empty array of PROXY-ENCODED
# endpoint names it renders side-by-side columns (each POSTing to a SIBLING proxy
# path ``../<encoded-name>/calc``); otherwise a single column POSTing to its OWN
# ``calc`` (proxy-relative). EVERY fetch is ``new URL(rel, location.href)`` so it
# resolves under the controller proxy's ``/proxy/<encoded-name>/`` prefix (the
# proxy never rewrites HTML/JSON bodies, so an absolute ``/calc`` would escape).
# ``__COMPARE_JSON__`` is replaced at serve time with the endpoints JSON list.
# ---------------------------------------------------------------------------
DASHBOARD_HTML_CALC = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>tunix · delphi-calc</title>
<style>
  :root { --bg:#0d1117; --panel:#161b22; --border:#30363d; --fg:#e6edf3;
          --muted:#8b949e; --accent:#2f81f7; --good:#3fb950; --err:#f85149;
          --tool:#d29922; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  .wrap { max-width:980px; margin:0 auto; padding:26px 16px 64px; }
  header { display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; }
  h1 { font-size:20px; margin:0; font-weight:650; letter-spacing:-.01em; }
  .sub { color:var(--muted); font-size:13px; }
  .status { margin-left:auto; font-size:12px; color:var(--muted); white-space:nowrap; }
  .dot { display:inline-block; width:8px; height:8px; border-radius:50%;
         background:var(--muted); margin-right:6px; vertical-align:middle; }
  .dot.ok { background:var(--good); } .dot.bad { background:var(--err); }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:10px;
          padding:16px; margin-top:18px; }
  label { display:block; font-size:11px; color:var(--muted); margin-bottom:6px;
          text-transform:uppercase; letter-spacing:.05em; }
  input { font-family:inherit; color:var(--fg); background:#0b0f14;
          border:1px solid var(--border); border-radius:8px; }
  input[type=text] { width:100%; padding:10px 12px;
          font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:14px; }
  .row { display:flex; gap:14px; flex-wrap:wrap; align-items:end; margin-top:14px; }
  button { margin-left:auto; background:var(--accent); border:none; border-radius:8px;
          padding:10px 22px; font-size:14px; font-weight:600; color:#fff; cursor:pointer; }
  button:disabled { opacity:.5; cursor:default; }
  .hint { color:var(--muted); font-size:12px; margin-top:12px; }
  .cols { display:flex; gap:16px; flex-wrap:wrap; margin-top:16px; }
  .col { flex:1 1 0; min-width:280px; }
  .col h2 { font-size:13px; margin:0 0 8px; font-weight:600; color:var(--fg); }
  .trace { white-space:pre-wrap; word-break:break-word;
          font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:13px;
          background:#0b0f14; border:1px solid var(--border); border-radius:8px;
          padding:13px; min-height:64px; overflow-x:auto; }
  .trace .asst { color:var(--fg); }
  .trace .calc { color:var(--accent); }
  .trace .tool { color:var(--tool); }
  .trace .ans { color:var(--good); font-weight:700; }
  .trace .err { color:var(--err); }
  .steps { color:var(--muted); font-size:12px; margin-top:8px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>tunix · <span id="model">delphi-calc</span></h1>
    <span class="sub">Delphi-447M · multi-turn CALC tool-use agent · served on Iris TPU</span>
    <span class="status"><span id="dot" class="dot"></span><span id="statustext">connecting…</span></span>
  </header>
  <div class="card">
    <label for="problem">Problem (chained multiply)</label>
    <input id="problem" type="text" value="47 * 53"/>
    <div class="row">
      <div class="hint">The agent emits <code>CALC(a * b)</code>; a server-side
        calculator runs and the result is fed back as <code>Tool result: X</code>,
        repeating until it answers. Try <code>12 * 5 * 7</code>. Enter to run.</div>
      <button id="go">Run</button>
    </div>
    <div id="cols" class="cols"></div>
  </div>
</div>
<script>
window.COMPARE = __COMPARE_JSON__;
const $ = id => document.getElementById(id);
const api = path => new URL(path, location.href).toString();
const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
// Compare columns POST to a SIBLING proxy path: the dashboard is served at
// /proxy/<thisname>/dashboard, so `../<name>/calc` resolves client-side to
// /proxy/<name>/calc, which the controller forwards to that endpoint.
const sibling = name => new URL('../' + name + '/calc', location.href).toString();
const COMPARE = Array.isArray(window.COMPARE) ? window.COMPARE : [];

// One render column: a label + a div the trace is painted into.
function makeColumn(title){
  const col=document.createElement('div'); col.className='col';
  const h=document.createElement('h2'); h.textContent=title; col.appendChild(h);
  const t=document.createElement('div'); t.className='trace'; t.textContent='—';
  col.appendChild(t);
  const s=document.createElement('div'); s.className='steps'; col.appendChild(s);
  return {col, trace:t, steps:s};
}

// Render the engine's {answer, tool_result, transcript, steps} as a chat trace.
function renderTrace(el, j){
  let html='';
  for (const m of (j.transcript||[])){
    if (m.role==='assistant'){
      const isCall=/CALC\\(/.test(m.content||'');
      html += '<div class="'+(isCall?'calc':'asst')+'">'+esc(m.content)+'</div>';
    } else if (m.role==='tool'){
      html += '<div class="tool">Tool result: '+esc(m.content)+'</div>';
    }
  }
  html += '<div class="ans">= '+esc(j.answer)+'</div>';
  el.trace.innerHTML=html;
  el.steps.textContent=(j.steps||0)+' step(s)'+(j.tool_result!=null?' · last tool result '+esc(j.tool_result):'');
}

async function runOne(el, url, problem){
  el.trace.innerHTML='<span class="asst">running…</span>'; el.steps.textContent='';
  try {
    const r=await fetch(url,{method:'POST',headers:{'content-type':'application/json'},
            body:JSON.stringify({problem})});
    const raw=await r.text();
    if(!r.ok){ el.trace.innerHTML='<span class="err">'+r.status+' — '+esc(raw)+'</span>'; return; }
    renderTrace(el, JSON.parse(raw));
  } catch(e){ el.trace.innerHTML='<span class="err">'+esc(String(e))+'</span>'; }
}

let COLUMNS=[];
function buildColumns(){
  const host=$('cols'); host.innerHTML=''; COLUMNS=[];
  if (COMPARE.length){
    // Compare mode: one column per endpoint, fixed labels base vs post-RL.
    const labels=['base (raw Delphi)','post-RL (SFT→Dr.GRPO)'];
    COMPARE.forEach((name, i) => {
      const el=makeColumn((labels[i]||name)+' · '+name); el.url=sibling(name);
      host.appendChild(el.col); COLUMNS.push(el);
    });
    $('go').textContent='Run both';
  } else {
    // Single mode: one column POSTing to our OWN /calc (proxy-relative).
    const el=makeColumn('delphi-calc'); el.url=api('calc');
    host.appendChild(el.col); COLUMNS.push(el);
  }
}

async function run(){
  const problem=$('problem').value.trim(); if(!problem) return;
  $('go').disabled=true;
  try { await Promise.all(COLUMNS.map(el => runOne(el, el.url, problem))); }
  finally { $('go').disabled=false; }
}

async function health(){
  try {
    const r = await fetch(api('health'));
    const j = await r.json().catch(()=>({}));
    if (r.ok) { $('dot').className='dot ok'; $('statustext').textContent='ready';
                if (j.model) $('model').textContent=j.model; }
    else { $('dot').className='dot'; $('statustext').textContent=(j.detail||'loading…'); }
  } catch(e){ $('dot').className='dot bad'; $('statustext').textContent='unreachable'; }
}

buildColumns();
$('go').addEventListener('click', run);
$('problem').addEventListener('keydown', e => { if(e.key==='Enter') run(); });
health(); setInterval(health, 15000);
</script>
</body>
</html>"""


def _render_calc_dashboard(compare_endpoints: str) -> str:
  """Returns the calc dashboard HTML with ``window.COMPARE`` filled in.

  Args:
    compare_endpoints: the raw ``SERVE_COMPARE_ENDPOINTS`` value (comma-separated
      proxy-encoded endpoint names, e.g.
      ``"tunix.delphi-calc-base,tunix.delphi-calc-rl"``). Empty => single mode.

  Returns:
    The dashboard HTML with the ``__COMPARE_JSON__`` sentinel replaced by a JSON
    array literal of the (trimmed, non-empty) endpoint names.
  """
  import json

  names = [n.strip() for n in compare_endpoints.split(",") if n.strip()]
  return DASHBOARD_HTML_CALC.replace("__COMPARE_JSON__", json.dumps(names))


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
def build_app(engine: GenerationEngine) -> "Any":
  """Builds the FastAPI app around ``engine``.

  Routes: ``GET /`` and ``GET /dashboard`` (browser UI, viewable through the
  controller proxy), ``GET /health``, ``POST /generate``, ``POST /v1/completions``.

  Args:
    engine: a (possibly not-yet-loaded) :class:`GenerationEngine`. ``/health``
      returns ok only once ``engine.ready`` is True.

  Returns:
    A configured ``fastapi.FastAPI`` instance.
  """
  from fastapi import FastAPI, HTTPException
  from fastapi.responses import HTMLResponse

  app = FastAPI(title="delphi-rl serving", version="0.1.0")

  @app.get("/", response_class=HTMLResponse)
  @app.get("/dashboard", response_class=HTMLResponse)
  def dashboard() -> str:
    """The browser dashboard (works under the controller proxy prefix)."""
    return DASHBOARD_HTML

  @app.get("/health")
  def health() -> dict:
    if not engine.ready:
      raise HTTPException(status_code=503, detail="model loading")
    return {"status": "ok", "model": engine._cfg.model}

  @app.post("/generate", response_model=GenerateResponse)
  def generate(req: GenerateRequest) -> GenerateResponse:
    if not engine.ready:
      raise HTTPException(status_code=503, detail="model loading")
    try:
      prompts, batched = req.resolve_prompts()
    except ValueError as exc:
      raise HTTPException(status_code=400, detail=str(exc)) from exc
    texts = engine.generate(
        prompts,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        seed=req.seed,
        stop=req.stop,
    )
    if batched:
      return GenerateResponse(texts=texts)
    return GenerateResponse(text=texts[0])

  @app.post("/v1/completions")
  def completions(req: CompletionRequest) -> dict:
    """A minimal OpenAI-ish completions shim mapping to ``/generate``."""
    if not engine.ready:
      raise HTTPException(status_code=503, detail="model loading")
    texts = engine.generate(
        [req.prompt],
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        seed=req.seed,
        stop=req.stop,
    )
    return {
        "object": "text_completion",
        "model": engine._cfg.model,
        "choices": [{"index": 0, "text": texts[0], "finish_reason": "stop"}],
    }

  # Stash for tests / introspection.
  app.state.engine = engine  # type: ignore[attr-defined]
  return app


def build_calc_app(engine: CalcAgentEngine) -> "Any":
  """Builds the FastAPI app for the calc (CALC tool-use) serving mode.

  Routes: ``GET /`` and ``GET /dashboard`` (the calc dashboard -- single or
  compare mode per ``SERVE_COMPARE_ENDPOINTS``), ``GET /health``, ``POST /calc``
  (run the multi-turn tool loop on one problem). There is no ``/generate`` here:
  the calc model is driven only through the multi-turn loop.

  Args:
    engine: a (possibly not-yet-loaded) :class:`CalcAgentEngine`. ``/health``
      returns ok only once ``engine.ready`` is True.

  Returns:
    A configured ``fastapi.FastAPI`` instance.
  """
  from fastapi import FastAPI, HTTPException
  from fastapi.responses import HTMLResponse

  app = FastAPI(title="delphi-calc serving", version="0.1.0")
  dashboard_html = _render_calc_dashboard(engine._cfg.compare_endpoints)

  @app.get("/", response_class=HTMLResponse)
  @app.get("/dashboard", response_class=HTMLResponse)
  def dashboard() -> str:
    """The calc dashboard (single or compare mode; proxy-safe fetches)."""
    return dashboard_html

  @app.get("/health")
  def health() -> dict:
    if not engine.ready:
      raise HTTPException(status_code=503, detail="model loading")
    return {"status": "ok", "model": engine._cfg.model, "stage": engine._stage}

  @app.post("/calc")
  def calc(req: CalcRequest) -> dict:
    """Runs the CALC tool loop on one problem; returns the transcript + answer."""
    if not engine.ready:
      raise HTTPException(status_code=503, detail="model loading")
    try:
      problem = req.resolve_problem()
    except ValueError as exc:
      raise HTTPException(status_code=400, detail=str(exc)) from exc
    return engine.run(problem)

  # Stash for tests / introspection.
  app.state.engine = engine  # type: ignore[attr-defined]
  return app


# ---------------------------------------------------------------------------
# Iris wiring: pick the port, serve in the background, register, block.
# ---------------------------------------------------------------------------
def _resolve_bind_port(cfg: ServeConfig):
  """Returns ``(port, iris_ctx_or_none, advertise_host_or_none)``.

  Inside an Iris job, reads the allocated named port and advertise host. Off
  cluster (no ``iris_ctx``) returns the local fallback port and ``None`` ctx so
  the caller skips registration.
  """
  try:
    from iris.client import iris_ctx
    from iris.cluster.client import get_job_info
  except Exception as exc:  # pragma: no cover - iris always present in this repo
    logger.warning("iris import failed (%s); serving locally.", exc)
    return cfg.local_port, None, None

  try:
    ctx = iris_ctx()
  except Exception as exc:
    logger.info("Not inside an Iris job (%s); serving locally on %d.", exc, cfg.local_port)
    return cfg.local_port, None, None

  port = ctx.get_port(cfg.port_name)
  info = get_job_info()
  advertise_host = info.advertise_host if info is not None else "127.0.0.1"
  return port, ctx, advertise_host


def _serve_uvicorn_background(app: "Any", port: int) -> "threading.Thread":
  """Starts uvicorn single-process on ``0.0.0.0:port`` in a daemon thread."""
  import uvicorn

  config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info", workers=1)
  server = uvicorn.Server(config)
  thread = threading.Thread(target=server.run, name="uvicorn", daemon=True)
  thread.start()
  return thread


def _wait_healthy(port: int, *, timeout_s: float = 600.0) -> None:
  """Polls ``http://127.0.0.1:<port>/health`` until 200 or ``timeout_s``."""
  import httpx

  deadline = time.time() + timeout_s
  url = f"http://127.0.0.1:{port}/health"
  while time.time() < deadline:
    try:
      r = httpx.get(url, timeout=5.0)
      if r.status_code == 200:
        logger.info("Local /health OK on port %d.", port)
        return
    except Exception:
      pass
    time.sleep(1.0)
  raise TimeoutError(f"Server not healthy on port {port} within {timeout_s}s.")


def _serve_register_block(cfg: ServeConfig, app: "Any", engine: "Any") -> None:
  """Serves ``app``, loads ``engine``, registers the Iris endpoint, blocks.

  Shared by both task modes (coding and calc): both build a (task-specific)
  FastAPI app + engine, then run the identical serve/load/register/block
  sequence. The endpoint is registered with the VERBATIM ``cfg.endpoint_name``
  (a leading-slash name bypasses the controller's namespace prefixing).

  Args:
    cfg: the resolved server config.
    app: the task-specific FastAPI app (its ``/health`` gates on ``engine.ready``).
    engine: the task-specific engine exposing ``load()`` (blocking) and ``ready``.
  """
  port, ctx, advertise_host = _resolve_bind_port(cfg)

  # Start serving BEFORE loading weights so /health (503 while loading) is live;
  # load in this thread, then flip engine.ready, then register.
  _serve_uvicorn_background(app, port)
  engine.load()
  _wait_healthy(port)

  if ctx is not None:
    address = f"http://{advertise_host}:{port}"
    metadata = {"model": cfg.model, "endpoint": cfg.endpoint_name, "task": cfg.task}
    endpoint_id = ctx.registry.register(cfg.endpoint_name, address, metadata)
    logger.info(
        "Registered Iris endpoint name=%r address=%s id=%s",
        cfg.endpoint_name,
        address,
        endpoint_id,
    )
  else:
    logger.info("No Iris context; serving locally on port %d (no registration).", port)

  logger.info("Serving. Blocking forever.")
  # Block forever; the daemon uvicorn thread does the work. Sleep loop so a
  # container kill / KeyboardInterrupt cleanly exits.
  try:
    while True:
      time.sleep(3600)
  except KeyboardInterrupt:  # pragma: no cover
    logger.info("Interrupted; exiting.")


def main() -> None:
  """Loads the model, serves over HTTP, registers the Iris endpoint, blocks.

  Branches on ``cfg.task``: ``coding`` (default) serves the single-turn Qwen3
  coding agent (:class:`GenerationEngine` + :func:`build_app`); ``calc`` serves
  the multi-turn Delphi CALC tool-use agent (:class:`CalcAgentEngine` +
  :func:`build_calc_app`). Everything after app/engine construction is shared
  (:func:`_serve_register_block`).
  """
  logging.basicConfig(level=logging.INFO, format="[serve] %(message)s")
  cfg = ServeConfig.from_env()
  logger.info("Config: %s", cfg)

  task = cfg.task.strip().lower()
  if task == "calc":
    calc_engine = CalcAgentEngine(cfg)
    app = build_calc_app(calc_engine)
    _serve_register_block(cfg, app, calc_engine)
  elif task == "coding":
    engine = GenerationEngine(cfg)
    app = build_app(engine)
    _serve_register_block(cfg, app, engine)
  else:
    raise ValueError(f"SERVE_TASK={cfg.task!r} not in {{'coding', 'calc'}}.")


if __name__ == "__main__":
  main()
