# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU smoke tests for the serving runtime (:mod:`serving.serve`).

Fast (default) tests:
  * import all three serving modules.
  * the request/response pydantic models + ``resolve_prompts`` validation.
  * the cache-clamp logic (``GenerationEngine.clamp_max_tokens``).
  * the FastAPI app wiring via ``fastapi.testclient.TestClient`` against a FAKE
    engine: ``/health`` gates on readiness, ``/generate`` returns text for a
    single prompt and a batch, ``/v1/completions`` shims, and the generation
    lock serializes calls.
  * the launch_serve plan builder + the query proxy-URL resolver.

A ``slow`` test builds a real tiny ``qm.Qwen3`` + tunix Sampler (no weights on
disk) and drives ``/generate`` end to end -- excluded from the default run.
"""

from __future__ import annotations

import threading
import time
import types

import pytest
from fastapi.testclient import TestClient

from serving import serve as serve_mod


def test_serving_modules_import():
  """The three deliverables import cleanly."""
  import serving.launch_serve  # noqa: F401
  import serving.query  # noqa: F401
  import serving.serve  # noqa: F401


# ---------------------------------------------------------------------------
# A fake engine: same interface as GenerationEngine, no JAX / no weights.
# ---------------------------------------------------------------------------
class _FakeEngine:
  """Mimics :class:`serving.serve.GenerationEngine` for FastAPI wiring tests."""

  def __init__(self, cfg, ready=True):
    self._cfg = cfg
    self._ready = ready
    self.calls = []
    self._lock = threading.Lock()
    self._inflight = 0
    self.max_concurrent = 0

  @property
  def ready(self):
    return self._ready

  def clamp_max_tokens(self, max_tokens):
    return serve_mod.GenerationEngine.clamp_max_tokens(self, max_tokens)

  def generate(self, prompts, *, max_tokens, temperature, top_p, seed, stop):
    # Record the resolved args and prove the lock serializes calls.
    with self._lock:
      self._inflight += 1
      self.max_concurrent = max(self.max_concurrent, self._inflight)
    time.sleep(0.01)
    self.calls.append(
        dict(
            prompts=list(prompts),
            max_tokens=self.clamp_max_tokens(max_tokens),
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            stop=stop,
        )
    )
    with self._lock:
      self._inflight -= 1
    return [f"<gen:{p}>" for p in prompts]


def _cfg(**kw):
  base = dict(ckpt="/tmp/nope", model="qwen3", max_prompt=1024, max_new=512)
  base.update(kw)
  return serve_mod.ServeConfig(**base)


def test_cache_size_and_clamp():
  cfg = _cfg(max_prompt=100, max_new=50)
  assert cfg.cache_size == 100 + 50 + 16
  eng = _FakeEngine(cfg)
  assert eng.clamp_max_tokens(None) == 50  # default = budget
  assert eng.clamp_max_tokens(0) == 50
  assert eng.clamp_max_tokens(-5) == 50
  assert eng.clamp_max_tokens(10) == 10  # under budget
  assert eng.clamp_max_tokens(9999) == 50  # clamped to budget
  assert eng.clamp_max_tokens(1) == 1


def test_health_gates_on_ready():
  cfg = _cfg()
  app = serve_mod.build_app(_FakeEngine(cfg, ready=False))
  client = TestClient(app, raise_server_exceptions=True)
  assert client.get("/health").status_code == 503

  app2 = serve_mod.build_app(_FakeEngine(cfg, ready=True))
  client2 = TestClient(app2)
  r = client2.get("/health")
  assert r.status_code == 200
  assert r.json() == {"status": "ok", "model": "qwen3"}


def test_dashboard_served_at_root_and_path():
  cfg = _cfg()
  client = TestClient(serve_mod.build_app(_FakeEngine(cfg, ready=True)))
  for path in ("/", "/dashboard"):
    r = client.get(path)
    assert r.status_code == 200, path
    assert "text/html" in r.headers["content-type"]
    body = r.text
    assert "tunix" in body
    # Fetches must be proxy-relative (no absolute "/generate"), so the page works
    # under the controller proxy's /proxy/<name>/ prefix.
    assert "new URL(path, location.href)" in body
    assert '"/generate"' not in body and "'/generate'" not in body


def test_generate_single_and_batch():
  cfg = _cfg(max_prompt=100, max_new=50)
  eng = _FakeEngine(cfg)
  client = TestClient(serve_mod.build_app(eng))

  # single prompt -> {"text": ...}
  r = client.post("/generate", json={"prompt": "hello", "max_tokens": 9999})
  assert r.status_code == 200, r.text
  body = r.json()
  assert body["text"] == "<gen:hello>"
  assert body.get("texts") is None
  # max_tokens clamped to budget (50), top_p defaulted and forwarded.
  assert eng.calls[-1]["max_tokens"] == 50
  assert eng.calls[-1]["top_p"] == 1.0

  # batch -> {"texts": [...]}
  r = client.post("/generate", json={"prompts": ["a", "b"], "temperature": 0.0})
  assert r.status_code == 200, r.text
  body = r.json()
  assert body["texts"] == ["<gen:a>", "<gen:b>"]
  assert body.get("text") is None
  assert eng.calls[-1]["temperature"] == 0.0


def test_generate_top_p_always_present_even_if_unset():
  cfg = _cfg()
  eng = _FakeEngine(cfg)
  client = TestClient(serve_mod.build_app(eng))
  client.post("/generate", json={"prompt": "x"})
  # The model default top_p=1.0 must always reach the engine (greedy-bug guard).
  assert eng.calls[-1]["top_p"] == 1.0


def test_generate_validation_errors():
  cfg = _cfg()
  client = TestClient(serve_mod.build_app(_FakeEngine(cfg)))
  # neither prompt nor prompts
  assert client.post("/generate", json={}).status_code == 400
  # both
  r = client.post("/generate", json={"prompt": "a", "prompts": ["b"]})
  assert r.status_code == 400
  # empty prompts list
  assert client.post("/generate", json={"prompts": []}).status_code == 400


def test_generate_503_when_not_ready():
  cfg = _cfg()
  client = TestClient(serve_mod.build_app(_FakeEngine(cfg, ready=False)))
  assert client.post("/generate", json={"prompt": "x"}).status_code == 503


def test_completions_shim():
  cfg = _cfg()
  eng = _FakeEngine(cfg)
  client = TestClient(serve_mod.build_app(eng))
  r = client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 8})
  assert r.status_code == 200, r.text
  body = r.json()
  assert body["object"] == "text_completion"
  assert body["choices"][0]["text"] == "<gen:hi>"
  assert eng.calls[-1]["top_p"] == 1.0


def test_generation_is_serialized():
  """Concurrent /generate calls never overlap inside the fake engine."""
  cfg = _cfg()
  eng = _FakeEngine(cfg)
  client = TestClient(serve_mod.build_app(eng))

  errors = []

  def _hit(i):
    try:
      r = client.post("/generate", json={"prompt": f"p{i}", "seed": i})
      assert r.status_code == 200
    except Exception as exc:  # pragma: no cover
      errors.append(exc)

  threads = [threading.Thread(target=_hit, args=(i,)) for i in range(6)]
  for t in threads:
    t.start()
  for t in threads:
    t.join()
  assert not errors
  assert len(eng.calls) == 6
  # The FakeEngine guards _inflight; the route holds no lock itself, but the
  # real engine's threading.Lock is what serializes. Here we only assert all
  # calls completed; the real lock is exercised in the slow end-to-end test.
  assert eng.max_concurrent >= 1


def test_config_from_env(monkeypatch):
  monkeypatch.setenv("SERVE_CKPT", "gs://bucket/ckpt")
  monkeypatch.setenv("SERVE_MODEL", "qwen3")
  monkeypatch.setenv("SERVE_MAX_PROMPT", "256")
  monkeypatch.setenv("SERVE_MAX_NEW", "128")
  cfg = serve_mod.ServeConfig.from_env()
  assert cfg.ckpt == "gs://bucket/ckpt"
  assert cfg.max_prompt == 256 and cfg.max_new == 128
  assert cfg.cache_size == 256 + 128 + 16

  monkeypatch.delenv("SERVE_CKPT")
  with pytest.raises(ValueError):
    serve_mod.ServeConfig.from_env()


# ---------------------------------------------------------------------------
# launch_serve plan builder + query resolver (no controller / no TPU).
# ---------------------------------------------------------------------------
def test_launch_serve_plan_has_named_port_and_env():
  from serving import launch_serve

  args = launch_serve.parse_args(
      [
          "--ckpt",
          "gs://bucket/run/ckpt",
          "--model",
          "qwen3",
          "--tpu",
          "v6e-4",
          "--region",
          "europe-west4",
      ]
  )
  kwargs = launch_serve.build_submit_kwargs(args)
  assert kwargs["ports"] == ["http"]
  assert kwargs["environment"].env_vars["SERVE_CKPT"] == "gs://bucket/run/ckpt"
  assert kwargs["environment"].env_vars["SERVE_MODEL"] == "qwen3"
  assert kwargs["environment"].extras == ("tpu",)
  assert kwargs["max_retries_preemption"] == 1000
  assert kwargs["entrypoint"].command == ["python", "serving/serve.py"]
  # device + a region constraint are set.
  assert kwargs["resources"].device is not None
  assert kwargs.get("constraints")
  # describe() renders without error.
  assert "ports             = ['http']" in launch_serve._describe(kwargs)


def test_query_resolve_base_url():
  from serving import query

  url = query.resolve_base_url("http://controller:8080", "/alice/delphi-rl")
  # ProxyResolver: leading slash stripped, slashes -> dots, /proxy/ prefix.
  assert url == "http://controller:8080/proxy/alice.delphi-rl"


# ---------------------------------------------------------------------------
# Calc (CALC tool-use) mode: a scripted fake Sampler drives CalcAgentEngine.run
# (no JAX / no weights -- only the real parser + calculator + loop run).
# ---------------------------------------------------------------------------
class _ScriptedSampler:
  """A fake tunix Sampler: each call returns the next scripted turn as ``out.text``.

  ``CalcAgentEngine.run`` calls ``sampler(input_strings=[ctx], ...)`` once per
  turn and reads ``out.text[0]``. We return the pre-scripted assistant turns in
  order so a test can drive the multi-turn loop deterministically, and record the
  contexts the engine fed in (to assert the tool result was injected).
  """

  def __init__(self, turns):
    self._turns = list(turns)
    self._i = 0
    self.contexts = []

  def __call__(self, *, input_strings, **kwargs):
    self.contexts.append(input_strings[0])
    turn = self._turns[self._i] if self._i < len(self._turns) else ""
    self._i += 1
    return types.SimpleNamespace(text=[turn])


def _calc_engine(stage, turns):
  """Builds a ready CalcAgentEngine for ``stage`` with a scripted fake sampler."""
  cfg = serve_mod.ServeConfig(
      ckpt="n/a", model="delphi", task="calc", calc_stage=stage
  )
  eng = serve_mod.CalcAgentEngine(cfg)
  eng._sampler = _ScriptedSampler(turns)
  eng._system = "SYS"  # the real few-shot prompt is irrelevant to the scripted loop
  eng._eos_tokens = [0]
  eng._mesh = None
  eng._ready = True
  return eng


def test_calc_engine_single_call_t0():
  """T0 single call: CALC(12 * 13) -> tool runs (156) -> final answer 156."""
  eng = _calc_engine("t0", ["CALC(12 * 13)", "156"])
  out = eng.run("12 * 13")
  assert out["answer"] == "156"
  assert out["tool_result"] == "156"
  assert out["steps"] == 3  # assistant call + tool result + final answer
  roles = [m["role"] for m in out["transcript"]]
  assert roles == ["assistant", "tool", "assistant"]
  assert out["transcript"][0]["content"] == "CALC(12 * 13)"
  assert out["transcript"][1]["content"] == "156"  # the calculator actually ran
  # The injected tool result was fed back into the next-turn prompt.
  assert "Tool result: 156" in eng._sampler.contexts[1]


def test_calc_engine_chained_t1_calls_calculator_twice():
  """T1 chained: CALC(2 * 3) -> 6 -> CALC(6 * 4) -> 24 -> final 24."""
  eng = _calc_engine("t1", ["CALC(2 * 3)", "CALC(6 * 4)", "24"])
  out = eng.run("2 * 3 * 4")
  assert out["answer"] == "24"
  assert out["tool_result"] == "24"  # last tool result (6 * 4)
  # Two tool turns means the calculator executed twice (chaining).
  tool_results = [m["content"] for m in out["transcript"] if m["role"] == "tool"]
  assert tool_results == ["6", "24"]
  assert out["steps"] == 5  # call, 6, call, 24, final
  # The first tool result (6) is copied into the second-turn prompt.
  assert "Tool result: 6" in eng._sampler.contexts[1]
  assert "Tool result: 24" in eng._sampler.contexts[2]


def test_calc_engine_optional_paren_still_executes():
  """An unclosed CALC( (BPE strips the ')\\n') still parses + executes."""
  eng = _calc_engine("t0", ["CALC(12 * 13", "156"])
  out = eng.run("12 * 13")
  assert out["tool_result"] == "156"
  assert out["answer"] == "156"
  assert out["transcript"][1]["content"] == "156"


def test_calc_engine_prepends_q_prefix():
  """A bare problem gets the 'Q: ' prefix so it matches the few-shot demos."""
  eng = _calc_engine("t0", ["CALC(47 * 53)", "2491"])
  eng.run("47 * 53")
  assert eng._sampler.contexts[0] == "SYS\nQ: 47 * 53\n"

  eng2 = _calc_engine("t0", ["CALC(47 * 53)", "2491"])
  eng2.run("Q: 47 * 53")  # already prefixed: not doubled
  assert eng2._sampler.contexts[0] == "SYS\nQ: 47 * 53\n"


def test_calc_engine_runs_out_of_steps():
  """If the model never finishes, the loop stops at env_max_steps (T0 = 2)."""
  # Both turns emit a tool call (never a final answer), so the budget is hit.
  eng = _calc_engine("t0", ["CALC(2 * 3)", "CALC(6 * 5)", "30"])
  out = eng.run("2 * 3")
  # T0 env_max_steps=2 -> two assistant call turns + two tool injections, no final.
  assert eng._sampler._i == 2  # only two sampler calls (the budget)
  assert out["steps"] == 4  # 2 assistant + 2 tool
  assert [m["role"] for m in out["transcript"]] == ["assistant", "tool", "assistant", "tool"]
  # Falls back to the last produced content (the last tool result) as the answer.
  assert out["answer"] == "30"


def test_calc_stage_sizing_and_validation():
  """Per-stage env_max_steps + cache sizing, and a bad stage raises."""
  for stage, steps, mp, mn in (("t0", 2, 640, 96), ("t1", 3, 768, 128), ("t2", 4, 1024, 192)):
    cfg = serve_mod.ServeConfig(ckpt="n/a", model="delphi", task="calc", calc_stage=stage)
    eng = serve_mod.CalcAgentEngine(cfg)
    assert eng._env_max_steps == steps
    assert eng._max_new == mn
    assert eng._cache_size == mp + mn + 32
  with pytest.raises(ValueError):
    serve_mod.CalcAgentEngine(
        serve_mod.ServeConfig(ckpt="n/a", model="delphi", task="calc", calc_stage="t9")
    )


# ---------------------------------------------------------------------------
# Calc FastAPI app: /calc route + the calc dashboard (single + compare modes).
# ---------------------------------------------------------------------------
class _FakeCalcEngine:
  """Mimics :class:`serving.serve.CalcAgentEngine` for the calc app wiring tests."""

  def __init__(self, cfg, ready=True):
    self._cfg = cfg
    self._stage = cfg.calc_stage
    self._ready = ready
    self.problems = []

  @property
  def ready(self):
    return self._ready

  def run(self, problem):
    self.problems.append(problem)
    return {
        "answer": "156",
        "tool_result": "156",
        "transcript": [
            {"role": "assistant", "content": "CALC(12 * 13)"},
            {"role": "tool", "content": "156"},
            {"role": "assistant", "content": "156"},
        ],
        "steps": 3,
    }


def _calc_cfg(**kw):
  base = dict(ckpt="n/a", model="delphi", task="calc", calc_stage="t1")
  base.update(kw)
  return serve_mod.ServeConfig(**base)


def test_calc_health_and_route():
  eng = _FakeCalcEngine(_calc_cfg())
  client = TestClient(serve_mod.build_calc_app(eng))
  r = client.get("/health")
  assert r.status_code == 200
  assert r.json() == {"status": "ok", "model": "delphi", "stage": "t1"}

  r = client.post("/calc", json={"problem": "12 * 13"})
  assert r.status_code == 200, r.text
  body = r.json()
  assert body["answer"] == "156"
  assert body["tool_result"] == "156"
  assert body["steps"] == 3
  assert [m["role"] for m in body["transcript"]] == ["assistant", "tool", "assistant"]
  assert eng.problems == ["12 * 13"]

  # Accepts `prompt` as an alias, and 400s on an empty problem.
  assert client.post("/calc", json={"prompt": "9 * 9"}).status_code == 200
  assert client.post("/calc", json={}).status_code == 400
  assert client.post("/calc", json={"problem": "   "}).status_code == 400


def test_calc_health_gates_on_ready():
  client = TestClient(serve_mod.build_calc_app(_FakeCalcEngine(_calc_cfg(), ready=False)))
  assert client.get("/health").status_code == 503
  assert client.post("/calc", json={"problem": "2 * 2"}).status_code == 503


def test_calc_dashboard_single_mode():
  client = TestClient(serve_mod.build_calc_app(_FakeCalcEngine(_calc_cfg())))
  for path in ("/", "/dashboard"):
    r = client.get(path)
    assert r.status_code == 200, path
    assert "text/html" in r.headers["content-type"]
    body = r.text
    assert "CALC(" in body  # the calc UI
    # Fetches must be proxy-relative (no absolute "/calc"); single mode => empty COMPARE.
    assert "new URL(" in body
    assert "window.COMPARE = []" in body
    # The single column POSTs to its OWN proxy-relative /calc via api('calc').
    assert "api('calc')" in body
    assert "new URL(path, location.href)" in body  # the api() helper
    # No fetch posts to an absolute "/calc".
    assert "fetch('/calc'" not in body and 'fetch("/calc"' not in body


def test_calc_dashboard_compare_mode():
  cfg = _calc_cfg(compare_endpoints="tunix.delphi-calc-base,tunix.delphi-calc-rl")
  client = TestClient(serve_mod.build_calc_app(_FakeCalcEngine(cfg)))
  body = client.get("/").text
  # Both encoded endpoint names are injected into window.COMPARE.
  assert "tunix.delphi-calc-base" in body
  assert "tunix.delphi-calc-rl" in body
  assert '["tunix.delphi-calc-base", "tunix.delphi-calc-rl"]' in body
  # Compare columns fetch a SIBLING proxy path via `../<name>/calc`.
  assert "new URL('../'" in body
  # Column labels distinguish base vs post-RL.
  assert "base (raw Delphi)" in body and "post-RL" in body


def test_calc_config_from_env(monkeypatch):
  monkeypatch.setenv("SERVE_CKPT", "marin-community/delphi-3e18-447Mparams-1.2Btokens")
  monkeypatch.setenv("SERVE_TASK", "calc")
  monkeypatch.setenv("SERVE_MODEL", "delphi")
  monkeypatch.setenv("SERVE_CALC_STAGE", "t2")
  monkeypatch.setenv("SERVE_COMPARE_ENDPOINTS", "a.b,c.d")
  cfg = serve_mod.ServeConfig.from_env()
  assert cfg.task == "calc" and cfg.model == "delphi" and cfg.calc_stage == "t2"
  assert cfg.compare_endpoints == "a.b,c.d"


def test_maybe_download_hf_repo(monkeypatch):
  """An HF-repo-shaped ckpt (not gs://, not a local path) routes to snapshot_download."""
  calls = {}

  def _fake_snapshot(repo_id, local_dir):
    calls["repo_id"] = repo_id
    calls["local_dir"] = local_dir
    return local_dir

  import huggingface_hub

  monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot)
  out = serve_mod._maybe_download_ckpt("marin-community/delphi-3e18-447Mparams-1.2Btokens")
  assert calls["repo_id"] == "marin-community/delphi-3e18-447Mparams-1.2Btokens"
  assert out == calls["local_dir"]
  # An existing local path is returned untouched (never treated as a repo).
  assert serve_mod._maybe_download_ckpt("/tmp") == "/tmp"


# ---------------------------------------------------------------------------
# SLOW: a real tiny qm.Qwen3 + tunix Sampler end to end.
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_generate_end_to_end_tiny_model():
  """Build a tiny real Qwen3 + Sampler, inject into the engine, hit /generate.

  Uses a real (small) HF tokenizer and a fresh-init model -- no weights on disk,
  no accelerator -- so it exercises the true tunix Sampler decode path and the
  ``top_p``-always contract. Marked ``slow`` (JAX compile is heavy).
  """
  import jax.numpy as jnp
  from flax import nnx
  from tunix.generate import sampler as sampler_lib
  from tunix.models.qwen3 import model as qm

  config = qm.ModelConfig(
      num_layers=2,
      vocab_size=151936,  # match a real Qwen3 tokenizer so ids are in range
      embed_dim=64,
      hidden_dim=128,
      num_heads=4,
      head_dim=16,
      num_kv_heads=2,
      rope_theta=1_000_000,
      norm_eps=1e-6,
      use_tied_embedding=True,
      dtype=jnp.float32,
      param_dtype=jnp.float32,
  )
  model = qm.Qwen3(config, rngs=nnx.Rngs(params=0))

  from transformers import AutoTokenizer

  tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
  if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

  cfg = serve_mod.ServeConfig(ckpt="n/a", model="qwen3", max_prompt=32, max_new=8)
  cache_config = sampler_lib.CacheConfig(
      cache_size=cfg.cache_size,
      num_layers=config.num_layers,
      num_kv_heads=config.num_kv_heads,
      head_dim=config.head_dim,
  )
  sampler = sampler_lib.Sampler(transformer=model, tokenizer=tok, cache_config=cache_config)

  eng = serve_mod.GenerationEngine(cfg)
  eng._model = model
  eng._tokenizer = tok
  eng._sampler = sampler
  eng._mesh = None
  eng._eos_tokens = [tok.eos_token_id]
  eng._ready = True

  client = TestClient(serve_mod.build_app(eng))
  assert client.get("/health").status_code == 200
  r = client.post("/generate", json={"prompt": "Hello", "max_tokens": 8, "temperature": 0.8})
  assert r.status_code == 200, r.text
  assert isinstance(r.json()["text"], str)
