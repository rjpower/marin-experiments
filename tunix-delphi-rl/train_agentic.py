"""Delphi arithmetic GRPO on tunix's AGENTIC learner (milestone M-port).

Ports the proven non-agentic Delphi GRPO harness (``train_delphi.py``) onto
``tunix.rl.agentic.agentic_grpo_learner.GRPOLearner`` -- the single-turn, no-tool
agentic code path -- to prove the agentic plumbing works on the
known-learnable single-digit-add task before any tools are added.

What is reused verbatim from the non-agentic harness:
  * ``delphi_qwen3.load_delphi`` (fp32 actor / bf16 ref, worker-shippable rope
    monkeypatch), ``load_tokenizer`` (pad=eos=128001), ``DELPHI_EOS_ID``.
  * The arithmetic dataset (``prompts`` + ``answer`` columns) and the
    ``answer_reward`` / ``format_reward`` / ``metric_fn`` callables.
  * The colocated ``(ndev, 1)`` fsdp/tp mesh and the ``vanilla`` rollout.

What changes for the agentic path (verified against tunix 0.1.7 source):
  * The learner is the agentic ``GRPOLearner`` with the default single-turn
    ``ModelAgent`` + ``TaskEnvironment``. The agent reads the dataset's
    ``prompts`` column as the user turn (``ConversationAgentBase
    ._observation_to_messages`` keys on ``"prompts"``).
  * Single-turn reward: the env-level ``reward_fn`` is left unset (so the
    per-trajectory ``trajectory_reward`` is 0), and our reward functions are
    passed as ``reward_fns=`` to the learner. The ``agentic-sequence-level``
    reward manager evaluates them post-rollout via
    ``reward_fn(prompts=..., completions=..., answer=..., **task)`` -- the SAME
    signature the non-agentic path used -- and ADDS them to the (zero)
    trajectory reward. The ``answer`` column reaches them through the
    trajectory's ``original_input``.
  * A raw-text ``chat_parser`` (:class:`agentic_common.DelphiRawTextChatParser`)
    is required because Delphi has no chat template (the stock Qwen parser emits
    out-of-vocab control tokens).
  * ``GRPOConfig.max_response_length`` MUST equal the rollout's
    ``max_tokens_to_generate`` and (with ``use_rollout_logps=True``) the rollout
    MUST set ``return_logprobs=True`` -- both enforced by
    ``AgenticRLLearner._validate_rollout_config``. We default
    ``use_rollout_logps=False`` (matching the gsm8k demo); set it True to surface
    the sampler-vs-trainer ``logp_diff`` tokenization-consistency canary.

Per-step ``mean_reward`` + ``solve_ratio`` history is captured from the cluster
metrics buffer via :class:`_AgenticMetricsCapture` (mirroring
``train_delphi._MetricsCapture``), reading the ``agentic-sequence-level`` reward
manager's per-reward-fn metrics and :func:`arithmetic.metric_fn`'s solve ratio.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.agentic.agentic_grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.agentic.agents.model_agent import ModelAgent
from tunix.rl.agentic.environments.task_environment import TaskEnvironment
from tunix.rl.rollout import base_rollout

from agentic_common import DelphiRawTextChatParser
from arithmetic import (
    answer_reward,
    build_arithmetic_dataset,
    format_reward,
    metric_fn,
)
from delphi_qwen3 import DELPHI_EOS_ID, load_delphi, load_tokenizer


@dataclasses.dataclass
class AgenticTrainResult:
  """Result of a Delphi agentic-GRPO M-port run.

  Attributes:
    reward_history: per-global-step mean summed reward (answer + format).
    solve_ratio_history: per-global-step ``arithmetic/solve_ratio`` (exact-match
      rate; the curriculum-gating signal).
    logp_diff_history: per-step mean sampler-vs-trainer per-token logp diff, when
      the engine emits it (``use_rollout_logps=True`` + ``return_logprobs=True``);
      empty otherwise. A value well under ~0.01 nat is the tokenization-
      consistency canary for the agentic path.
    steps_ran: number of global steps with a captured reward.
  """

  reward_history: list[float]
  solve_ratio_history: list[float]
  logp_diff_history: list[float]
  steps_ran: int


class _AgenticMetricsCapture:
  """Captures per-step reward, solve_ratio and logp_diff from the metrics buffer.

  ``RLCluster.with_external_metrics_logger`` installs a callable invoked with a
  ``MetricsBuffer`` whose ``.metrics`` maps name -> ``([values], aggfn)`` and
  carries ``.mode``. The ``agentic-sequence-level`` reward manager logs each
  reward fn's per-prompt rewards under ``rewards/<fn_name>`` (e.g.
  ``rewards/answer_reward``); their sum is the GRPO-scored reward.
  :func:`arithmetic.metric_fn` logs ``arithmetic/solve_ratio``. The agentic
  learner additionally logs ``sampler_trainer/logp_diff_mean`` when rollout
  logprobs are available.
  """

  def __init__(self):
    self.reward_history: list[float] = []
    self.solve_ratio_history: list[float] = []
    self.logp_diff_history: list[float] = []

  @staticmethod
  def _mean_of(entry: Any) -> float | None:
    """Returns the mean of a buffered ``([values], aggfn)`` entry, or None."""
    if entry is None or not entry[0]:
      return None
    return float(np.mean(np.asarray(entry[0], dtype=np.float32)))

  def __call__(self, metrics_buffer, *args, **kwargs) -> None:
    """Records mean reward + solve_ratio + logp_diff from one flushed buffer."""
    del args, kwargs
    if "train" not in str(getattr(metrics_buffer, "mode", "")).lower():
      return
    metrics = getattr(metrics_buffer, "metrics", {})

    # Summed reward = sum over the per-reward-fn means (answer + format).
    reward_terms = [
        self._mean_of(metrics.get(f"rewards/{fn.__name__}"))
        for fn in (answer_reward, format_reward)
    ]
    reward_terms = [r for r in reward_terms if r is not None]
    if reward_terms:
      self.reward_history.append(float(sum(reward_terms)))

    solve = self._mean_of(metrics.get("arithmetic/solve_ratio"))
    if solve is not None:
      self.solve_ratio_history.append(solve)

    logp_diff = self._mean_of(metrics.get("sampler_trainer/logp_diff_mean"))
    if logp_diff is not None:
      self.logp_diff_history.append(logp_diff)


class _NormalizingGRPOLearner(GRPOLearner):
  """Agentic GRPO learner that normalizes per-example dict values to Python str.

  The grain dataset batches its ``prompts`` / ``answer`` columns as numpy string
  arrays; after the engine's size-1 micro-batching each single example carries
  one-element numpy arrays. Left as-is, the agent would inject a numpy array as a
  message ``content`` and the reward fns would see numpy scalars. We collapse
  each leaf to a plain Python ``str`` before the (agent, env) pair is built so
  the chat parser renders raw text and ``answer`` is a clean string. Mirrors the
  ``VTCGRPOLearner`` override in the upstream gsm8k agentic demo.
  """

  def _create_agent_env_pair(self, single_example, group_id: int, pair_index: int):
    return super()._create_agent_env_pair(
        _normalize_example(single_example),
        group_id=group_id,
        pair_index=pair_index,
    )


def _normalize_leaf(value: Any) -> Any:
  """Collapses numpy/bytes leaves to a Python scalar (str for our columns)."""
  if isinstance(value, np.ndarray):
    flat = value.reshape(-1).tolist()
    return _normalize_leaf(flat[0]) if len(flat) == 1 else [
        _normalize_leaf(v) for v in flat
    ]
  if isinstance(value, (bytes, np.bytes_)):
    return value.decode("utf-8") if isinstance(value, bytes) else value.tobytes().decode("utf-8")
  if isinstance(value, np.generic):
    return value.item()
  return value


def _normalize_example(example: Dict[str, Any]) -> Dict[str, Any]:
  """Normalizes every column value of a single example to a Python scalar."""
  return {key: _normalize_leaf(value) for key, value in example.items()}


def _build_mesh() -> jax.sharding.Mesh:
  """Builds a colocated ``(ndev, 1)`` fsdp/tp mesh over all local devices."""
  return jax.sharding.Mesh(
      np.asarray(jax.devices()).reshape(jax.device_count(), 1),
      axis_names=("fsdp", "tp"),
  )


def train_agentic_port(
    *,
    model_dir: str,
    stage: int = 0,
    steps: int = 50,
    num_generations: int = 8,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    temperature: float = 0.9,
    max_prompt_length: int = 128,
    max_tokens_to_generate: int = 64,
    beta: float = 0.0,
    seed: int = 0,
    eval_every_n_steps: int = 10**9,
    use_rollout_logps: bool = False,
    stop_on_newline: bool = True,
    mesh: jax.sharding.Mesh | None = None,
) -> AgenticTrainResult:
  """Runs Delphi single-digit-add GRPO through the AGENTIC learner.

  Builds the full agentic ``RLCluster`` + ``GRPOLearner`` pipeline with the REAL
  Delphi model and the default single-turn ``ModelAgent`` / ``TaskEnvironment``:
  load (fp32 actor) -> agentic rollout -> reward_fns -> group-relative advantage
  -> AdamW update. Returns the per-step reward / solve-rate / logp-diff history.

  Args:
    model_dir: directory containing Delphi's ``model.safetensors`` + tokenizer.
    stage: arithmetic curriculum stage (0 = single-digit add; the M-port task).
    steps: number of GRPO global steps (``max_steps``).
    num_generations: GRPO group size (responses per prompt; must be > 1).
    batch_size: prompts per global step (on-policy batch =
      ``batch_size * num_generations``).
    learning_rate: AdamW learning rate.
    temperature: rollout sampling temperature.
    max_prompt_length: max prompt length (the few-shot prompt is short).
    max_tokens_to_generate: completion length budget; MUST equal
      ``GRPOConfig.max_response_length`` (enforced by the learner).
    beta: KL-to-reference weight. ``0.0`` drops the reference model entirely.
    seed: PRNG seed for the problem set.
    eval_every_n_steps: eval stride (defaults to effectively never).
    use_rollout_logps: when True, the rollout returns logprobs and the learner
      logs the sampler-vs-trainer ``logp_diff`` tokenization canary; requires
      ``return_logprobs=True`` (set automatically here).
    stop_on_newline: when True (default), the newline token is added to
      ``eos_tokens`` so a single-turn completion stops at the few-shot answer
      boundary (``" 8\\n"``). This is REQUIRED for the agentic path: the Delphi
      base LM never emits the real EOS for arithmetic, so without a newline stop
      every completion fills the entire ``max_response_length`` budget and the
      engine's ``_response_token_count >= max_response_length`` context-limit
      check (``trajectory_collect_engine._one_step``) trips and DISCARDS the
      completion *before* it is recorded -> empty completions, all-zero rewards.
      Stopping at the newline keeps completions short (a few tokens), well under
      budget, so they are recorded and scored. The first integer in the
      completion is still the model's answer (matching the non-agentic harness).
    mesh: optional device mesh; defaults to a colocated ``(ndev, 1)`` mesh.

  Returns:
    An :class:`AgenticTrainResult` with the reward / solve_ratio / logp_diff
    histories.
  """
  tokenizer = load_tokenizer(model_dir)

  # Actor storage MUST be fp32 (small Adam updates round to zero under bf16).
  actor = load_delphi(model_dir, dtype=jnp.float32)
  reference = None if beta == 0.0 else load_delphi(model_dir, dtype=jnp.bfloat16)

  if mesh is None:
    mesh = _build_mesh()

  # Newline-stop (see ``stop_on_newline`` docstring): bound single-turn
  # completions to the few-shot answer line so they do not fill the whole
  # response budget and get discarded by the agentic context-limit check. The
  # newline id is read from the tokenizer (Llama-3: 198) rather than hardcoded.
  eos_tokens = [DELPHI_EOS_ID]  # pad=eos=128001 (set on the tokenizer)
  if stop_on_newline:
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    if newline_ids:
      eos_tokens.append(newline_ids[-1])

  kv_cache_size = max_prompt_length + max_tokens_to_generate + 8
  rollout_config = base_rollout.RolloutConfig(
      max_prompt_length=max_prompt_length,
      kv_cache_size=kv_cache_size,
      max_tokens_to_generate=max_tokens_to_generate,
      temperature=temperature,
      top_p=1.0,
      top_k=None,
      # return_logprobs is REQUIRED by _validate_rollout_config when
      # use_rollout_logps=True; harmless (and surfaces the logp_diff canary)
      # otherwise, so we tie it to use_rollout_logps.
      return_logprobs=use_rollout_logps,
      eos_tokens=eos_tokens,
  )

  cluster_config = rl_cluster_lib.ClusterConfig(
      role_to_mesh={
          rl_cluster_lib.Role.ACTOR: mesh,
          rl_cluster_lib.Role.REFERENCE: mesh,
          rl_cluster_lib.Role.ROLLOUT: mesh,
      },
      rollout_engine="vanilla",
      offload_to_cpu=False,
      training_config=rl_cluster_lib.RLTrainingConfig(
          actor_optimizer=optax.adamw(
              learning_rate=learning_rate, b1=0.9, b2=0.99, weight_decay=0.0
          ),
          eval_every_n_steps=eval_every_n_steps,
          max_steps=steps,
      ),
      rollout_config=rollout_config,
  )

  # max_response_length MUST equal rollout max_tokens_to_generate (validated).
  grpo_config = GRPOConfig(
      num_generations=num_generations,
      num_iterations=1,
      beta=beta,
      epsilon=0.2,
      advantage_estimator="grpo",
      degenerate_group_masking=False,
      use_rollout_logps=use_rollout_logps,
      system_prompt="",
      max_response_length=max_tokens_to_generate,
      max_concurrency=batch_size * num_generations,
      loss_agg_mode="sequence-mean-token-mean",
  )

  rl_cluster = rl_cluster_lib.RLCluster(
      actor=actor,
      reference=reference,
      tokenizer=tokenizer,
      cluster_config=cluster_config,
  )

  capture = _AgenticMetricsCapture()
  rl_cluster.with_external_metrics_logger(capture)

  learner = _NormalizingGRPOLearner(
      rl_cluster=rl_cluster,
      algo_config=grpo_config,
      reward_fns=[answer_reward, format_reward],
      chat_parser=DelphiRawTextChatParser(),
      metric_fns=[metric_fn],
      agent_class=ModelAgent,
      env_class=TaskEnvironment,
  )

  train_ds = build_arithmetic_dataset(
      stage=stage,
      n=steps * batch_size + batch_size,
      seed=seed,
      batch_size=batch_size,
  )

  learner.train(train_ds, eval_dataset=None)

  return AgenticTrainResult(
      reward_history=capture.reward_history,
      solve_ratio_history=capture.solve_ratio_history,
      logp_diff_history=capture.logp_diff_history,
      steps_ran=len(capture.reward_history),
  )
