"""Delphi arithmetic GRPO training entry (Milestone M4).

Builds the tunix ``RLCluster`` + ``GRPOLearner`` exactly per the proven non-
agentic wiring (``.agents/logs/tunix-iris/grpo-wiring.md`` + M2 ``toy_cats.py``),
swapping the toy model for the REAL Delphi 447M Qwen3 (loaded via
``delphi_qwen3.load_delphi``, which applies the worker-shippable rope monkeypatch)
and the toy task for the single-turn arithmetic environment (``arithmetic.py``).

Key choices (justified):
  * **Actor storage fp32** (``load_delphi(dtype=fp32)``): bf16 storage rounds the
    small Adam updates (~1e-6 at lr 1e-6) below bf16 ULP (~7.8e-5), so the policy
    never moves. Compute may still be bf16 via the model config dtype, but for a
    447M actor we keep fp32 storage + fp32 compute for the first run.
  * **No reference model** (``reference=None`` + ``beta=0``): saves a full model
    copy in HBM (~0.9 GB bf16) and a KL term for the first run, per the wiring
    cheatsheet recommendation. KL/reference can be re-enabled later by passing
    ``beta>0`` and a loaded reference.
  * **Rollout** ``vanilla`` (the only backend that runs Delphi's non-vLLM-
    registered arch in this venv), ``eos_tokens=[128001]``, pad=eos=128001.
  * **Mesh** colocated ``(ndev, 1)`` fsdp/tp built from ``jax.device_count()``
    (1 on CPU, N on a single-host TPU), shared across ACTOR/REFERENCE/ROLLOUT.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo import grpo_learner as grpo_lib
from tunix.rl.rollout import base_rollout

from problems.arithmetic import (
    answer_reward,
    build_arithmetic_dataset,
    format_reward,
    metric_fn,
    proximity_reward,
)
from models.delphi_qwen3 import DELPHI_EOS_ID, load_delphi, load_tokenizer


@dataclasses.dataclass
class DelphiTrainResult:
  """Result of a Delphi arithmetic GRPO run.

  Attributes:
    reward_history: per-global-step mean reward (answer + format, summed).
    solve_ratio_history: per-global-step ``arithmetic/solve_ratio`` from the
      metric fn (the curriculum-gating signal).
    steps_ran: number of global steps actually executed.
  """

  reward_history: list[float]
  solve_ratio_history: list[float]
  steps_ran: int


class _MetricsCapture:
  """Captures per-step train reward and solve_ratio from the metrics buffer.

  ``RLCluster.with_external_metrics_logger`` installs a callable invoked with a
  ``MetricsBuffer`` whose ``.metrics`` maps name -> ``([values], aggfn)`` and
  carries ``.mode``. We read the train-mode ``rewards/score/mean`` (the summed
  reward) and ``arithmetic/solve_ratio`` (from :func:`arithmetic.metric_fn`).
  """

  def __init__(self):
    self.reward_history: list[float] = []
    self.solve_ratio_history: list[float] = []

  def __call__(self, metrics_buffer, *args, **kwargs):
    """Records mean reward + solve_ratio from one flushed train buffer."""
    del args, kwargs
    if "train" not in str(getattr(metrics_buffer, "mode", "")):
      return
    metrics = getattr(metrics_buffer, "metrics", {})
    reward_entry = metrics.get("rewards/score/mean")
    if reward_entry is not None and reward_entry[0]:
      self.reward_history.append(
          float(np.mean(np.asarray(reward_entry[0], dtype=np.float32)))
      )
    solve_entry = metrics.get("arithmetic/solve_ratio")
    if solve_entry is not None and solve_entry[0]:
      self.solve_ratio_history.append(
          float(np.mean(np.asarray(solve_entry[0], dtype=np.float32)))
      )


def _build_mesh() -> jax.sharding.Mesh:
  """Builds a colocated ``(ndev, 1)`` fsdp/tp mesh over all local devices."""
  return jax.sharding.Mesh(
      np.asarray(jax.devices()).reshape(jax.device_count(), 1),
      axis_names=("fsdp", "tp"),
  )


def train_delphi_arithmetic(
    *,
    model_dir: str,
    stage: int = 0,
    steps: int = 50,
    num_generations: int = 8,
    batch_size: int = 4,
    learning_rate: float = 1e-6,
    temperature: float = 0.9,
    max_prompt_length: int = 128,
    max_tokens_to_generate: int = 64,
    beta: float = 0.0,
    seed: int = 0,
    eval_every_n_steps: int = 10**9,
    reward_mode: str = "exact",
    mesh: jax.sharding.Mesh | None = None,
) -> DelphiTrainResult:
  """Runs Delphi arithmetic GRPO and returns the reward / solve-rate trajectory.

  Builds the full ``RLCluster`` + ``GRPOLearner`` pipeline with the REAL Delphi
  model: load (fp32 actor) -> vanilla rollout -> reward -> group-relative
  advantage -> AdamW update.

  Args:
    model_dir: directory containing Delphi's ``model.safetensors`` + tokenizer.
    stage: arithmetic curriculum stage (0 single-digit add; 1 add/sub/mul 2-dig).
    steps: number of GRPO global steps.
    num_generations: GRPO group size (responses per prompt; advantage is
      normalized within the group, so >1 is required).
    batch_size: prompts per global step (on-policy batch =
      ``batch_size * num_generations``).
    learning_rate: AdamW learning rate (~1e-6 for a real pretrained model).
    temperature: rollout sampling temperature.
    max_prompt_length: max prompt length (the few-shot prompt is short).
    max_tokens_to_generate: completion length budget.
    beta: KL-to-reference weight. ``0.0`` drops the reference model entirely.
    seed: PRNG seed for the problem set.
    eval_every_n_steps: eval stride (defaults to effectively never).
    mesh: optional device mesh; defaults to a colocated ``(ndev, 1)`` mesh.

  Returns:
    A :class:`DelphiTrainResult` with the reward + solve_ratio histories.
  """
  tokenizer = load_tokenizer(model_dir)

  # Actor storage MUST be fp32 (small Adam updates round to zero under bf16).
  actor = load_delphi(model_dir, dtype=jnp.float32)
  reference = None if beta == 0.0 else load_delphi(model_dir, dtype=jnp.bfloat16)

  if mesh is None:
    mesh = _build_mesh()

  kv_cache_size = max_prompt_length + max_tokens_to_generate + 8
  rollout_config = base_rollout.RolloutConfig(
      max_prompt_length=max_prompt_length,
      kv_cache_size=kv_cache_size,
      max_tokens_to_generate=max_tokens_to_generate,
      temperature=temperature,
      top_p=1.0,
      top_k=None,
      return_logprobs=False,
      eos_tokens=[DELPHI_EOS_ID],  # pad=eos=128001 (set on the tokenizer)
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

  grpo_config = grpo_lib.GRPOConfig(
      num_generations=num_generations,
      num_iterations=1,
      beta=beta,
      epsilon=0.2,
  )

  rl_cluster = rl_cluster_lib.RLCluster(
      actor=actor,
      reference=reference,
      tokenizer=tokenizer,
      cluster_config=cluster_config,
  )

  capture = _MetricsCapture()
  rl_cluster.with_external_metrics_logger(capture)

  learner = grpo_lib.GRPOLearner(
      rl_cluster=rl_cluster,
      algo_config=grpo_config,
      reward_fns=(
          [proximity_reward, format_reward]
          if reward_mode == "shaped"
          else [answer_reward, format_reward]
      ),
      metric_fns=[metric_fn],
  )

  train_ds = build_arithmetic_dataset(
      stage=stage,
      n=steps * batch_size + batch_size,
      seed=seed,
      batch_size=batch_size,
  )

  learner.train(train_ds, eval_ds=None)

  return DelphiTrainResult(
      reward_history=capture.reward_history,
      solve_ratio_history=capture.solve_ratio_history,
      steps_ran=len(capture.reward_history),
  )
