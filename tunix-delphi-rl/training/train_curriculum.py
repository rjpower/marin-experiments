# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Delphi curriculum coding driver (issue #8): SFT warm-up -> curriculum Dr.GRPO.

The test-case-graded, curriculum-scheduled redesign (``CURRICULUM_DESIGN.md``).
A light SFT warm-up teaches the ``def solve`` + CoT format on the easy levels;
Dr.GRPO then trains on a curriculum-scheduled stream of test-case problems
(:mod:`curriculum_env`) whose dense reward is the fraction of tests passed -- so a
group has continuous reward variance and RL has a gradient (the fix for the
no-exploration-gap result in #8). Eval is per-level pass@1/pass@k on HELD-OUT
instances, before and after RL, vs an SFT-only control at matched compute.

Reuses the agentic Dr.GRPO wiring (fp32 actor, clipped_adamw, beta=0, vanilla
rollout, per-call rollout seed, per-turn END stop, numpy-leaf normalization) from
:mod:`train_multiturn`.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.agentic.agentic_grpo_learner import GRPOConfig
from tunix.rl.rollout import base_rollout

from training.agentic_common import DelphiRawTextChatParser, clipped_adamw
from training.agentic_sft import run_sft_warmup
from environments.agentic_tools import install_per_call_rollout_seed
from environments.coding_agent_env import PassKResult, RunCodeAgent, program_terminal_eos_tokens
from environments.curriculum import CurriculumConfig
from environments.curriculum_env import (
    CODE_SOLVE_SYSTEM_PROMPT,
    TestCaseEnvironment,
    build_curriculum_dataset,
    evaluate_problems_passk,
    load_eval_suite,
    solve_metric_fn,
    solve_segments,
)
from models.delphi_qwen3 import DELPHI_EOS_ID, load_delphi, load_tokenizer
from training.train_multiturn import _NormalizingGRPOLearner, _build_mesh


@dataclasses.dataclass
class CurriculumTrainResult:
  """Result of a Delphi curriculum coding SFT+Dr.GRPO run."""

  best_solve_history: list[float]
  first_solve_history: list[float]
  reward_history: list[float]
  level_history: list[float]
  steps_ran: int
  passk_after_sft: PassKResult | None
  passk_after_rl: PassKResult | None


class _CurriculumMetricsCapture:
  """Captures per-step solve/frac/level metrics from the training metrics buffer."""

  def __init__(self):
    self.best_solve_history: list[float] = []
    self.first_solve_history: list[float] = []
    self.reward_history: list[float] = []
    self.level_history: list[float] = []

  @staticmethod
  def _mean_of(entry):
    if entry is None or not entry[0]:
      return None
    return float(np.mean(np.asarray(entry[0], dtype=np.float32)))

  def __call__(self, metrics_buffer, *args, **kwargs):
    del args, kwargs
    if "train" not in str(getattr(metrics_buffer, "mode", "")).lower():
      return
    metrics = getattr(metrics_buffer, "metrics", {})
    for name, history in (
        ("coding/best_solve", self.best_solve_history),
        ("coding/first_solve", self.first_solve_history),
        ("coding/reward_mean", self.reward_history),
        ("coding/mean_level", self.level_history),
    ):
      value = self._mean_of(metrics.get(name))
      if value is not None:
        history.append(value)


def _passk(actor, tokenizer, problems, *, k, temperature, max_new_tokens, max_prompt_length, mesh, label):
  result = evaluate_problems_passk(
      actor,
      tokenizer,
      problems,
      k=k,
      temperature=temperature,
      max_new_tokens=max_new_tokens,
      max_prompt_length=max_prompt_length,
      mesh=mesh,
  )
  print(f"[curric] PASS@K {label}: {result.summary()}", flush=True)
  return result


def train_curriculum(
    *,
    model_dir: str,
    train_levels: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
    eval_levels: tuple[int, ...] | None = None,
    eval_n_per_level: int = 12,
    rounds: int = 3,
    steps: int = 200,
    steps_per_level: int = 30,
    promote_threshold: float = 0.0,
    num_generations: int = 16,
    batch_size: int = 8,
    learning_rate: float = 1e-5,
    temperature: float = 1.0,
    max_prompt_length: int = 1024,
    max_response_length: int = 768,
    beta: float = 0.0,
    seed: int = 0,
    sft_steps: int = 200,
    sft_levels: tuple[int, ...] = (1, 2),
    sft_batch_size: int = 16,
    sft_learning_rate: float = 1e-4,
    sft_max_seq_len: int = 640,
    passk: int = 16,
    passk_temperature: float = 1.0,
    eval_max_new_tokens: int = 256,
    do_eval: bool = True,
    mesh: jax.sharding.Mesh | None = None,
) -> CurriculumTrainResult:
  """SFT warm-up (easy levels) -> curriculum Dr.GRPO; per-level held-out pass@k."""
  tokenizer = load_tokenizer(model_dir)
  if mesh is None:
    mesh = _build_mesh()
  if eval_levels is None:
    eval_levels = train_levels
  eval_problems = load_eval_suite(eval_levels, eval_n_per_level, seed=99)

  actor = load_delphi(model_dir, dtype=jnp.float32, mesh=mesh)
  reference = None if beta == 0.0 else load_delphi(model_dir, dtype=jnp.bfloat16, mesh=mesh)

  # Stage 1: SFT warm-up on the easy levels (teach def solve + CoT format).
  if sft_steps > 0:
    actor = run_sft_warmup(
        actor,
        tokenizer,
        steps=sft_steps,
        batch_size=sft_batch_size,
        learning_rate=sft_learning_rate,
        mesh=mesh,
        segment_fn=lambda rng: solve_segments(rng, sft_levels),
        prompt_prefix=CODE_SOLVE_SYSTEM_PROMPT,
        max_seq_len=sft_max_seq_len,
        seed=seed,
    )

  passk_after_sft = None
  if do_eval and passk > 0:
    passk_after_sft = _passk(
        actor, tokenizer, eval_problems,
        k=passk, temperature=passk_temperature, max_new_tokens=eval_max_new_tokens,
        max_prompt_length=max_prompt_length, mesh=mesh,
        label="after-sft" if sft_steps > 0 else "few-shot",
    )

  if steps <= 0:
    return CurriculumTrainResult(
        best_solve_history=[], first_solve_history=[], reward_history=[],
        level_history=[], steps_ran=0,
        passk_after_sft=passk_after_sft, passk_after_rl=None,
    )

  # Stage 2: curriculum Dr.GRPO on test-case problems.
  eos_tokens = sorted(set([DELPHI_EOS_ID]) | set(program_terminal_eos_tokens(tokenizer)))
  if len(eos_tokens) <= 1:
    raise RuntimeError("No 'END' terminal token found; per-turn stop would never fire.")

  kv_cache_size = max_prompt_length + max_response_length + 8
  rollout_config = base_rollout.RolloutConfig(
      max_prompt_length=max_prompt_length,
      kv_cache_size=kv_cache_size,
      max_tokens_to_generate=max_response_length,
      temperature=temperature,
      top_p=1.0,
      top_k=None,
      return_logprobs=False,
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
          actor_optimizer=clipped_adamw(learning_rate),
          eval_every_n_steps=10**9,
          max_steps=steps,
      ),
      rollout_config=rollout_config,
  )
  grpo_config = GRPOConfig(
      num_generations=num_generations,
      num_iterations=1,
      beta=beta,
      epsilon=0.2,
      advantage_estimator="drgrpo",
      loss_agg_mode="sequence-mean-token-scale",
      degenerate_group_masking=False,
      use_rollout_logps=False,
      system_prompt=CODE_SOLVE_SYSTEM_PROMPT,
      max_response_length=max_response_length,
      max_concurrency=batch_size * num_generations,
  )

  rl_cluster = rl_cluster_lib.RLCluster(
      actor=actor, reference=reference, tokenizer=tokenizer, cluster_config=cluster_config
  )
  install_per_call_rollout_seed(rl_cluster, base_seed=seed)

  capture = _CurriculumMetricsCapture()
  rl_cluster.with_external_metrics_logger(capture)

  learner = _NormalizingGRPOLearner(
      rl_cluster=rl_cluster,
      algo_config=grpo_config,
      reward_fns=None,  # the env's best-across-rounds terminal reward IS the reward
      chat_parser=DelphiRawTextChatParser(generation_suffix="\n"),
      metric_fns=[solve_metric_fn],
      agent_class=RunCodeAgent,
      env_class=TestCaseEnvironment,
      env_kwargs={"tool_map": {}, "max_steps": rounds},
  )

  cur_config = CurriculumConfig(
      num_levels=max(train_levels),
      steps_per_level=steps_per_level,
      promote_threshold=promote_threshold,
  )
  train_ds = build_curriculum_dataset(
      steps=steps, batch_size=batch_size, seed=seed, cur_config=cur_config
  )

  learner.train(train_ds, eval_dataset=None)

  passk_after_rl = None
  if do_eval and passk > 0:
    passk_after_rl = _passk(
        rl_cluster.actor_trainer.model, tokenizer, eval_problems,
        k=passk, temperature=passk_temperature, max_new_tokens=eval_max_new_tokens,
        max_prompt_length=max_prompt_length, mesh=mesh, label="after-rl",
    )

  return CurriculumTrainResult(
      best_solve_history=capture.best_solve_history,
      first_solve_history=capture.first_solve_history,
      reward_history=capture.reward_history,
      level_history=capture.level_history,
      steps_ran=len(capture.reward_history),
      passk_after_sft=passk_after_sft,
      passk_after_rl=passk_after_rl,
  )
