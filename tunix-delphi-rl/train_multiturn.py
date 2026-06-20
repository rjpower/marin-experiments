# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Delphi multi-turn agentic-CODING driver (issue #8): SFT warm-up -> Dr.GRPO RL.

Escalates :mod:`train_coding` (single-turn) to a MULTI-TURN write -> run -> revise
loop on tunix's *agentic* GRPO stack, with the Dr.GRPO knobs set directly on the
agentic ``GRPOConfig`` (``advantage_estimator="drgrpo"`` -- group-mean-centered,
no std division; ``loss_agg_mode="sequence-mean-token-scale"`` -- constant
normalizer, no per-response length bias). The pipeline is the same three stages
codified in ``AGENTS.md``:

  1. **SFT for the multi-turn format** -- :func:`agentic_sft.run_sft_warmup` on
     :func:`coding_agent_env.code_agent_segments` (Task -> [buggy -> Tool result ->]
     solution), with ``prompt_prefix=CODE_AGENT_SYSTEM_PROMPT`` so the SFT context
     == the RL rollout prompt (invariant D). A MINORITY of transcripts show a
     read-output-and-fix, making that behavior in-distribution but rare.
  2. **(the "tool" is the interpreter)** -- invoked by the env each round, not
     called in-band; the program *is* the action.
  3. **Dr.GRPO RL** on the multi-turn rollout (:class:`coding_agent_env.CodeRunEnvironment`
     + :class:`coding_agent_env.RunCodeAgent`). The reward is the env's
     best-across-rounds dense grade (``reward_fns=None``; the env terminal reward
     IS the trajectory reward), so RL is rewarded for *reaching* a correct program
     within the round budget -- amplifying the fix behavior SFT seeded.

Eval is the greedy multi-turn loop (:func:`coding_agent_env.evaluate_tasks_multiturn`)
on the fixed ladder -- once after SFT and once after RL -- reporting first-attempt
vs best-across-rounds solve per tier, so a single run shows the multi-turn lift
and how RL grows it.

Reused invariants (see ``AGENTS.md``): fp32 actor; ``clipped_adamw`` (crash
prevention); ``beta=0``; ``rollout_engine="vanilla"``; per-call rollout seed
(:func:`agentic_tools.install_per_call_rollout_seed`, else a fixed seed makes all
group samples identical -> zero advantage); per-turn ``END`` stop
(:func:`coding_agent_env.program_terminal_eos_tokens`) since programs are
multi-line; numpy-leaf normalization of each example before the (agent, env) pair.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.agentic.agentic_grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.rollout import base_rollout

from agentic_common import DelphiRawTextChatParser, clipped_adamw
from agentic_sft import run_sft_warmup
from agentic_tools import install_per_call_rollout_seed
from coding_agent_env import (
    CODE_AGENT_SYSTEM_PROMPT,
    MultiTurnEvalResult,
    CodeRunEnvironment,
    RunCodeAgent,
    build_code_agent_dataset,
    code_agent_metric_fn,
    code_agent_segments,
    evaluate_tasks_multiturn,
    program_terminal_eos_tokens,
)
from coding_tasks import load_tasks
from delphi_qwen3 import DELPHI_EOS_ID, load_delphi, load_tokenizer


@dataclasses.dataclass
class MultiTurnTrainResult:
  """Result of a Delphi multi-turn coding SFT+Dr.GRPO run."""

  solve_ratio_history: list[float]
  first_solve_history: list[float]
  reward_history: list[float]
  steps_ran: int
  eval_after_sft: MultiTurnEvalResult | None
  eval_after_rl: MultiTurnEvalResult | None


class _MultiTurnMetricsCapture:
  """Captures per-step best/first solve + mean reward from the metrics buffer."""

  def __init__(self):
    self.solve_ratio_history: list[float] = []
    self.first_solve_history: list[float] = []
    self.reward_history: list[float] = []

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
        ("coding/solve_ratio", self.solve_ratio_history),
        ("coding/first_solve", self.first_solve_history),
        ("coding/reward_mean", self.reward_history),
    ):
      value = self._mean_of(metrics.get(name))
      if value is not None:
        history.append(value)


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
  return {key: _normalize_leaf(value) for key, value in example.items()}


class _NormalizingGRPOLearner(GRPOLearner):
  """Agentic GRPO learner that collapses each example's numpy leaves to Python scalars.

  The grain dataset batches ``prompts``/``answer`` as numpy string arrays; after
  size-1 micro-batching each example carries one-element arrays. We collapse them
  to plain ``str`` before the (agent, env) pair is built so the chat parser
  renders raw text and the env reads a clean gold string (mirrors the CALC
  ``_NormalizingGRPOLearner``).
  """

  def _create_agent_env_pair(self, single_example, group_id: int, pair_index: int):
    return super()._create_agent_env_pair(
        _normalize_example(single_example), group_id=group_id, pair_index=pair_index
    )


def _build_mesh() -> jax.sharding.Mesh:
  """Builds a colocated ``(ndev, 1)`` fsdp/tp mesh over all local devices."""
  return jax.sharding.Mesh(
      np.asarray(jax.devices()).reshape(jax.device_count(), 1),
      axis_names=("fsdp", "tp"),
  )


def _eval(actor, tokenizer, tasks, *, max_rounds, max_new_tokens, max_prompt_length, mesh, label):
  result = evaluate_tasks_multiturn(
      actor,
      tokenizer,
      tasks,
      max_rounds=max_rounds,
      max_new_tokens=max_new_tokens,
      max_prompt_length=max_prompt_length,
      mesh=mesh,
  )
  print(f"[mt-coding] EVAL {label}: {result.summary()}", flush=True)
  return result


def train_multiturn(
    *,
    model_dir: str,
    tiers: tuple[int, ...] = (3, 4, 5),
    rounds: int = 5,
    steps: int = 120,
    num_generations: int = 16,
    batch_size: int = 8,
    learning_rate: float = 1e-5,
    temperature: float = 1.0,
    max_prompt_length: int = 1024,
    max_response_length: int = 640,
    beta: float = 0.0,
    seed: int = 0,
    sft_steps: int = 0,
    sft_batch_size: int = 16,
    sft_learning_rate: float = 1e-4,
    sft_max_seq_len: int = 576,
    sft_fix_prob: float = 0.3,
    eval_max_rounds: int | None = None,
    eval_max_new_tokens: int = 192,
    eval_tiers: tuple[int, ...] | None = None,
    do_eval: bool = True,
    mesh: jax.sharding.Mesh | None = None,
) -> MultiTurnTrainResult:
  """Runs the Delphi multi-turn coding SFT warm-up + Dr.GRPO RL and evaluates.

  Args:
    model_dir: directory with Delphi's ``model.safetensors`` + tokenizer.
    tiers: curriculum tiers to TRAIN on (selects ``coding_env`` families).
    rounds: max write->run->revise rounds per episode (the env ``max_steps``).
    steps: Dr.GRPO global steps (0 = SFT/eval only).
    num_generations: Dr.GRPO group size (> 1 required).
    batch_size: prompts per global step.
    learning_rate: RL actor AdamW lr (global-norm clipped).
    temperature: rollout sampling temperature (intra-group diversity).
    max_prompt_length: max accumulated-conversation prompt length (few-shot +
      task + up to ``rounds`` of program/output turns).
    max_response_length: per-EPISODE generated+injected token budget across all
      turns; MUST equal the rollout ``max_tokens_to_generate`` (validated).
    beta: KL-to-reference weight (0 drops the reference model).
    seed: PRNG seed.
    sft_steps: multi-turn-format SFT warm-up steps (0 skips).
    sft_batch_size / sft_learning_rate / sft_max_seq_len: SFT knobs.
    sft_fix_prob: fraction of SFT transcripts that show a read-output-and-fix.
    eval_max_rounds: rounds for the greedy eval loop (defaults to ``rounds``).
    eval_max_new_tokens: per-turn greedy generation budget at eval.
    eval_tiers: tiers to EVAL on (defaults to ``tiers``); the fixed-task ladder is
      filtered to these.
    do_eval: greedy multi-turn eval after SFT and after RL.
    mesh: optional device mesh; defaults to a colocated ``(ndev, 1)`` mesh.

  Returns:
    A :class:`MultiTurnTrainResult` with per-step histories + the two evals.
  """
  tokenizer = load_tokenizer(model_dir)
  if mesh is None:
    mesh = _build_mesh()
  if eval_max_rounds is None:
    eval_max_rounds = rounds
  if eval_tiers is None:
    eval_tiers = tiers
  eval_task_list = [t for t in load_tasks() if t.tier in eval_tiers]

  # Load the actor FSDP-sharded on the mesh (required for SFT; keeps eval's
  # sampler on the same sharding either way).
  actor = load_delphi(model_dir, dtype=jnp.float32, mesh=mesh)
  reference = (
      None if beta == 0.0 else load_delphi(model_dir, dtype=jnp.bfloat16, mesh=mesh)
  )

  # Stage 1: SFT warm-up on the multi-turn execution-format transcripts.
  if sft_steps > 0:
    actor = run_sft_warmup(
        actor,
        tokenizer,
        steps=sft_steps,
        batch_size=sft_batch_size,
        learning_rate=sft_learning_rate,
        mesh=mesh,
        segment_fn=lambda rng: code_agent_segments(rng, tiers, fix_prob=sft_fix_prob),
        prompt_prefix=CODE_AGENT_SYSTEM_PROMPT,
        max_seq_len=sft_max_seq_len,
        seed=seed,
    )

  eval_after_sft = None
  if do_eval:
    eval_after_sft = _eval(
        actor,
        tokenizer,
        eval_task_list,
        max_rounds=eval_max_rounds,
        max_new_tokens=eval_max_new_tokens,
        max_prompt_length=max_prompt_length,
        mesh=mesh,
        label="after-sft" if sft_steps > 0 else "few-shot",
    )

  if steps <= 0:
    return MultiTurnTrainResult(
        solve_ratio_history=[],
        first_solve_history=[],
        reward_history=[],
        steps_ran=0,
        eval_after_sft=eval_after_sft,
        eval_after_rl=None,
    )

  # Stage 3: Dr.GRPO RL on the multi-turn rollout. Per-turn END stop (multi-line
  # programs); the base LM never emits EOS, so without it the first turn fills the
  # whole episode budget.
  eos_tokens = sorted(set([DELPHI_EOS_ID]) | set(program_terminal_eos_tokens(tokenizer)))
  if len(eos_tokens) <= 1:
    raise RuntimeError(
        "No 'END' terminal token found in the tokenizer vocab; per-turn stop "
        "would never fire. Check program_terminal_eos_tokens()."
    )

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

  # Dr.GRPO on the agentic path: drgrpo advantage (no std division) + constant
  # loss normalizer (no length bias), set directly on the agentic GRPOConfig.
  grpo_config = GRPOConfig(
      num_generations=num_generations,
      num_iterations=1,
      beta=beta,
      epsilon=0.2,
      advantage_estimator="drgrpo",
      loss_agg_mode="sequence-mean-token-scale",
      degenerate_group_masking=False,
      use_rollout_logps=False,
      system_prompt=CODE_AGENT_SYSTEM_PROMPT,
      max_response_length=max_response_length,
      max_concurrency=batch_size * num_generations,
  )

  rl_cluster = rl_cluster_lib.RLCluster(
      actor=actor,
      reference=reference,
      tokenizer=tokenizer,
      cluster_config=cluster_config,
  )

  # Fresh per-call rollout seed (else all group members are byte-identical ->
  # zero advantage -> no gradient).
  install_per_call_rollout_seed(rl_cluster, base_seed=seed)

  capture = _MultiTurnMetricsCapture()
  rl_cluster.with_external_metrics_logger(capture)

  learner = _NormalizingGRPOLearner(
      rl_cluster=rl_cluster,
      algo_config=grpo_config,
      reward_fns=None,  # the env's best-across-rounds terminal reward IS the reward
      chat_parser=DelphiRawTextChatParser(generation_suffix="\n"),
      metric_fns=[code_agent_metric_fn],
      agent_class=RunCodeAgent,
      env_class=CodeRunEnvironment,
      env_kwargs={"tool_map": {}, "max_steps": rounds},
  )

  train_ds = build_code_agent_dataset(
      n=steps * batch_size + batch_size,
      seed=seed,
      batch_size=batch_size,
      tiers=tiers,
  )

  learner.train(train_ds, eval_dataset=None)

  eval_after_rl = None
  if do_eval:
    eval_after_rl = _eval(
        rl_cluster.actor_trainer.model,
        tokenizer,
        eval_task_list,
        max_rounds=eval_max_rounds,
        max_new_tokens=eval_max_new_tokens,
        max_prompt_length=max_prompt_length,
        mesh=mesh,
        label="after-rl",
    )

  return MultiTurnTrainResult(
      solve_ratio_history=capture.solve_ratio_history,
      first_solve_history=capture.first_solve_history,
      reward_history=capture.reward_history,
      steps_ran=len(capture.reward_history),
      eval_after_sft=eval_after_sft,
      eval_after_rl=eval_after_rl,
  )
