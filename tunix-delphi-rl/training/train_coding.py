# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Delphi agentic-CODING training driver (issue #7): SFT warm-up -> Dr.GRPO RL.

Single-turn code generation through the proven NON-agentic GRPO wiring
(``train_delphi.py``): the model writes a Python program, the :mod:`micropython`
interpreter executes + grades it, and the reward is an exact stdout match. The
recipe is the same three-stage bootstrap codified in ``AGENTS.md``:

  1. **SFT for token/program format** -- :func:`agentic_sft.run_sft_warmup` on
     ``coding_env.code_segments`` (the ``Task: ... <program> END`` transcript)
     with ``prompt_prefix=CODE_FEWSHOT`` so the SFT context == the RL prompt
     (invariant D). This puts "emit a valid program in the format" in
     distribution before RL.
  2. **(no separate tool-call SFT)** -- the "tool" is the interpreter, invoked by
     the grader, not called in-band; the program *is* the action.
  3. **Dr.GRPO RL** -- :class:`DrGRPOLearner` (advantage = group-mean-centered with
     NO std division; loss = constant-normalized, NO per-response length bias),
     which is more robust on a tiny base model than vanilla GRPO. A dense reward
     (:func:`coding_env.code_reward`) bootstraps from the SFT-warmed policy.

Eval is greedy decode on the fixed 50-task ladder (``coding_tasks.py``) -- once
after the SFT warm-up (the pre-RL baseline; with ``sft_steps=0`` this is the
few-shot-only baseline) and once after RL -- so a single run shows the SFT->RL
lift per tier. "How far did we get" = the after-RL per-tier solve rate.

Reused invariants (see ``AGENTS.md``): fp32 actor storage; ``clipped_adamw``
(global-norm clip, crash-prevention); ``beta=0`` (no reference model);
``rollout_engine="vanilla"``; the base LM never emits EOS so generation runs to
the token budget and :func:`coding_env.extract_program` cuts at the ``END``
sentinel (no newline-stop -- programs are multi-line).
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo.drgrpo_learner import DrGRPOConfig, DrGRPOLearner
from tunix.rl.rollout import base_rollout

from training.agentic_common import clipped_adamw
from training.agentic_sft import run_sft_warmup
from environments.coding_env import (
    CODE_FEWSHOT,
    CodingEvalResult,
    build_code_dataset,
    code_metric_fn,
    code_reward,
    code_segments,
    evaluate_tasks,
)
from problems.coding_tasks import load_tasks
from models.delphi_qwen3 import DELPHI_EOS_ID, load_delphi, load_tokenizer


@dataclasses.dataclass
class CodingTrainResult:
  """Result of a Delphi coding SFT+Dr.GRPO run.

  Attributes:
    reward_history: per-step mean summed reward.
    solve_ratio_history: per-step ``coding/solve_ratio`` (exact-match on the
      randomized training distribution).
    ran_ok_history: per-step ``coding/ran_ok`` (programs that ran without error).
    has_code_history: per-step ``coding/has_code`` (non-empty programs emitted).
    steps_ran: number of RL global steps executed.
    eval_after_sft: greedy eval on the fixed 50 tasks after the SFT warm-up
      (the pre-RL baseline), or None if not evaluated.
    eval_after_rl: greedy eval on the fixed 50 tasks after RL, or None.
  """

  reward_history: list[float]
  solve_ratio_history: list[float]
  ran_ok_history: list[float]
  has_code_history: list[float]
  steps_ran: int
  eval_after_sft: CodingEvalResult | None
  eval_after_rl: CodingEvalResult | None


class _CodeMetricsCapture:
  """Captures per-step coding metrics from the train-mode metrics buffer."""

  def __init__(self):
    self.reward_history: list[float] = []
    self.solve_ratio_history: list[float] = []
    self.ran_ok_history: list[float] = []
    self.has_code_history: list[float] = []

  @staticmethod
  def _mean_of(entry):
    if entry is None or not entry[0]:
      return None
    return float(np.mean(np.asarray(entry[0], dtype=np.float32)))

  def __call__(self, metrics_buffer, *args, **kwargs):
    del args, kwargs
    if "train" not in str(getattr(metrics_buffer, "mode", "")):
      return
    metrics = getattr(metrics_buffer, "metrics", {})
    reward = self._mean_of(metrics.get("rewards/score/mean"))
    if reward is not None:
      self.reward_history.append(reward)
    for name, history in (
        ("coding/solve_ratio", self.solve_ratio_history),
        ("coding/ran_ok", self.ran_ok_history),
        ("coding/has_code", self.has_code_history),
    ):
      value = self._mean_of(metrics.get(name))
      if value is not None:
        history.append(value)


def _build_mesh() -> jax.sharding.Mesh:
  """Builds a colocated ``(ndev, 1)`` fsdp/tp mesh over all local devices."""
  return jax.sharding.Mesh(
      np.asarray(jax.devices()).reshape(jax.device_count(), 1),
      axis_names=("fsdp", "tp"),
  )


def _eval(actor, tokenizer, *, max_new_tokens, max_prompt_length, mesh, label):
  """Greedy-evaluates ``actor`` on the fixed 50 tasks and prints a summary."""
  result = evaluate_tasks(
      actor,
      tokenizer,
      load_tasks(),
      max_new_tokens=max_new_tokens,
      max_prompt_length=max_prompt_length,
      mesh=mesh,
  )
  print(f"[coding] EVAL {label}: {result.summary()}", flush=True)
  return result


def train_coding(
    *,
    model_dir: str,
    tiers: tuple[int, ...] = (0, 1, 2, 3, 4),
    steps: int = 120,
    num_generations: int = 16,
    batch_size: int = 8,
    learning_rate: float = 1e-5,
    temperature: float = 0.9,
    max_prompt_length: int = 384,
    max_tokens_to_generate: int = 160,
    beta: float = 0.0,
    seed: int = 0,
    eval_every_n_steps: int = 10**9,
    sft_steps: int = 0,
    sft_batch_size: int = 16,
    sft_learning_rate: float = 1e-4,
    sft_max_seq_len: int = 384,
    eval_max_new_tokens: int = 160,
    do_eval: bool = True,
    mesh: jax.sharding.Mesh | None = None,
) -> CodingTrainResult:
  """Runs the Delphi coding SFT warm-up + Dr.GRPO RL and evaluates on the ladder.

  Args:
    model_dir: directory with Delphi's ``model.safetensors`` + tokenizer.
    tiers: curriculum tiers to train on (selects ``coding_env`` families).
    steps: number of Dr.GRPO global steps (0 = SFT/few-shot eval only).
    num_generations: Dr.GRPO group size (responses per prompt; > 1 required).
    batch_size: prompts per global step.
    learning_rate: RL actor AdamW lr (global-norm clipped).
    temperature: rollout sampling temperature (diversity for the group).
    max_prompt_length: max prompt length (few-shot prefix + task).
    max_tokens_to_generate: per-episode program budget (multi-line programs;
      no newline-stop, so generation runs the full budget and the parser cuts
      at ``END``).
    beta: KL-to-reference weight (0 drops the reference model).
    seed: PRNG seed for the problem set.
    eval_every_n_steps: in-loop eval stride (defaults to effectively never).
    sft_steps: SFT warm-up steps before RL (0 skips -> few-shot-only baseline).
    sft_batch_size: transcripts per SFT step.
    sft_learning_rate: SFT AdamW lr (global-norm clipped).
    sft_max_seq_len: padded SFT transcript length (few-shot prefix + program;
      the tier-4 transcripts are the longest).
    eval_max_new_tokens: greedy generation budget for the fixed-task eval.
    do_eval: when True, greedy-eval the fixed 50 tasks after SFT and after RL.
    mesh: optional device mesh; defaults to a colocated ``(ndev, 1)`` mesh.

  Returns:
    A :class:`CodingTrainResult` with the per-step histories + the two evals.
  """
  tokenizer = load_tokenizer(model_dir)
  if mesh is None:
    mesh = _build_mesh()

  # Load the actor FSDP-sharded on the mesh up front: required when an SFT
  # warm-up runs (PeftTrainer shards optimizer state to the full mesh), and it
  # keeps the greedy eval's sampler on the same sharding either way.
  actor = load_delphi(model_dir, dtype=jnp.float32, mesh=mesh)
  reference = (
      None if beta == 0.0 else load_delphi(model_dir, dtype=jnp.bfloat16, mesh=mesh)
  )

  # Stage 1: SFT warm-up on coding transcripts (Task -> program -> END), with
  # the few-shot prefix prepended masked so the SFT context matches the RL
  # prompt exactly (invariant D). Same in-memory actor flows into RL.
  if sft_steps > 0:
    actor = run_sft_warmup(
        actor,
        tokenizer,
        steps=sft_steps,
        batch_size=sft_batch_size,
        learning_rate=sft_learning_rate,
        mesh=mesh,
        segment_fn=lambda rng: code_segments(rng, tiers),
        prompt_prefix=CODE_FEWSHOT,
        max_seq_len=sft_max_seq_len,
        seed=seed,
    )

  eval_after_sft = None
  if do_eval:
    eval_after_sft = _eval(
        actor,
        tokenizer,
        max_new_tokens=eval_max_new_tokens,
        max_prompt_length=max_prompt_length,
        mesh=mesh,
        label="after-sft" if sft_steps > 0 else "few-shot",
    )

  if steps <= 0:
    return CodingTrainResult(
        reward_history=[],
        solve_ratio_history=[],
        ran_ok_history=[],
        has_code_history=[],
        steps_ran=0,
        eval_after_sft=eval_after_sft,
        eval_after_rl=None,
    )

  # Stage 3: Dr.GRPO RL. No newline-stop (programs are multi-line); the base LM
  # never emits EOS, so generation runs to the budget and the parser cuts at END.
  kv_cache_size = max_prompt_length + max_tokens_to_generate + 8
  rollout_config = base_rollout.RolloutConfig(
      max_prompt_length=max_prompt_length,
      kv_cache_size=kv_cache_size,
      max_tokens_to_generate=max_tokens_to_generate,
      temperature=temperature,
      top_p=1.0,
      top_k=None,
      return_logprobs=False,
      eos_tokens=[DELPHI_EOS_ID],
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
          # Global-norm-clipped AdamW (crash-prevention; see clipped_adamw).
          actor_optimizer=clipped_adamw(learning_rate),
          eval_every_n_steps=eval_every_n_steps,
          max_steps=steps,
      ),
      rollout_config=rollout_config,
  )

  # Dr.GRPO: advantage = r - group_mean (no /std), loss = constant-normalized
  # (no per-response length bias). Drop-in for GRPOConfig/GRPOLearner.
  drgrpo_config = DrGRPOConfig(
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

  capture = _CodeMetricsCapture()
  rl_cluster.with_external_metrics_logger(capture)

  learner = DrGRPOLearner(
      rl_cluster=rl_cluster,
      algo_config=drgrpo_config,
      reward_fns=[code_reward],
      metric_fns=[code_metric_fn],
  )

  train_ds = build_code_dataset(
      n=steps * batch_size + batch_size,
      seed=seed,
      batch_size=batch_size,
      tiers=tiers,
  )

  learner.train(train_ds, eval_ds=None)

  eval_after_rl = None
  if do_eval:
    eval_after_rl = _eval(
        rl_cluster.actor_trainer.model,
        tokenizer,
        max_new_tokens=eval_max_new_tokens,
        max_prompt_length=max_prompt_length,
        mesh=mesh,
        label="after-rl",
    )

  return CodingTrainResult(
      reward_history=capture.reward_history,
      solve_ratio_history=capture.solve_ratio_history,
      ran_ok_history=capture.ran_ok_history,
      has_code_history=capture.has_code_history,
      steps_ran=len(capture.reward_history),
      eval_after_sft=eval_after_sft,
      eval_after_rl=eval_after_rl,
  )
