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
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.agentic.agentic_grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.agentic.agents.model_agent import ModelAgent
from tunix.rl.agentic.environments.task_environment import TaskEnvironment
from tunix.rl.rollout import base_rollout

from agentic_common import DelphiRawTextChatParser, clipped_adamw
from agentic_sft import run_sft_warmup, t0_segments, t1_segments, t2_segments
from agentic_tools import (
    CalcToolEnvironment,
    DelphiToolAgent,
    T0_SYSTEM_PROMPT,
    T0_TOOL_MAP,
    T1_SYSTEM_PROMPT,
    T2_SYSTEM_PROMPT,
    arg_reward as t0_arg_reward,
    build_t0_dataset,
    build_t1_dataset,
    build_t2_dataset,
    install_per_call_rollout_seed,
    newline_terminal_eos_tokens,
    t0_metric_fn,
)
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


@dataclasses.dataclass
class T0TrainResult:
  """Result of a Delphi T0 tool-call GRPO run.

  Attributes:
    reward_history: per-global-step mean summed reward (env answer-in-output
      +1.0, learner arg +0.5, learner format +0.1).
    tool_call_rate_history: per-step ``tool/tool_call_rate`` (parseable turn-1
      tool-call rate; ~1.0 from cold).
    arg_acc_history: per-step ``tool/arg_acc`` (correct-operand rate; the KEY
      learnable signal, ~0.10 cold).
    solve_ratio_history: per-step ``arithmetic/solve_ratio`` (final-answer-
      contains-product rate).
    steps_ran: number of global steps with a captured reward.
  """

  reward_history: list[float]
  tool_call_rate_history: list[float]
  arg_acc_history: list[float]
  solve_ratio_history: list[float]
  steps_ran: int


class _T0MetricsCapture:
  """Captures per-step reward / tool_call_rate / arg_acc / solve_ratio.

  Mirrors :class:`_AgenticMetricsCapture` but reads the T0 reward fns'
  per-reward-fn means (``rewards/<fn>``) plus the T0 ``tool/`` and
  ``arithmetic/solve_ratio`` metrics emitted by
  :func:`agentic_tools.t0_metric_fn`. The summed reward also includes the env
  ``trajectory_reward`` (+1.0 answer-in-output), which the agentic reward
  manager folds into the per-fn rewards via ``trajectory_rewards``; we
  reconstruct the displayed mean from the learner reward fns and add the
  solve-driven trajectory term so the printed reward tracks the full signal.
  """

  def __init__(self):
    self.reward_history: list[float] = []
    self.tool_call_rate_history: list[float] = []
    self.arg_acc_history: list[float] = []
    self.solve_ratio_history: list[float] = []

  @staticmethod
  def _mean_of(entry: Any) -> float | None:
    if entry is None or not entry[0]:
      return None
    return float(np.mean(np.asarray(entry[0], dtype=np.float32)))

  def __call__(self, metrics_buffer, *args, **kwargs) -> None:
    del args, kwargs
    if "train" not in str(getattr(metrics_buffer, "mode", "")).lower():
      return
    metrics = getattr(metrics_buffer, "metrics", {})

    # Learner reward-fn means (arg + format); the env answer-in-output reward is
    # surfaced via solve_ratio below (it is the +1.0 trajectory term).
    reward_terms = [
        self._mean_of(metrics.get(f"rewards/{name}"))
        for name in ("arg_reward", "format_reward")
    ]
    reward_terms = [r for r in reward_terms if r is not None]

    tool_call_rate = self._mean_of(metrics.get("tool/tool_call_rate"))
    if tool_call_rate is not None:
      self.tool_call_rate_history.append(tool_call_rate)
    arg_acc = self._mean_of(metrics.get("tool/arg_acc"))
    if arg_acc is not None:
      self.arg_acc_history.append(arg_acc)
    solve = self._mean_of(metrics.get("arithmetic/solve_ratio"))
    if solve is not None:
      self.solve_ratio_history.append(solve)

    # Summed reward shown per step = learner terms + env answer term (~solve).
    if reward_terms or solve is not None:
      total = float(sum(reward_terms)) + (solve if solve is not None else 0.0)
      self.reward_history.append(total)
      # Stream the step live to stdout so long TPU runs show progress (and
      # partial results survive a late preemption) instead of only dumping the
      # full trajectory after train() returns.
      step = len(self.reward_history) - 1
      tcr = tool_call_rate if tool_call_rate is not None else float("nan")
      aacc = arg_acc if arg_acc is not None else float("nan")
      sr = solve if solve is not None else float("nan")
      print(
          f"[t0] step {step:4d}: mean_reward={total:.4f} "
          f"tool_call_rate={tcr:.4f} arg_acc={aacc:.4f} solve_ratio={sr:.4f}",
          flush=True,
      )


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
          # Global-norm-clipped AdamW: clipping is load-bearing (unclipped
          # multi-turn GRPO crashes the TPU with a libtpu SIGSEGV on inf/NaN
          # grads). See agentic_common.clipped_adamw for the full rationale.
          actor_optimizer=clipped_adamw(learning_rate),
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


def _train_agentic_calc(
    *,
    model_dir: str,
    dataset_builder=build_t0_dataset,
    sft_segment_fn=t0_segments,
    system_prompt: str = T0_SYSTEM_PROMPT,
    env_max_steps: int = 2,
    steps: int = 50,
    num_generations: int = 8,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    temperature: float = 0.9,
    # The few-shot tool-call system prompt is ~177 tokens, and the multi-turn
    # conversation accumulates (task + tool call + tool result + finish), so the
    # per-turn prompt grows well past M-port's. Size generously: a crash here is
    # "Total sampling steps N must be less than cache size" (kv_cache_size below).
    max_prompt_length: int = 640,
    max_tokens_to_generate: int = 96,
    beta: float = 0.0,
    seed: int = 0,
    eval_every_n_steps: int = 10**9,
    use_rollout_logps: bool = False,
    stop_on_newline: bool = True,
    sft_steps: int = 0,
    sft_batch_size: int = 16,
    sft_learning_rate: float = 1e-4,
    sft_prompt_prefix: str = "",
    sft_max_seq_len: int = 80,
    mesh: jax.sharding.Mesh | None = None,
) -> T0TrainResult:
  """Runs Delphi T0 single-calculator-call GRPO through the AGENTIC tool stack.

  Builds the full agentic ``RLCluster`` + ``GRPOLearner`` pipeline with the REAL
  Delphi model and the tunix tool stack: ``ToolEnvironment`` (stock
  ``CalculatorTool``) + :class:`agentic_tools.DelphiToolAgent` (suppressed Qwen
  tool docs + task-as-user-turn) + the suppressed Qwen ``<tool_call>`` parser +
  :class:`agentic_common.DelphiRawTextChatParser` (with ``generation_suffix="\\n"``
  so the model starts each turn on a fresh line and does NOT emit a leading
  newline that the newline-stop would discard).

  Shaped reward (dense gradient on operand-copy):
    * env ``reward_fn(task, action)`` -> +1.0 if the final answer contains the
      correct product (the ``trajectory_reward``).
    * learner ``arg_reward`` -> +0.5 if the turn-1 tool call carries the correct
      operands (the KEY learnable signal; ~0.10 cold).
    * learner ``format_reward`` -> +0.1 if the turn-1 text is a parseable call.

  Reward-kwarg flow (verified in ``agentic_grpo_learner._process_results``): the
  dataset's ``a`` / ``b`` / ``answer`` columns become the env task dict (the
  trajectory's ``original_input``); the learner forwards every ``original_input``
  column except ``prompts`` to the reward fns and the metric fn as kwargs. The
  ``completions`` passed to the learner reward fns are the FIRST assistant
  message (the turn-1 ``<tool_call>`` text).

  Args:
    model_dir: directory containing Delphi's ``model.safetensors`` + tokenizer.
    steps: number of GRPO global steps (``max_steps``).
    num_generations: GRPO group size (responses per prompt; must be > 1).
    batch_size: prompts per global step.
    learning_rate: AdamW learning rate.
    temperature: rollout sampling temperature.
    max_prompt_length: max prompt length (the few-shot system prompt + task; the
      T0 few-shot transcripts are ~180 tokens, so 256 is comfortable).
    max_tokens_to_generate: per-EPISODE completion budget across all turns; MUST
      equal ``GRPOConfig.max_response_length`` (enforced by the learner). A
      2-turn T0 episode uses ~40 tokens (call ~30 + tool result ~5 + answer ~2),
      so 96 leaves headroom.
    beta: KL-to-reference weight. ``0.0`` drops the reference model entirely.
    seed: PRNG seed for the problem set.
    eval_every_n_steps: eval stride (defaults to effectively never).
    use_rollout_logps: when True the rollout returns logprobs and the learner
      logs the sampler-vs-trainer logp diff; defaults False (recompute logps;
      sidesteps the 0.1.7 cross-turn logp-parity weakness).
    stop_on_newline: when True (default) the newline token is added to
      ``eos_tokens`` so each single-line turn (the tool call, the final answer)
      stops at its newline boundary and the episode stays well under the
      response budget. REQUIRED: a base LM never emits real EOS, so without a
      single-token stop every turn fills the whole budget and the agentic
      context-limit check discards it (empty completions, zero reward). The
      chat parser's ``generation_suffix="\\n"`` ensures the model emits content
      BEFORE the first newline (so the stop does not fire on an empty turn).
    sft_steps: if > 0, run that many SUPERVISED warm-up steps on synthetic CALC
      transcripts (via :func:`agentic_sft.run_sft_warmup`) on the same actor
      before RL -- makes the answer-COPY in-distribution so GRPO can amplify it
      (RL alone cannot bootstrap a behavior the base LM never samples). 0 skips.
    sft_batch_size: transcripts per SFT step.
    sft_learning_rate: AdamW lr for the SFT warm-up (clipped at global-norm 1.0).
    mesh: optional device mesh; defaults to a colocated ``(ndev, 1)`` mesh.

  Returns:
    A :class:`T0TrainResult` with the reward / tool_call_rate / arg_acc /
    solve_ratio histories.
  """
  tokenizer = load_tokenizer(model_dir)

  if mesh is None:
    mesh = _build_mesh()

  # When an SFT warm-up will run, the actor must be FSDP-sharded across the mesh
  # BEFORE SFT: PeftTrainer shards the optimizer state to the full mesh, and an
  # unsharded device-0 model trips "Received incompatible devices ... [0] vs
  # [0,1,2,3]". load_delphi shards at load time when given the mesh. The RL-only
  # path keeps loading unsharded (RLCluster reshards internally), unchanged.
  load_mesh = mesh if sft_steps > 0 else None
  actor = load_delphi(model_dir, dtype=jnp.float32, mesh=load_mesh)
  reference = (
      None
      if beta == 0.0
      else load_delphi(model_dir, dtype=jnp.bfloat16, mesh=load_mesh)
  )

  # Optional SFT warm-up (in-memory, same actor object). T0 RL masters the tool
  # CALL + operands but cannot bootstrap the final-answer COPY -- copying the
  # injected tool result is OOD for the base LM, so it is sampled too rarely for
  # GRPO to amplify (solve_ratio peaks ~0.1 then collapses). A few hundred SFT
  # steps on clean CALC transcripts put call+copy in-distribution; the warmed
  # actor flows straight into the RLCluster below. See agentic_sft.
  if sft_steps > 0:
    actor = run_sft_warmup(
        actor,
        tokenizer,
        steps=sft_steps,
        batch_size=sft_batch_size,
        learning_rate=sft_learning_rate,
        mesh=mesh,
        segment_fn=sft_segment_fn,
        prompt_prefix=sft_prompt_prefix,
        max_seq_len=sft_max_seq_len,
        seed=seed,
    )

  # Robust newline-stop: a base LM never emits real EOS, so each single-line
  # turn must stop at its newline. The Llama-3 BPE FUSES a newline with the
  # preceding char (``"</tool_call>\n"`` ends in token 397 ``">\n"``, not 198),
  # so a single-id newline stop misses it and the model runs past the call. We
  # stop on ANY token ending in a trailing newline (no token has an internal
  # newline, so this == "stop at the first newline" at the BPE level). See
  # ``agentic_tools.newline_terminal_eos_tokens``.
  eos_tokens = [DELPHI_EOS_ID]
  if stop_on_newline:
    eos_tokens = sorted(set(eos_tokens) | set(newline_terminal_eos_tokens(tokenizer)))

  kv_cache_size = max_prompt_length + max_tokens_to_generate + 8
  rollout_config = base_rollout.RolloutConfig(
      max_prompt_length=max_prompt_length,
      kv_cache_size=kv_cache_size,
      max_tokens_to_generate=max_tokens_to_generate,
      temperature=temperature,
      top_p=1.0,
      top_k=None,
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
          # Global-norm-clipped AdamW: clipping is load-bearing (unclipped
          # multi-turn GRPO crashes the TPU with a libtpu SIGSEGV on inf/NaN
          # grads). See agentic_common.clipped_adamw for the full rationale.
          actor_optimizer=clipped_adamw(learning_rate),
          eval_every_n_steps=eval_every_n_steps,
          max_steps=steps,
      ),
      rollout_config=rollout_config,
  )

  grpo_config = GRPOConfig(
      num_generations=num_generations,
      num_iterations=1,
      beta=beta,
      epsilon=0.2,
      advantage_estimator="grpo",
      degenerate_group_masking=False,
      use_rollout_logps=use_rollout_logps,
      # The agent's system content = system_prompt + (suppressed) tool docs; the
      # few-shot tool transcripts ARE the system prompt.
      system_prompt=system_prompt,
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

  # CRITICAL: vary the rollout seed per generation. tunix 0.1.7 generates each
  # group member with a SEPARATE generate() call but a FIXED RolloutConfig.seed,
  # so all num_generations samples would be byte-identical -> zero intra-group
  # variance -> zero GRPO advantage -> no gradient. This patches the rollout to
  # fold a fresh per-call seed. (See agentic_tools.install_per_call_rollout_seed.)
  install_per_call_rollout_seed(rl_cluster, base_seed=seed)

  capture = _T0MetricsCapture()
  rl_cluster.with_external_metrics_logger(capture)

  # CalcToolEnvironment computes its own copy-aware shaped TERMINAL reward
  # (copy term +0.4 if the final answer copies the executed tool result, solve
  # term +1.0 if it contains the gold product), so no reward_fn is passed via
  # env_kwargs. The learner arg_reward (+0.5, correct operands) is summed on top;
  # format_reward is dropped (redundant now that CALC parsing is closing-optional
  # and the dense copy term carries the turn-2 signal). The (agent, env) pair
  # builder forwards env_kwargs into CalcToolEnvironment(single_example,
  # tool_map=..., max_steps=2).
  learner = _NormalizingGRPOLearner(
      rl_cluster=rl_cluster,
      algo_config=grpo_config,
      reward_fns=[t0_arg_reward],
      chat_parser=DelphiRawTextChatParser(generation_suffix="\n"),
      metric_fns=[t0_metric_fn],
      agent_class=DelphiToolAgent,
      env_class=CalcToolEnvironment,
      env_kwargs={
          "tool_map": T0_TOOL_MAP,
          "max_steps": env_max_steps,
      },
  )

  train_ds = dataset_builder(
      n=steps * batch_size + batch_size,
      seed=seed,
      batch_size=batch_size,
  )

  learner.train(train_ds, eval_dataset=None)

  return T0TrainResult(
      reward_history=capture.reward_history,
      tool_call_rate_history=capture.tool_call_rate_history,
      arg_acc_history=capture.arg_acc_history,
      solve_ratio_history=capture.solve_ratio_history,
      steps_ran=len(capture.reward_history),
  )


def train_agentic_t0(**kwargs) -> T0TrainResult:
  """T0: a SINGLE calculator call (2-digit multiply).

  Thin wrapper over :func:`_train_agentic_calc` with its T0 defaults
  (``build_t0_dataset`` / ``t0_segments`` / ``T0_SYSTEM_PROMPT`` /
  ``env_max_steps=2``). See that function for the full pipeline + arg docs.
  """
  return _train_agentic_calc(**kwargs)


def train_agentic_t1(
    *,
    max_prompt_length: int = 768,
    max_tokens_to_generate: int = 128,
    **kwargs,
) -> T0TrainResult:
  """T1: TWO CHAINED calculator calls (``a * b * c``).

  Same agentic GRPO pipeline as T0, but the episode is three turns -- the model
  must (1) ``CALC(a * b)``, (2) COPY that ~4-digit intermediate into a second
  ``CALC(<a*b> * c)`` (true chaining: the turn-1 tool OUTPUT is a turn-2 ARGUMENT),
  and (3) copy the final product. ``env_max_steps=3`` admits the extra tool turn;
  the gold is the precomputed ``answer`` column (``a*b*c``), so the stock
  :class:`CalcToolEnvironment` copy/solve reward and ``t0_metric_fn`` carry over
  unchanged (``arg_acc`` still scores the turn-1 ``CALC(a * b)`` operands). The
  prompt/response budgets are larger than T0 to fit the extra turn + the longer
  intermediate. The SFT warm-up (:func:`agentic_sft.t1_segments`) is what makes
  the chained copy in-distribution; without it RL stalls exactly as T0 did.

  Args:
    max_prompt_length: max accumulated prompt length (system + up to 3 turns).
    max_tokens_to_generate: per-EPISODE budget across all 3 turns.
    **kwargs: forwarded to :func:`_train_agentic_calc` (steps, num_generations,
      batch_size, learning_rate, sft_steps, ...).
  """
  return _train_agentic_calc(
      dataset_builder=build_t1_dataset,
      sft_segment_fn=t1_segments,
      system_prompt=T1_SYSTEM_PROMPT,
      # Prepend the same few-shot prompt to the SFT transcripts (masked) so the
      # SFT context == the RL rollout prompt. Without this, T1's SFT corrupts the
      # turn-1 CALC emission (the few-shot ALONE gets it right; the mismatched SFT
      # breaks it). T0 does not need this (its mismatch is benign).
      sft_prompt_prefix=T1_SYSTEM_PROMPT,
      sft_max_seq_len=256,
      env_max_steps=3,
      max_prompt_length=max_prompt_length,
      max_tokens_to_generate=max_tokens_to_generate,
      **kwargs,
  )


def train_agentic_t2(
    *,
    max_prompt_length: int = 1024,
    max_tokens_to_generate: int = 192,
    **kwargs,
) -> T0TrainResult:
  """T2: THREE CHAINED calculator calls (``a * b * c * d``).

  A deeper chain than T1: the model must ``CALC(a * b)``, COPY it into
  ``CALC(<a*b> * c)``, COPY that ~6-digit intermediate into ``CALC(<a*b*c> * d)``,
  then copy the final ~8-digit product. ``env_max_steps=4`` admits the third tool
  turn; everything else is identical to T1 (same env/reward/metrics, gold is the
  precomputed ``answer`` = ``a*b*c*d``, ``arg_acc`` scores the turn-1 operands).
  The SFT warm-up uses :func:`agentic_sft.t2_segments` and -- as for T1 -- the
  few-shot prompt is prepended (masked) so the SFT context == the RL prompt; a
  larger ``sft_max_seq_len`` covers the longer demo + 4-turn transcript. The
  prompt/response budgets are larger to fit the extra turn and longer numbers.

  Args:
    max_prompt_length: max accumulated prompt length (system + up to 4 turns).
    max_tokens_to_generate: per-EPISODE budget across all 4 turns.
    **kwargs: forwarded to :func:`_train_agentic_calc`.
  """
  return _train_agentic_calc(
      dataset_builder=build_t2_dataset,
      sft_segment_fn=t2_segments,
      system_prompt=T2_SYSTEM_PROMPT,
      sft_prompt_prefix=T2_SYSTEM_PROMPT,
      sft_max_seq_len=384,
      env_max_steps=4,
      max_prompt_length=max_prompt_length,
      max_tokens_to_generate=max_tokens_to_generate,
      **kwargs,
  )
