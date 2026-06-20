"""Toy "emit more cats" GRPO smoke test for tunix on CPU (Milestone M2).

This module de-risks the *whole* tunix RL pipeline (rollout -> reward ->
group-relative advantage -> optimizer update) before any TPU spend, using a
TINY fresh-init ``tunix.models.qwen3`` model and a toy task: learn to emit a
chosen "cat" token more often than chance.

Design notes (why each choice):
  * **Model**: a fresh-init tunix Qwen3 (same class M1 validated against Delphi),
    deliberately tiny (vocab 32, 2 layers, embed 64). No checkpoint, no rope
    fix needed. Actor storage is **fp32** (per the GRPO wiring cheatsheet: bf16
    storage silently rounds the small Adam updates to zero and the policy never
    moves).
  * **Reference**: dropped (``beta=0.0``, ``reference=None``). For a toy this
    avoids a second model copy and a KL term that would only fight the signal.
  * **Reward (DENSE)**: fraction of generated tokens equal to the cat token, in
    ``[0, 1]``. A dense fraction (not a binary count) guarantees within-group
    reward variance early, which group-normalized GRPO needs for a non-zero
    advantage and hence an early gradient.
  * **Tokenizer**: a custom id-level toy tokenizer satisfying the minimal
    ``TokenizerType.NONE`` interface the vanilla Sampler requires (``encode``,
    ``decode``, ``bos_id``, ``eos_id``, ``pad_id``). ``decode`` maps the cat
    token id to the literal substring ``"cat"`` so the reward (which receives
    *decoded strings*) can count cat occurrences. The HF fallback the brief
    allowed was not needed.
  * **Rollout**: ``rollout_engine="vanilla"`` (the only backend that runs a
    custom arch), sampling temperature ~1.0 for the diversity GRPO needs.

Entry point: :func:`train_toy`. Run the gates via ``test_smoke_cats.py``.

Run with::

    JAX_PLATFORMS=cpu python -c "import toy_cats; print(toy_cats.train_toy())"
"""

import dataclasses

from flax import nnx
import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tunix.models.qwen3 import model as qm
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo import grpo_learner as grpo_lib
from tunix.rl.rollout import base_rollout


# ---- Toy vocabulary layout ------------------------------------------------
# A tiny vocabulary. Reserved low ids mirror the usual special-token slots so
# the tokenizer interface is unsurprising; the rest are ordinary content tokens.
VOCAB_SIZE = 32
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
# The "cat" token. Any ordinary content id works; pick one well clear of the
# special ids so it cannot be confused with pad/bos/eos.
CAT_ID = 7
# The fixed prompt: a single BOS token. Completions are sampled from there.
PROMPT_STR = "<bos>"


class ToyTokenizer:
  """Minimal id-level tokenizer for the vanilla tunix Sampler.

  Satisfies the ``TokenizerType.NONE`` branch of tunix's ``TokenizerAdapter``
  (it has ``encode``/``decode``/``bos_id``/``eos_id``/``pad_id``). Tokens are
  encoded/decoded as whitespace-joined ``"t<id>"`` words, except the cat token
  which decodes to the bare word ``"cat"`` and BOS which encodes from the fixed
  ``PROMPT_STR``. This lets the reward function operate on the decoded string:
  it simply counts ``"cat"`` words.

  ``bos_id`` returns a *truthy* id so the Sampler prepends BOS to the prompt
  (``sampler.tokenize``: ``bos_tok = [bos_id] if bos_id else []``); the toy
  prompt is therefore exactly ``[BOS]``.
  """

  def __init__(self, cat_id: int = CAT_ID):
    """Initializes the toy tokenizer.

    Args:
      cat_id: the token id that decodes to the literal word ``"cat"``.
    """
    self._cat_id = cat_id

  def encode(self, text: str, **kwargs) -> list[int]:
    """Encodes a space-separated string of token words to ids.

    Recognizes the special prompt word ``"<bos>"`` and ``"cat"``; any other
    ``"t<id>"`` word is parsed back to its integer id. Unknown words are
    dropped (the toy only ever encodes the fixed prompt).

    The input is coerced to ``str`` first: tunix's data pipeline can hand the
    prompt through as a 0-d/1-element ``numpy`` array or ``numpy.str_`` rather
    than a Python ``str``, so we normalize defensively.

    Args:
      text: a space-separated string of token words (or numpy str/array).
      **kwargs: ignored (HF-compat).

    Returns:
      The list of token ids.
    """
    del kwargs
    text = np.asarray(text).item() if not isinstance(text, str) else text
    ids: list[int] = []
    for word in text.split(" "):
      if not word:
        continue
      if word == "<bos>":
        ids.append(BOS_ID)
      elif word == "cat":
        ids.append(self._cat_id)
      elif word.startswith("t"):
        ids.append(int(word[1:]))
    return ids

  def decode(self, ids: list[int], **kwargs) -> str:
    """Decodes ids to a space-separated string of token words.

    The cat id decodes to the bare word ``"cat"`` (so the reward can count
    occurrences); every other id decodes to ``"t<id>"``.

    Args:
      ids: the token ids to decode.
      **kwargs: ignored (HF-compat).

    Returns:
      A space-separated string of token words.
    """
    del kwargs
    words = ["cat" if i == self._cat_id else f"t{int(i)}" for i in ids]
    return " ".join(words)

  def bos_id(self) -> int:
    """Returns the beginning-of-sequence token id (truthy => prepended)."""
    return BOS_ID

  def eos_id(self) -> int:
    """Returns the end-of-sequence token id."""
    return EOS_ID

  def pad_id(self) -> int:
    """Returns the padding token id."""
    return PAD_ID


def toy_config(
    *,
    dtype: jnp.dtype = jnp.float32,
    param_dtype: jnp.dtype = jnp.float32,
) -> qm.ModelConfig:
  """Returns the tiny fresh-init Qwen3 config for the toy.

  Deliberately small so CPU jit of the sampler ``while_loop`` is tractable. Same
  ``qm.Qwen3`` class M1 validated against Delphi. ``param_dtype`` defaults to
  fp32 because the actor's small Adam updates round to zero under bf16 storage.

  Args:
    dtype: compute dtype for activations.
    param_dtype: storage dtype for parameters (keep fp32 for the actor).

  Returns:
    A tiny ``qm.ModelConfig``.
  """
  return qm.ModelConfig(
      num_layers=2,
      vocab_size=VOCAB_SIZE,
      embed_dim=64,
      hidden_dim=128,
      num_heads=4,
      head_dim=16,
      num_kv_heads=4,  # no GQA
      rope_theta=1_000_000,
      norm_eps=1e-5,
      use_tied_embedding=False,
      dtype=dtype,
      param_dtype=param_dtype,
  )


def build_model(seed: int = 0) -> qm.Qwen3:
  """Builds a fresh-init tiny Qwen3 actor in fp32.

  Args:
    seed: PRNG seed for the fresh init.

  Returns:
    A live ``qm.Qwen3`` nnx module (fp32 params).
  """
  return qm.Qwen3(toy_config(), rngs=nnx.Rngs(seed))


class _PromptSource(grain.RandomAccessDataSource):
  """A fixed-prompt grain source (every row is the same toy prompt)."""

  def __init__(self, num_rows: int):
    """Initializes the source.

    Args:
      num_rows: number of rows to expose.
    """
    self._num_rows = num_rows

  def __len__(self) -> int:
    return self._num_rows

  def __getitem__(self, idx: int) -> str:
    del idx
    return PROMPT_STR


def build_dataset(num_rows: int, batch_size: int) -> grain.MapDataset:
  """Builds the GRPO training dataset as a batched grain dataset.

  Every row is the same fixed prompt; the learning is entirely in the policy's
  completion distribution. The column is named ``"prompts"`` exactly as the
  GRPOLearner contract requires. We use grain (not HF ``datasets``) so the
  batched ``prompts`` column is a single ``numpy`` array leaf; HF ``.batch()``
  yields a Python ``list`` per row, which tunix's ``jax.tree.map(np.repeat,...)``
  step recurses into per-character, corrupting the prompt batch.

  Args:
    num_rows: number of rows (>= max_steps * batch_size).
    batch_size: prompts per global step.

  Returns:
    A batched ``grain.MapDataset`` with a single ``"prompts"`` column.
  """
  return (
      grain.MapDataset.source(_PromptSource(num_rows))
      .batch(batch_size)
      .map(lambda x: {"prompts": x})
  )


# A module-level side-channel so the smoke test can directly inspect what the
# rollout actually produced (the WIRED gate), without reaching into learner
# internals. Each entry is one rollout group's decoded completions. Reset at the
# start of each train_toy run. The FIRST entry shows early-training sampling
# diversity; the LAST shows the converged behavior.
_COMPLETION_BATCHES: list[list[str]] = []


def cat_reward(prompts, completions, **kwargs) -> list[float]:
  """Dense reward: fraction of generated tokens equal to the cat token.

  ``completions`` are *decoded strings* (tunix decodes rollout tokens before
  calling reward fns). With the toy tokenizer each generated token is one
  whitespace word, and the cat token decodes to ``"cat"``; so the cat fraction
  is ``count("cat") / num_words``. Range ``[0, 1]``. Dense by construction,
  giving GRPO within-group reward variance early.

  Args:
    prompts: the batch of prompt strings (unused).
    completions: the batch of decoded completion strings.
    **kwargs: forwarded dataset columns (none here).

  Returns:
    One float in ``[0, 1]`` per completion.
  """
  del prompts, kwargs
  _COMPLETION_BATCHES.append(list(completions))
  rewards: list[float] = []
  for text in completions:
    words = text.split()
    if not words:
      rewards.append(0.0)
      continue
    rewards.append(sum(1 for w in words if w == "cat") / len(words))
  return rewards


def metric_fn(prompts, completions, rewards, advantages, **kwargs) -> dict:
  """Reports cat-fraction summary stats for logging.

  Args:
    prompts: the batch of prompts (unused).
    completions: the batch of decoded completions (unused).
    rewards: per-completion rewards (the cat fractions).
    advantages: per-completion advantages (unused).
    **kwargs: forwarded dataset columns (none here).

  Returns:
    A dict of metric name -> (value, aggregation_fn).
  """
  del prompts, completions, advantages, kwargs
  rewards = np.asarray(rewards, dtype=np.float32)
  return {
      "cat_fraction/mean": (float(rewards.mean()), np.mean),
      "cat_fraction/max": (float(rewards.max()), np.max),
  }


@dataclasses.dataclass
class ToyResult:
  """Result of a toy GRPO run.

  Attributes:
    reward_history: per-global-step mean cat-fraction reward.
    checkpoint_means: (step, mean-reward) sampled every ``log_every`` steps.
    first_completions: decoded completions from the FIRST rollout (for the WIRED
      gate: early-training sampling diversity, before the policy converges).
    last_completions: decoded completions from the final rollout (lengths > 1).
    model: the trained actor (for direct cache-path probing).
    tokenizer: the toy tokenizer.
  """

  reward_history: list[float]
  checkpoint_means: list[tuple[int, float]]
  first_completions: list[str]
  last_completions: list[str]
  model: qm.Qwen3
  tokenizer: ToyTokenizer


class _RewardHistoryLogger:
  """Captures the per-step ``rewards/score/mean`` (mean cat-fraction) curve.

  The learner buffers ``rewards/score/mean`` (grpo_learner.py:327) every global
  step. ``RLCluster.with_external_metrics_logger`` installs a callable invoked
  with a ``MetricsBuffer`` whose ``.metrics`` maps metric name ->
  ``([values], aggregation_fn)`` and which carries ``.mode``/``.global_steps``.
  We read the train-mode ``rewards/score/mean`` and append the per-step mean,
  which is exactly the training reward curve we gate on.
  """

  def __init__(self):
    self.history: list[float] = []

  def __call__(self, metrics_buffer, *args, **kwargs):
    """Records the mean reward from one flushed train metrics buffer."""
    del args, kwargs
    if "train" not in str(getattr(metrics_buffer, "mode", "")):
      return
    entry = getattr(metrics_buffer, "metrics", {}).get("rewards/score/mean")
    if entry is None:
      return
    values = entry[0]
    if not values:
      return
    self.history.append(float(np.mean(np.asarray(values, dtype=np.float32))))


def train_toy(
    *,
    steps: int = 80,
    num_generations: int = 8,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    temperature: float = 1.0,
    max_prompt_length: int = 8,
    max_tokens_to_generate: int = 16,
    seed: int = 0,
    log_every: int = 10,
) -> ToyResult:
  """Builds the cluster + GRPO learner and runs the toy "emit more cats" loop.

  Args:
    steps: number of GRPO global steps.
    num_generations: GRPO group size (responses per prompt; advantage is
      normalized within the group, so >1 is required).
    batch_size: prompts per global step (the on-policy batch is
      ``batch_size * num_generations``).
    learning_rate: AdamW learning rate. Larger than a finetuning LR because the
      model is fresh-init and tiny.
    temperature: rollout sampling temperature (~1.0 for GRPO diversity).
    max_prompt_length: max prompt length (the toy prompt is just BOS).
    max_tokens_to_generate: completion length (constant; the toy has no eos in
      its content tokens, so completions run the full budget).
    seed: PRNG seed for the fresh actor init.
    log_every: checkpoint sampling stride for the reported trajectory.

  Returns:
    A :class:`ToyResult` with the reward history and the trained artifacts.
  """
  _COMPLETION_BATCHES.clear()

  actor = build_model(seed=seed)
  tokenizer = ToyTokenizer()

  # Colocated mesh shared across all roles. fsdp spans every available device
  # (1 on CPU, N on a single-host TPU); tp is length-1. The tiny toy dims are
  # divisible by typical single-host TPU chip counts so fsdp sharding is safe.
  mesh = jax.sharding.Mesh(
      np.asarray(jax.devices()).reshape(jax.device_count(), 1),
      axis_names=("fsdp", "tp"),
  )

  kv_cache_size = max_prompt_length + max_tokens_to_generate + 8
  rollout_config = base_rollout.RolloutConfig(
      max_prompt_length=max_prompt_length,
      kv_cache_size=kv_cache_size,
      max_tokens_to_generate=max_tokens_to_generate,
      temperature=temperature,
      top_p=1.0,
      top_k=None,
      return_logprobs=False,
      # eos is a content-free special token the policy is unlikely to emit;
      # leaving it as the only stop token means completions run the full budget,
      # which keeps every group the same length (cleaner cat-fraction).
      eos_tokens=[EOS_ID],
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
          eval_every_n_steps=10**9,  # no eval pass for the toy
          max_steps=steps,
      ),
      rollout_config=rollout_config,
  )

  grpo_config = grpo_lib.GRPOConfig(
      num_generations=num_generations,
      num_iterations=1,
      beta=0.0,  # no KL / no reference model for the toy
      epsilon=0.2,
  )

  # beta=0.0 => no reference model needed.
  rl_cluster = rl_cluster_lib.RLCluster(
      actor=actor,
      reference=None,
      tokenizer=tokenizer,
      cluster_config=cluster_config,
  )

  reward_logger = _RewardHistoryLogger()
  rl_cluster.with_external_metrics_logger(reward_logger)

  learner = grpo_lib.GRPOLearner(
      rl_cluster=rl_cluster,
      algo_config=grpo_config,
      reward_fns=[cat_reward],
      metric_fns=[metric_fn],
  )

  # Enough rows to cover steps * batch_size prompts.
  train_ds = build_dataset(
      num_rows=steps * batch_size + batch_size, batch_size=batch_size
  )

  learner.train(train_ds, eval_ds=None)

  history = reward_logger.history
  checkpoint_means = [
      (i, history[i]) for i in range(0, len(history), log_every)
  ]
  if history and (len(history) - 1) % log_every != 0:
    checkpoint_means.append((len(history) - 1, history[-1]))

  return ToyResult(
      reward_history=history,
      checkpoint_means=checkpoint_means,
      first_completions=list(_COMPLETION_BATCHES[0]) if _COMPLETION_BATCHES else [],
      last_completions=list(_COMPLETION_BATCHES[-1]) if _COMPLETION_BATCHES else [],
      model=actor,
      tokenizer=tokenizer,
  )


def probe_cache_advances(
    model: qm.Qwen3,
    *,
    prompt_ids: list[int] | None = None,
    max_new_tokens: int = 8,
) -> tuple[list[int], list[int]]:
  """Directly exercises the KV cache and records its ``end_index`` per step.

  Mirrors M1's ``greedy_generate``: prefill the prompt, then decode one token at
  a time threading the ``{'k','v','end_index'}`` ring-buffer cache. Returns the
  generated tokens and the cache ``end_index`` after each forward, proving the
  cache path advances (the WIRED gate, asserted directly rather than inferred
  from reward-go-up).

  Args:
    model: the (trained) toy actor.
    prompt_ids: prompt token ids; defaults to ``[BOS_ID]``.
    max_new_tokens: number of decode steps to run.

  Returns:
    A tuple ``(generated_token_ids, end_index_per_step)``.
  """
  if prompt_ids is None:
    prompt_ids = [BOS_ID]
  prompt_len = len(prompt_ids)
  cache_size = prompt_len + max_new_tokens + 1
  cache = model.init_cache(
      batch_size=1, cache_size=cache_size, dtype=model.config.dtype
  )

  ids = jnp.asarray(prompt_ids, dtype=jnp.int32)[None, :]
  positions = jnp.arange(prompt_len)[None, :]
  causal = jnp.tril(jnp.ones((prompt_len, prompt_len), dtype=jnp.bool_))
  prefill_mask = jnp.zeros((1, prompt_len, cache_size), dtype=jnp.bool_)
  prefill_mask = prefill_mask.at[:, :, :prompt_len].set(causal[None])
  logits, cache = model(ids, positions, cache, prefill_mask)

  end_indices = [int(cache["layer_0"]["end_index"][0])]
  next_tok = int(jnp.argmax(logits[0, -1]))
  generated = [next_tok]

  cur_pos = prompt_len
  for _ in range(max_new_tokens - 1):
    step_ids = jnp.asarray([[next_tok]], dtype=jnp.int32)
    step_pos = jnp.asarray([[cur_pos]], dtype=jnp.int32)
    step_mask = (jnp.arange(cache_size) <= cur_pos)[None, None, :]
    logits, cache = model(step_ids, step_pos, cache, step_mask)
    end_indices.append(int(cache["layer_0"]["end_index"][0]))
    next_tok = int(jnp.argmax(logits[0, -1]))
    generated.append(next_tok)
    cur_pos += 1

  return generated, end_indices
