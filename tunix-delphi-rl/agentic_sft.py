"""Supervised warm-up for the Delphi agentic tool stages (T0 / T1).

RL masters the tool CALL + operands but cannot bootstrap the COPY behaviors that
ground a multi-turn tool episode -- copying the injected ``Tool result: X`` into
the next call (chaining) or into the final answer is out-of-distribution for the
447M Delphi base LM, so it is sampled too rarely for GRPO to amplify (T0
``solve_ratio`` peaked ~0.1 then collapsed as the policy sharpened). This is the
classic "RL only amplifies what the base policy already samples" wall.

This module does the standard fix: a short SUPERVISED fine-tune that makes the
call+copy pattern in-distribution BEFORE RL, using tunix's stock
:class:`~tunix.sft.peft_trainer.PeftTrainer` on the SAME in-memory model object
(no checkpoint round-trip). The warmed ``actor`` flows straight into the
``RLCluster``; the handoff works because both phases mutate the same ``nnx``
module in place and ``RLCluster`` re-shards as needed.

Each transcript is the clean tool episode for its stage:

  * T0 (single call)::            Q: a * b
                                  CALC(a * b)
                                  Tool result: a*b
                                  a*b
  * T1 (two chained calls)::      Q: a * b * c
                                  CALC(a * b)
                                  Tool result: a*b
                                  CALC(<a*b> * c)
                                  Tool result: a*b*c
                                  a*b*c

with a per-turn loss mask: train (mask 1) on the MODEL's turns -- every
``CALC(...)`` call and the final answer -- and NOT (mask 0) on the question or
the environment-provided ``Tool result:`` lines (the env emits those at RL time;
the model must learn to COPY them, not produce them). ``positions`` and the
causal ``attention_mask`` are built from a separate PADDING mask, exactly as the
rollout / RL loss path does it.

The PeftTrainer's default loss is next-token NLL over ``input_mask`` (see
``peft_trainer._default_loss_fn``); the default ``gen_model_input_fn`` is the
identity, so we install :func:`sft_model_input_fn` to expand each batched
``{input_tokens, loss_mask, pad_mask}`` row into the loss-fn kwargs.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Tuple

import grain.python as grain
import jax
import numpy as np
import optax

from tunix.sft import utils as sft_utils
from tunix.sft.peft_trainer import PeftTrainer, TrainingConfig

from delphi_qwen3 import DELPHI_BOS_ID, DELPHI_EOS_ID

# A transcript is a list of (text, train_on_it?) segments. Mask 1 == the model's
# own turns (the tool calls + the copied answer); mask 0 == context the model
# conditions on but must not be trained to emit (the question, the tool results).
Segments = List[Tuple[str, int]]
SegmentFn = Callable[[random.Random], Segments]


def t0_segments(rng: random.Random) -> Segments:
  """One single-call ``a * b`` transcript (T0)."""
  a = rng.randint(11, 99)
  b = rng.randint(11, 99)
  gold = a * b
  return [
      (f"Q: {a} * {b}\n", 0),
      (f"CALC({a} * {b})\n", 1),
      (f"Tool result: {gold}\n", 0),
      (f"{gold}\n", 1),
  ]


def t1_segments(rng: random.Random) -> Segments:
  """One chained two-call ``a * b * c`` transcript (T1).

  The second call ``CALC(<a*b> * c)`` carries the turn-1 RESULT as an argument,
  so the warm-up teaches chaining (copy the tool output into the next call), not
  just two independent calls.
  """
  a = rng.randint(11, 99)
  b = rng.randint(11, 99)
  c = rng.randint(11, 99)
  ab = a * b
  gold = ab * c
  return [
      (f"Q: {a} * {b} * {c}\n", 0),
      (f"CALC({a} * {b})\n", 1),
      (f"Tool result: {ab}\n", 0),
      (f"CALC({ab} * {c})\n", 1),
      (f"Tool result: {gold}\n", 0),
      (f"{gold}\n", 1),
  ]


def t2_segments(rng: random.Random) -> Segments:
  """One chained THREE-call ``a * b * c * d`` transcript (T2).

  Three dependent calls -- ``CALC(a*b)`` -> ``CALC(<a*b>*c)`` -> ``CALC(<a*b*c>*d)``
  -- so the warm-up teaches a deeper chain (two intermediate copies forward, up
  to ~6 digits, then a final copy up to ~8 digits).
  """
  a = rng.randint(11, 99)
  b = rng.randint(11, 99)
  c = rng.randint(11, 99)
  d = rng.randint(11, 99)
  ab = a * b
  abc = ab * c
  gold = abc * d
  return [
      (f"Q: {a} * {b} * {c} * {d}\n", 0),
      (f"CALC({a} * {b})\n", 1),
      (f"Tool result: {ab}\n", 0),
      (f"CALC({ab} * {c})\n", 1),
      (f"Tool result: {abc}\n", 0),
      (f"CALC({abc} * {d})\n", 1),
      (f"Tool result: {gold}\n", 0),
      (f"{gold}\n", 1),
  ]


def _encode_segments(
    tokenizer, segments: Segments, max_seq_len: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Tokenizes one transcript into (input_tokens, loss_mask, pad_mask).

  Segments are encoded independently (``add_special_tokens=False``) and
  concatenated so the loss mask lines up exactly with segment boundaries; the
  minor BPE boundary effects between segments are irrelevant for teaching the
  pattern. A leading BOS is prepended (mask 0) and the whole thing is
  right-padded to ``max_seq_len`` with ``pad=eos`` (mask 0, pad_mask 0).
  """
  ids: List[int] = [DELPHI_BOS_ID]
  loss: List[int] = [0]
  for text, train_flag in segments:
    seg_ids = tokenizer.encode(text, add_special_tokens=False)
    ids.extend(seg_ids)
    loss.extend([train_flag] * len(seg_ids))

  ids = ids[:max_seq_len]
  loss = loss[:max_seq_len]
  real_len = len(ids)

  input_tokens = np.full(max_seq_len, DELPHI_EOS_ID, dtype=np.int32)
  loss_mask = np.zeros(max_seq_len, dtype=np.float32)
  pad_mask = np.zeros(max_seq_len, dtype=np.bool_)
  input_tokens[:real_len] = np.asarray(ids, dtype=np.int32)
  loss_mask[:real_len] = np.asarray(loss, dtype=np.float32)
  pad_mask[:real_len] = True
  return input_tokens, loss_mask, pad_mask


class _SFTSource(grain.RandomAccessDataSource):
  """A grain source of pre-tokenized (input_tokens, loss_mask, pad_mask) rows.

  When ``prompt_prefix`` is non-empty it is prepended (loss mask 0) to every
  transcript, so the SFT context matches the RL rollout prompt EXACTLY (the
  rollout prepends the few-shot ``system_prompt``). Without this, SFT trains the
  ``Q -> CALC`` mapping in a bare ``BOS Q:...`` context that does not transfer to
  the RL ``<few-shot> Q:...`` context -- benign for the single-call T0, but for
  the harder chained T1 it actively corrupts turn-1 (the model stops emitting
  ``CALC(`` and just echoes the operands).
  """

  def __init__(
      self,
      tokenizer,
      n: int,
      seed: int,
      max_seq_len: int,
      segment_fn: SegmentFn,
      prompt_prefix: str = "",
  ):
    rng = random.Random(seed)
    prefix_segs: Segments = [(prompt_prefix + "\n", 0)] if prompt_prefix else []
    self._rows = [
        _encode_segments(tokenizer, prefix_segs + segment_fn(rng), max_seq_len)
        for _ in range(n)
    ]

  def __len__(self) -> int:
    return len(self._rows)

  def __getitem__(self, idx: int):
    return self._rows[idx]


def build_sft_dataset(
    tokenizer,
    n: int,
    seed: int,
    batch_size: int,
    max_seq_len: int,
    segment_fn: SegmentFn = t0_segments,
    prompt_prefix: str = "",
) -> grain.MapDataset:
  """Builds a batched grain dataset of CALC SFT transcripts for a stage.

  ``.batch`` collates the 3-tuple rows field-wise into a 3-tuple of stacked
  ``[B, max_seq_len]`` arrays; ``.map`` names them into a dict consumed by
  :func:`sft_model_input_fn`. ``prompt_prefix`` (if set) is prepended masked to
  every transcript to match the RL rollout prompt.
  """
  source = _SFTSource(tokenizer, n, seed, max_seq_len, segment_fn, prompt_prefix)

  def _to_columns(batch):
    input_tokens, loss_mask, pad_mask = batch
    return {
        "input_tokens": input_tokens,
        "loss_mask": loss_mask,
        "pad_mask": pad_mask,
    }

  return grain.MapDataset.source(source).batch(batch_size).map(_to_columns)


def sft_model_input_fn(batch: Dict[str, Any]) -> Dict[str, Any]:
  """Expands a batched SFT row into PeftTrainer ``_default_loss_fn`` kwargs.

  ``input_mask`` is the LOSS mask (which tokens to train on). ``positions`` and
  the ``[B, L, L]`` causal ``attention_mask`` are derived from the separate
  PADDING mask (real tokens vs right-padding), matching the rollout loss path.
  """
  pad_mask = batch["pad_mask"]
  return {
      "input_tokens": batch["input_tokens"],
      "input_mask": batch["loss_mask"],
      "positions": sft_utils.build_positions_from_mask(pad_mask),
      "attention_mask": sft_utils.make_causal_attn_mask(pad_mask),
  }


def run_sft_warmup(
    model,
    tokenizer,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    mesh: jax.sharding.Mesh,
    segment_fn: SegmentFn = t0_segments,
    prompt_prefix: str = "",
    max_seq_len: int = 80,
    seed: int = 0,
) -> Any:
  """SFT-warms ``model`` in place on CALC transcripts, then returns it.

  The same ``nnx`` model object is handed back for the RL phase (no checkpoint).
  ``train()`` is run inside the mesh context so PeftTrainer's ``shard_input``
  shards the data batches across the ``fsdp`` axis (it reads the ambient mesh
  from ``pxla.thread_resources``). The actor MUST already be FSDP-sharded on the
  mesh (PeftTrainer shards the optimizer state to the full mesh); the caller
  arranges that via ``load_delphi(mesh=...)``.

  Args:
    model: the Delphi ``nnx`` actor (fp32), already sharded on ``mesh``.
    tokenizer: the Delphi HF tokenizer (pad=eos).
    steps: number of SFT optimizer steps.
    batch_size: transcripts per step.
    learning_rate: AdamW lr (clipped at global-norm 1.0, as in the RL phase).
    mesh: the device mesh the model is sharded on.
    segment_fn: the per-stage transcript builder (T0 single call vs T1 chained).
    max_seq_len: padded transcript length (T0 ~30 tokens, T1 ~50; 80 covers both).
    seed: PRNG seed for the synthetic problem set.

  Returns:
    The same ``model`` object, now warmed.
  """
  dataset = build_sft_dataset(
      tokenizer,
      n=(steps + 2) * batch_size,
      seed=seed,
      batch_size=batch_size,
      max_seq_len=max_seq_len,
      segment_fn=segment_fn,
      prompt_prefix=prompt_prefix,
  )
  optimizer = optax.chain(
      optax.clip_by_global_norm(1.0),
      optax.adamw(learning_rate=learning_rate, b1=0.9, b2=0.99, weight_decay=0.0),
  )
  trainer = PeftTrainer(
      model=model,
      optimizer=optimizer,
      training_config=TrainingConfig(
          eval_every_n_steps=10**9,  # no eval split
          max_steps=steps,
          metrics_logging_options=None,
      ),
  )
  trainer.with_gen_model_input_fn(sft_model_input_fn)
  print(
      f"[sft] warm-up: steps={steps} bs={batch_size} lr={learning_rate} "
      f"max_seq_len={max_seq_len}",
      flush=True,
  )
  with mesh:
    trainer.train(dataset)
  print("[sft] warm-up complete", flush=True)
  return model
