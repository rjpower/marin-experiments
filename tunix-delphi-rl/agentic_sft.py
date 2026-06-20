"""Supervised warm-up for the Delphi T0 agentic tool stage.

The T0 GRPO run learns the tool CALL and its operands well (``arg_acc`` 0.37 ->
0.99 on TPU) but the final-answer COPY -- emitting the injected tool result
instead of a guessed product -- never takes off: copying a multi-digit number
verbatim after ``"Tool result: X"`` is out-of-distribution for the 447M Delphi
base LM, so it happens too rarely in the rollouts for GRPO to reinforce
(``solve_ratio`` peaks ~0.1 then collapses back to ~0.03 as the policy sharpens
and exploration dies). This is the classic "RL can only amplify what the base
policy already samples" wall.

This module does the standard fix: a short SUPERVISED fine-tune that makes the
call+copy pattern in-distribution BEFORE RL, using tunix's stock
:class:`~tunix.sft.peft_trainer.PeftTrainer` on the SAME in-memory model object
(no checkpoint round-trip) -- the warmed ``actor`` flows straight into the
``RLCluster``. The handoff works because both phases mutate the same ``nnx``
module in place and ``RLCluster`` re-shards as needed.

Each transcript is the clean single-tool episode::

    Q: {a} * {b}
    CALC({a} * {b})
    Tool result: {a*b}
    {a*b}

with a 3-way mask: train (loss mask 1) on the MODEL's turns -- the
``CALC(a * b)`` call and the final answer line -- and NOT (loss mask 0) on the
question or the environment-provided ``Tool result:`` line (the env emits that at
RL time; the model must learn to copy it, not to produce it). ``positions`` and
the causal ``attention_mask`` are built from a separate PADDING mask (real tokens
vs right-padding), exactly as the rollout / RL loss path does it.

The PeftTrainer's default loss is next-token NLL over ``input_mask`` (see
``peft_trainer._default_loss_fn``); the default ``gen_model_input_fn`` is the
identity, so we install :func:`sft_model_input_fn` to expand each batched
``{input_tokens, loss_mask, pad_mask}`` row into the loss-fn kwargs
``{input_tokens, input_mask, positions, attention_mask}``.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

import grain.python as grain
import jax
import numpy as np
import optax

from tunix.sft import utils as sft_utils
from tunix.sft.peft_trainer import PeftTrainer, TrainingConfig

from delphi_qwen3 import DELPHI_BOS_ID, DELPHI_EOS_ID

# Each transcript segment is (text, train_on_it?). Mask 1 == the model's own
# turns (the tool call + the copied answer); mask 0 == context the model
# conditions on but must not be trained to emit (the question, the tool result).
_SFT_SEGMENTS = (
    ("Q: {a} * {b}\n", 0),
    ("CALC({a} * {b})\n", 1),
    ("Tool result: {gold}\n", 0),
    ("{gold}\n", 1),
)


def _encode_sft_example(
    tokenizer, a: int, b: int, max_seq_len: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Tokenizes one CALC transcript into (input_tokens, loss_mask, pad_mask).

  Segments are encoded independently (``add_special_tokens=False``) and
  concatenated so the loss mask lines up exactly with segment boundaries; the
  minor BPE boundary effects between segments are irrelevant for teaching the
  pattern. A leading BOS is prepended (mask 0) and the whole thing is
  right-padded to ``max_seq_len`` with ``pad=eos`` (mask 0, pad_mask 0).

  Returns:
    ``(input_tokens, loss_mask, pad_mask)`` numpy arrays of shape
    ``[max_seq_len]`` with dtypes int32 / float32 / bool.
  """
  gold = a * b
  ids: List[int] = [DELPHI_BOS_ID]
  loss: List[int] = [0]
  for template, train_flag in _SFT_SEGMENTS:
    text = template.format(a=a, b=b, gold=gold)
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
  """A grain source of pre-tokenized (input_tokens, loss_mask, pad_mask) rows."""

  def __init__(self, tokenizer, n: int, seed: int, max_seq_len: int):
    rng = random.Random(seed)
    self._rows = []
    for _ in range(n):
      a = rng.randint(11, 99)
      b = rng.randint(11, 99)
      self._rows.append(_encode_sft_example(tokenizer, a, b, max_seq_len))

  def __len__(self) -> int:
    return len(self._rows)

  def __getitem__(self, idx: int):
    return self._rows[idx]


def build_sft_dataset(
    tokenizer, n: int, seed: int, batch_size: int, max_seq_len: int
) -> grain.MapDataset:
  """Builds a batched grain dataset of CALC SFT transcripts.

  ``.batch`` collates the 3-tuple rows field-wise into a 3-tuple of stacked
  ``[B, max_seq_len]`` arrays; ``.map`` names them into a dict consumed by
  :func:`sft_model_input_fn`.
  """
  source = _SFTSource(tokenizer, n, seed, max_seq_len)

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
    max_seq_len: int = 64,
    seed: int = 0,
) -> Any:
  """SFT-warms ``model`` in place on CALC transcripts, then returns it.

  The same ``nnx`` model object is handed back for the RL phase (no checkpoint).
  ``train()`` is run inside the mesh context so PeftTrainer's ``shard_input``
  shards the data batches across the ``fsdp`` axis (it reads the ambient mesh
  from ``pxla.thread_resources``).

  Args:
    model: the Delphi ``nnx`` actor (fp32) to fine-tune in place.
    tokenizer: the Delphi HF tokenizer (pad=eos).
    steps: number of SFT optimizer steps.
    batch_size: transcripts per step.
    learning_rate: AdamW lr (clipped at global-norm 1.0, as in the RL phase).
    mesh: the device mesh the model is sharded on.
    max_seq_len: padded transcript length (a CALC episode is ~30 tokens).
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
