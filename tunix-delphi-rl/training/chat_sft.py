"""Conversational chat-SFT: teach a BASE LM a turn-taking instruction format.

Stage 1 of the Delphi-1.9B vs Qwen3-1.7B agentic post-training comparison. Both
targets are *base* LMs with no chat template, so we impose ONE model-agnostic
plain-text ChatML-style format (the literal markers ``<|user|>`` / ``<|assistant|>``
tokenize fine in both the Llama-3 and Qwen vocabularies) and SFT on it identically
for both models, so the only live variable downstream is the base model itself.

Reuses the generic SFT plumbing from :mod:`training.agentic_sft`
(:func:`sft_model_input_fn`, the grain 3-tuple row layout, ``PeftTrainer``); the
only new pieces are:
  * a model-agnostic chat encoder (BOS/EOS taken from the tokenizer, NOT the
    Delphi-hardcoded ids the CALC warm-up uses), and
  * a *streaming* source over :func:`sft_data.instruction_datasets.load_instruction_messages`
    (real tulu conversations) instead of a synthetic ``segment_fn``.

Loss mask: train (1) on the ASSISTANT content + its terminating EOS only; the
role headers and the system/user/tool turns are context (0). At eval time we
prompt with the same format up to ``<|assistant|>\\n`` and let the model generate
its turn -- so the header is part of the prompt (mask 0) and the model learns to
produce the content and then STOP (emit EOS).
"""

from __future__ import annotations

import random
from typing import Any, Callable, Iterable

import grain.python as grain
import jax
import numpy as np

from tunix.sft.peft_trainer import PeftTrainer, TrainingConfig

from training.agentic_common import clipped_adamw
from training.agentic_sft import sft_model_input_fn

# One plain-text chat format, shared by every model (base LMs, no chat template).
CHAT_ROLE_HEADER = {
    "system": "<|system|>\n",
    "user": "<|user|>\n",
    "assistant": "<|assistant|>\n",
    "tool": "<|tool|>\n",
}
ASSISTANT_HEADER = CHAT_ROLE_HEADER["assistant"]


def format_user_prompt(user_text: str, *, system_text: str | None = None) -> str:
  """Builds an eval prompt ending at ``<|assistant|>\\n`` (model generates next).

  Mirrors the training format exactly for the leading (context) turns, so the
  SFT'd model sees a familiar prefix and produces its assistant turn.

  Args:
    user_text: the user instruction.
    system_text: optional system message to prepend.

  Returns:
    The formatted prompt string.
  """
  parts: list[str] = []
  if system_text:
    parts.append(CHAT_ROLE_HEADER["system"] + system_text + "\n")
  parts.append(CHAT_ROLE_HEADER["user"] + user_text + "\n")
  parts.append(ASSISTANT_HEADER)
  return "".join(parts)


def encode_chat_messages(
    tokenizer,
    messages: list[dict[str, Any]],
    max_seq_len: int,
    *,
    bos_id: int | None,
    eos_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
  """Tokenizes one conversation into (input_tokens, loss_mask, pad_mask).

  The whole conversation is laid out as ``[BOS?] <header><content> ...`` with the
  loss mask set to 1 only over assistant content + its terminating EOS. Headers
  and non-assistant content are masked 0. Right-padded to ``max_seq_len`` with
  ``eos`` (pad_mask 0). Headers/content are encoded with ``add_special_tokens=
  False`` so the mask lines up with the segment boundaries.

  Args:
    tokenizer: an HF tokenizer (Delphi Llama-3 or Qwen3); ``bos_id``/``eos_id``
      are passed in so this stays model-agnostic.
    messages: OpenAI-format ``[{"role", "content"}, ...]``.
    max_seq_len: padded length; rows longer than this are truncated.
    bos_id: leading BOS token id, or ``None`` to omit (Qwen base has no BOS).
    eos_id: EOS token id (also the right-pad fill).

  Returns:
    The ``(input_tokens, loss_mask, pad_mask)`` triple, or ``None`` if no
    assistant content survived (empty or truncated away) -- such a row carries
    no training signal and is dropped by the source.
  """
  ids: list[int] = []
  loss: list[int] = []
  if bos_id is not None:
    ids.append(int(bos_id))
    loss.append(0)

  for msg in messages:
    role = msg.get("role", "user")
    content = msg.get("content") or ""
    header = CHAT_ROLE_HEADER.get(role, CHAT_ROLE_HEADER["user"])
    htoks = tokenizer.encode(header, add_special_tokens=False)
    ids.extend(htoks)
    loss.extend([0] * len(htoks))
    if role == "assistant":
      ctoks = tokenizer.encode(content, add_special_tokens=False)
      if ctoks:  # non-empty assistant turn: train on content + terminating EOS
        ids.extend(ctoks)
        loss.extend([1] * len(ctoks))
        ids.append(int(eos_id))
        loss.append(1)
      # empty assistant turn contributes its header as context only (no signal)
    else:
      ctoks = tokenizer.encode(content + "\n", add_special_tokens=False)
      ids.extend(ctoks)
      loss.extend([0] * len(ctoks))

  ids = ids[:max_seq_len]
  loss = loss[:max_seq_len]
  if 1 not in loss:  # no assistant token survived -> no training signal
    return None

  real_len = len(ids)
  input_tokens = np.full(max_seq_len, int(eos_id), dtype=np.int32)
  loss_mask = np.zeros(max_seq_len, dtype=np.float32)
  pad_mask = np.zeros(max_seq_len, dtype=np.bool_)
  input_tokens[:real_len] = np.asarray(ids, dtype=np.int32)
  loss_mask[:real_len] = np.asarray(loss, dtype=np.float32)
  pad_mask[:real_len] = True
  return input_tokens, loss_mask, pad_mask


class _ChatSFTSource(grain.RandomAccessDataSource):
  """A grain source of pre-tokenized chat rows streamed from an instruction set.

  Pulls conversations from :func:`load_instruction_messages` (HF streaming at a
  pinned revision), encodes each into the (input_tokens, loss_mask, pad_mask)
  layout :func:`sft_model_input_fn` consumes, and keeps the first ``n`` that fit.
  Rows are shuffled with ``seed`` (a streamed mixture is only roughly mixed). If
  the stream runs dry before ``n`` usable rows, what was collected is cycled to
  fill (logged), so the trainer still gets ``n`` rows.
  """

  def __init__(
      self,
      tokenizer,
      dataset_name: str,
      n: int,
      seed: int,
      max_seq_len: int,
      *,
      bos_id: int | None,
      eos_id: int,
      split: str | None = None,
      scan_cap: int | None = None,
      stream_fn: Callable[[], Iterable[dict]] | None = None,
  ):
    if stream_fn is None:
      from sft_data.instruction_datasets import load_instruction_messages

      stream_fn = lambda: load_instruction_messages(  # noqa: E731
          dataset_name, limit=None, split=split
      )

    cap = scan_cap if scan_cap is not None else max(n * 20, 2000)
    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    scanned = 0
    for ex in stream_fn():
      scanned += 1
      enc = encode_chat_messages(
          tokenizer, ex.get("messages", []), max_seq_len,
          bos_id=bos_id, eos_id=eos_id,
      )
      if enc is not None:
        rows.append(enc)
        if len(rows) >= n:
          break
      if scanned >= cap:
        break

    if not rows:
      raise ValueError(
          f"No usable chat examples from {dataset_name!r} after scanning "
          f"{scanned} rows (all empty or longer than max_seq_len={max_seq_len})."
      )
    random.Random(seed).shuffle(rows)
    if len(rows) < n:
      print(
          f"[chat-sft] only {len(rows)}/{n} usable rows from {dataset_name!r} "
          f"(scanned {scanned}); cycling to fill.",
          flush=True,
      )
      base = list(rows)
      while len(rows) < n:
        rows.append(base[len(rows) % len(base)])
    self._rows = rows[:n]

  def __len__(self) -> int:
    return len(self._rows)

  def __getitem__(self, idx: int):
    return self._rows[idx]


def _to_columns(batch):
  input_tokens, loss_mask, pad_mask = batch
  return {
      "input_tokens": input_tokens,
      "loss_mask": loss_mask,
      "pad_mask": pad_mask,
  }


def build_chat_sft_dataset(
    tokenizer,
    dataset_name: str,
    n: int,
    seed: int,
    batch_size: int,
    max_seq_len: int,
    *,
    bos_id: int | None,
    eos_id: int,
    split: str | None = None,
    stream_fn: Callable[[], Iterable[dict]] | None = None,
) -> grain.MapDataset:
  """Batched grain dataset of chat-SFT rows (same column layout as the CALC SFT).

  ``stream_fn`` (a zero-arg callable yielding ``{"messages": [...]}`` rows)
  overrides the default per-``dataset_name`` HF stream -- the caller passes one to
  feed a custom mixture (e.g. the tulu+tool-use "up to shape" stream).
  """
  source = _ChatSFTSource(
      tokenizer, dataset_name, n, seed, max_seq_len,
      bos_id=bos_id, eos_id=eos_id, split=split, stream_fn=stream_fn,
  )
  return grain.MapDataset.source(source).batch(batch_size).map(_to_columns)


def run_chat_sft(
    model,
    tokenizer,
    *,
    dataset_name: str,
    steps: int,
    batch_size: int,
    learning_rate: float,
    mesh: jax.sharding.Mesh,
    max_seq_len: int = 2048,
    seed: int = 0,
    split: str | None = None,
    stream_fn: Callable[[], Iterable[dict]] | None = None,
) -> Any:
  """Chat-SFTs ``model`` in place on ``dataset_name`` conversations, returns it.

  Mirrors :func:`training.agentic_sft.run_sft_warmup` (same ``PeftTrainer`` /
  ``sft_model_input_fn`` / clipped-AdamW path, same in-place ``nnx`` handoff) but
  draws rows from a streamed instruction set and uses the tokenizer's own
  BOS/EOS so it is model-agnostic. The actor must already be FSDP-sharded on
  ``mesh`` (the caller arranges that via the loader's ``mesh=`` argument).

  Args:
    model: the base ``nnx`` actor (fp32), already sharded on ``mesh``.
    tokenizer: the model's HF tokenizer (pad=eos).
    dataset_name: an instruction-set key (e.g. ``allenai/tulu-3-sft-mixture``).
    steps: number of SFT optimizer steps.
    batch_size: conversations per step.
    learning_rate: AdamW lr (clipped at global-norm 1.0, as in RL).
    mesh: the device mesh the model is sharded on.
    max_seq_len: padded conversation length; longer rows are truncated/dropped.
    seed: PRNG seed for row shuffling.
    split: optional dataset split override.

  Returns:
    The same ``model`` object, now chat-warmed.
  """
  bos_id = tokenizer.bos_token_id
  eos_id = tokenizer.eos_token_id
  n = (steps + 2) * batch_size
  dataset = build_chat_sft_dataset(
      tokenizer, dataset_name, n, seed, batch_size, max_seq_len,
      bos_id=bos_id, eos_id=eos_id, split=split, stream_fn=stream_fn,
  )
  optimizer = clipped_adamw(learning_rate)
  trainer = PeftTrainer(
      model=model,
      optimizer=optimizer,
      training_config=TrainingConfig(
          eval_every_n_steps=10**9,
          max_steps=steps,
          metrics_logging_options=None,
      ),
  )
  trainer.with_gen_model_input_fn(sft_model_input_fn)
  print(
      f"[chat-sft] dataset={dataset_name} steps={steps} bs={batch_size} "
      f"lr={learning_rate} max_seq_len={max_seq_len} bos={bos_id} eos={eos_id}",
      flush=True,
  )
  with mesh:
    trainer.train(dataset)
  print("[chat-sft] complete", flush=True)
  return model
