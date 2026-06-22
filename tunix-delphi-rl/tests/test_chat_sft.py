"""Offline tests for the chat-SFT formatter/encoder/source (no TPU / network).

Uses a tiny char-level fake tokenizer so the loss-mask layout is exactly
predictable, and monkeypatches the dataset stream so :class:`_ChatSFTSource`
is exercised without HuggingFace.
"""

import numpy as np

from training import chat_sft
from training.chat_sft import (
    ASSISTANT_HEADER,
    build_chat_sft_dataset,
    encode_chat_messages,
    format_user_prompt,
)


class FakeTok:
  """Char-level tokenizer: one token id per character (>=3); bos=1, eos=2."""

  bos_token_id = 1
  eos_token_id = 2

  def encode(self, text, add_special_tokens=False):  # noqa: D401
    return [ord(c) % 1000 + 3 for c in text]


TOK = FakeTok()


def test_user_assistant_mask_layout():
  msgs = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
  enc = encode_chat_messages(TOK, msgs, 64, bos_id=1, eos_id=2)
  assert enc is not None
  input_tokens, loss_mask, pad_mask = enc
  # real length = bos + "<|user|>\n"(9) + "hello\n"(6) + "<|assistant|>\n"(14) + "hi"(2) + eos(1)
  real_len = 1 + 9 + 6 + 14 + 2 + 1
  assert pad_mask.sum() == real_len
  assert input_tokens[0] == 1  # bos
  # only the assistant content (2) + eos (1) are trained
  assert loss_mask.sum() == 3.0
  assert loss_mask[real_len - 1] == 1.0  # eos trained
  assert loss_mask[real_len - 2] == 1.0  # last content char trained
  assert loss_mask[real_len - 3 - 14] == 0.0  # inside the user turn: context
  assert input_tokens[real_len - 1] == 2  # eos id at the trained terminator


def test_no_bos_when_none():
  msgs = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
  enc = encode_chat_messages(TOK, msgs, 64, bos_id=None, eos_id=2)
  assert enc is not None
  input_tokens, _, _ = enc
  # first token is the '<' of "<|user|>\n", not a bos
  assert input_tokens[0] == ord("<") % 1000 + 3


def test_empty_assistant_drops_to_none():
  msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": ""}]
  assert encode_chat_messages(TOK, msgs, 64, bos_id=1, eos_id=2) is None


def test_truncated_away_assistant_drops_to_none():
  # max_seq_len too small to reach any assistant content -> no signal -> None.
  msgs = [{"role": "user", "content": "x" * 50}, {"role": "assistant", "content": "ans"}]
  assert encode_chat_messages(TOK, msgs, 8, bos_id=1, eos_id=2) is None


def test_padding_fill_is_eos_and_padmask_false():
  msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
  input_tokens, loss_mask, pad_mask = encode_chat_messages(TOK, msgs, 64, bos_id=1, eos_id=2)
  real_len = int(pad_mask.sum())
  assert np.all(input_tokens[real_len:] == 2)  # eos pad fill
  assert np.all(~pad_mask[real_len:])
  assert np.all(loss_mask[real_len:] == 0.0)


def test_format_user_prompt_ends_with_assistant_header():
  p = format_user_prompt("do a thing", system_text="be brief")
  assert p.endswith(ASSISTANT_HEADER)
  assert "<|system|>\nbe brief\n" in p
  assert "<|user|>\ndo a thing\n" in p


def test_source_streams_collects_and_cycles(monkeypatch):
  # 3 usable + 1 empty (dropped); request n=8 -> cycles the 3 usable up to 8.
  fake_rows = [
      {"messages": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]},
      {"messages": [{"role": "user", "content": "q2"}, {"role": "assistant", "content": ""}]},  # dropped
      {"messages": [{"role": "user", "content": "q3"}, {"role": "assistant", "content": "a3"}]},
      {"messages": [{"role": "user", "content": "q4"}, {"role": "assistant", "content": "a4"}]},
  ]

  def fake_stream(name, *, limit=None, split=None):
    assert name == "fake/ds"
    return iter(fake_rows)

  monkeypatch.setattr(chat_sft, "load_instruction_messages", fake_stream, raising=False)
  # patch the symbol the source imports lazily
  import sft_data.instruction_datasets as ids_mod
  monkeypatch.setattr(ids_mod, "load_instruction_messages", fake_stream)

  ds = build_chat_sft_dataset(
      TOK, "fake/ds", n=8, seed=0, batch_size=4, max_seq_len=64, bos_id=1, eos_id=2,
  )
  batches = list(ds)
  assert len(batches) == 2  # 8 rows / batch 4
  for b in batches:
    assert b["input_tokens"].shape == (4, 64)
    assert b["loss_mask"].shape == (4, 64)
    assert b["pad_mask"].shape == (4, 64)
    assert b["loss_mask"].sum() > 0  # every row carries assistant signal
