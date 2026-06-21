# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for :mod:`sft_data.instruction_datasets`.

The adapter transforms run on small synthetic rows (no network) for each
``InputDatasetFormat`` the loader supports, plus registry sanity (tulu keys
present, revisions pinned). A real HF streaming smoke check lives behind the
``slow`` marker so the default suite stays offline.
"""

from __future__ import annotations

import re

import pytest

from sft_data.instruction_datasets import (
    INSTRUCTION_DATASET_NAME_TO_CONFIG,
    TULU_DATASETS,
    InputDatasetFormat,
    instruct_column_response_adapter,
    instruct_msg_response_adapter,
    instruction_response_adapter,
    load_instruction_messages,
    multi_turn_adapter,
    transform_row,
)

# A pinned revision is a short (>=7) lowercase hex commit prefix (or full hash).
_HASH_RE = re.compile(r"^[0-9a-f]{7,40}$")


def _cfg(adapter, metadata_columns=None):
  """A throwaway config wrapping *adapter* for transform_row tests."""
  from sft_data.instruction_datasets import InstructionDatasetConfig

  return InstructionDatasetConfig(
      hf_dataset_id="synthetic/test",
      revision="0000000",
      adapter=adapter,
      metadata_columns=metadata_columns or [],
  )


# ---------------------------------------------------------------------------
# Adapter format transforms (offline, synthetic rows)
# ---------------------------------------------------------------------------


def test_multi_turn_default_roles():
  # tulu shape: a `messages` column of {role, content} dicts passes through with
  # roles normalized to the OpenAI set.
  adapter = multi_turn_adapter()
  row = {
      "messages": [
          {"role": "system", "content": "be nice"},
          {"role": "user", "content": "hi"},
          {"role": "assistant", "content": "hello"},
      ],
      "dataset": "oasst1",
      "id": "abc",
  }
  out = transform_row(row, _cfg(adapter, ["dataset", "id"]))
  assert out["messages"] == [
      {"role": "system", "content": "be nice"},
      {"role": "user", "content": "hi"},
      {"role": "assistant", "content": "hello"},
  ]
  assert out["metadata"] == {"dataset": "oasst1", "id": "abc"}


def test_multi_turn_role_value_remap():
  # OpenHermes shape: from/value keys with human/gpt role values.
  adapter = multi_turn_adapter(
      conversation_column="conversations",
      role_key="from",
      user_value="human",
      assistant_value="gpt",
      content_key="value",
  )
  row = {
      "conversations": [
          {"from": "human", "value": "2+2?"},
          {"from": "gpt", "value": "4"},
      ]
  }
  out = transform_row(row, _cfg(adapter))
  assert [m["role"] for m in out["messages"]] == ["user", "assistant"]
  assert [m["content"] for m in out["messages"]] == ["2+2?", "4"]


def test_instruction_response():
  adapter = instruction_response_adapter(instruction_column="query", response_column="response")
  out = transform_row({"query": "cap of France?", "response": "Paris"}, _cfg(adapter))
  assert out["messages"] == [
      {"role": "user", "content": "cap of France?"},
      {"role": "assistant", "content": "Paris"},
  ]


def test_instruction_response_drops_missing():
  adapter = instruction_response_adapter(instruction_column="q", response_column="r")
  assert transform_row({"q": "x", "r": None}, _cfg(adapter)) is None
  assert transform_row({"q": None, "r": "y"}, _cfg(adapter)) is None


def test_instruction_response_filter_on_key():
  # AceCode shape: response is a list; pick the best by a numeric metric, then
  # extract content_key from the winner.
  adapter = instruction_response_adapter(
      instruction_column="question",
      response_column="inferences",
      content_key="completion",
      filter_on_key="pass_rate",
  )
  row = {
      "question": "write add()",
      "inferences": [
          {"completion": "bad", "pass_rate": 0.1},
          {"completion": "good", "pass_rate": 0.9},
      ],
  }
  out = transform_row(row, _cfg(adapter))
  assert out["messages"][1] == {"role": "assistant", "content": "good"}


def test_instruct_column_response():
  # natural_reasoning shape: responses is a list of dicts; take responses[0][content_key].
  adapter = instruct_column_response_adapter(
      instruction_column="question",
      response_column="responses",
      content_key="response",
  )
  row = {"question": "speed?", "responses": [{"response_model": "M", "response": "125 kmph"}]}
  out = transform_row(row, _cfg(adapter))
  assert out["messages"] == [
      {"role": "user", "content": "speed?"},
      {"role": "assistant", "content": "125 kmph"},
  ]


def test_instruct_msg_response_and_drops():
  # dolphin-r1-reasoning shape: instruction is a 1-msg list, response is a string.
  adapter = instruct_msg_response_adapter(instruction_column="messages", response_column="answer")
  row = {"messages": [{"role": "user", "content": "q?"}], "answer": "a."}
  out = transform_row(row, _cfg(adapter))
  assert out["messages"] == [
      {"role": "user", "content": "q?"},
      {"role": "assistant", "content": "a."},
  ]
  # Dropped: >1 instruction message, missing role_key, or null response.
  multi = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], "answer": "a"}
  assert transform_row(multi, _cfg(adapter)) is None
  no_role = {"messages": [{"content": "u"}], "answer": "a"}
  assert transform_row(no_role, _cfg(adapter)) is None
  null_resp = {"messages": [{"role": "user", "content": "u"}], "answer": None}
  assert transform_row(null_resp, _cfg(adapter)) is None


def test_think_replacements_applied():
  # Default replacements rewrite <think>/</think> to marin's sentinel tokens.
  adapter = multi_turn_adapter()
  row = {"messages": [{"role": "assistant", "content": "<think>hmm</think> done"}]}
  out = transform_row(row, _cfg(adapter))
  assert out["messages"][0]["content"] == "<|start_think|>hmm<|end_think|> done"


def test_replacements_can_be_disabled():
  adapter = multi_turn_adapter(replacements={})
  row = {"messages": [{"role": "assistant", "content": "<think>x</think>"}]}
  out = transform_row(row, _cfg(adapter))
  assert out["messages"][0]["content"] == "<think>x</think>"


def test_extra_metadata_fn():
  adapter = multi_turn_adapter(extra_metadata_fn=lambda r: {"flag": r.get("k") == 1})
  row = {"messages": [{"role": "user", "content": "x"}], "k": 1}
  out = transform_row(row, _cfg(adapter, ["k"]))
  assert out["metadata"] == {"k": 1, "flag": True}


# ---------------------------------------------------------------------------
# Registry sanity
# ---------------------------------------------------------------------------


def test_tulu_datasets_present_and_pinned():
  assert TULU_DATASETS, "TULU_DATASETS must be non-empty"
  assert "allenai/tulu-3-sft-mixture" in TULU_DATASETS
  for key in TULU_DATASETS:
    assert key in INSTRUCTION_DATASET_NAME_TO_CONFIG, f"{key} missing from registry"
    cfg = INSTRUCTION_DATASET_NAME_TO_CONFIG[key]
    assert _HASH_RE.match(cfg.revision), f"{key} revision {cfg.revision!r} is not a pinned hash"


def test_known_tulu_revisions_unchanged():
  # Guard the pinned revisions carried over from marin.
  expected = {
      "allenai/tulu-3-sft-mixture": "55e9fd6",
      "allenai/tulu-v2-sft-mixture": "6248b17",
      "allenai/tulu-v2-sft-mixture-olmo-4096": "7a7c388",
      "sherryy/tulu-3-sft-personas-instruction-following-expanded": "79ab2c4",
  }
  for key, rev in expected.items():
    assert INSTRUCTION_DATASET_NAME_TO_CONFIG[key].revision == rev


def test_tulu3_uses_multi_turn_format():
  cfg = INSTRUCTION_DATASET_NAME_TO_CONFIG["allenai/tulu-3-sft-mixture"]
  assert cfg.adapter.dataset_format == InputDatasetFormat.SINGLE_COLUMN_MULTI_TURN


def test_unknown_dataset_raises():
  with pytest.raises(KeyError):
    next(load_instruction_messages("not/a-dataset", limit=1))


# ---------------------------------------------------------------------------
# Network smoke (opt-in; excluded by default via the `slow` marker)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_stream_tulu3_smoke():
  from sft_data.instruction_datasets import load_tulu_sft

  examples = list(load_tulu_sft(limit=5))
  assert len(examples) == 5
  for ex in examples:
    assert ex["messages"], "expected non-empty messages"
    for m in ex["messages"]:
      assert m["role"] in {"system", "user", "assistant", "tool"}
      assert isinstance(m["content"], str) and m["content"].strip()
