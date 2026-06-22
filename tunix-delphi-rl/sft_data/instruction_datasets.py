# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Self-contained instruction / chat-SFT dataset loader (no ``marin.*`` runtime dep).

This module loads instruction datasets (tulu and friends) from HuggingFace and
normalizes each row into **OpenAI chat messages** — ``[{"role": ..., "content":
...}, ...]`` with roles drawn from ``{"user", "assistant", "system", "tool"}`` —
so they can feed a conversational chat-SFT training stage. It is a slimmed,
**dependency-free** reimplementation of marin's
``experiments/instruction_datasets.py`` + ``marin/transform/conversation/`` adapter
logic. The reference original is kept beside it as
``instruction_datasets_marin_original.py``; only the *registry* and the *row →
messages* transform semantics are reproduced here. All of marin's executor
plumbing (``ExecutorStep``, ``versioned``, Dolma conversion, GCS-pinned
``tokenized/...`` outputs) is dropped.

Where does the data come from at runtime?
-----------------------------------------
**HuggingFace streaming at the pinned revision** (``streaming=True`` + ``.take``),
applying our reimplemented adapter on the fly. We deliberately do *not* read a
pre-materialized marin copy, because there is no reliably-reachable one:

  * The messages-format intermediates marin produced
    (``gs://marin-us-central2/dolma/tulu_3_in_dolma-*/``,
    ``documents/allenai--tulu-3-sft-mixture-*``) now contain only ``.SUCCESS`` /
    executor markers — the actual ``*.jsonl.gz`` shards have been garbage
    collected.
  * The only surviving materialization is the **Llama-3-tokenized levanter cache**
    (``gs://marin-us-central2/tokenized/tulu_sft-*``) — that is token IDs in a
    flattened single-string-per-doc layout, not chat messages, so it is the wrong
    shape for this loader.
  * Those ``gs://marin-*`` buckets are **not world/anonymously readable** (they
    require GCS creds the iris worker is not guaranteed to have for arbitrary
    marin buckets).

Streaming the raw HF dataset at the pinned revision is therefore both the robust
default and the only shape-correct source. The iris worker already has
``datasets``/``huggingface_hub`` in its venv and an ``HF_TOKEN`` in env, which is
all this needs.

Notes for an iris run
---------------------
  * Requires network egress to ``huggingface.co`` and ``HF_TOKEN`` in env (some
    tulu mixtures are gated/large; the token is read by ``datasets`` automatically).
  * Streaming means the **first** ``next()`` pays a metadata + first-shard
    latency (typically a few seconds); there is no multi-GB up-front download.
  * Be gentle with very high parallelism against HF to avoid 429 rate limits
    (the marin original capped at ``max_parallelism=32`` for the same reason);
    this loader is single-stream, so that is rarely an issue.

Public API
----------
  * :func:`load_instruction_messages` — stream a named dataset as
    ``{"messages": [...], "metadata": {...}}`` dicts.
  * :data:`TULU_DATASETS` — the wired tulu dataset keys.
  * :data:`TOOLUSE_DATASETS` — the smoltalk2 tool-calling SFT split keys.
  * :func:`load_tulu_sft` — convenience over ``allenai/tulu-3-sft-mixture``.
  * :func:`load_up_to_shape_mixture` — weighted interleave of tulu-3 chat + the
    tool-use splits (mostly chat with a small tool-use slice).
  * :data:`INSTRUCTION_DATASET_NAME_TO_CONFIG` — name → config registry; adding a
    dataset stays a one-entry change.

Example::

    from sft_data.instruction_datasets import load_tulu_sft
    for ex in load_tulu_sft(limit=5):
        print(ex["messages"][0]["role"], ex["messages"][0]["content"][:40])
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# OpenAI roles we normalize to. The base LM has no chat template, but downstream
# SFT code renders these roles into raw text, so we keep the canonical set.
OPENAI_ROLES = ("system", "user", "assistant", "tool")

# Mirrors marin's transform_conversation.DEFAULT_TEXT_REPLACEMENTS: when an
# adapter does not override ``replacements``, ``<think>``/``</think>`` are mapped
# to the sentinel tokens marin's tokenizer expects. We keep the exact same
# default so streamed text matches what marin would have written.
DEFAULT_TEXT_REPLACEMENTS = {"<think>": "<|start_think|>", "</think>": "<|end_think|>"}


class InputDatasetFormat(str, Enum):
  """Row layout of a source dataset (reimplemented from marin's adapters).

  SINGLE_COLUMN_MULTI_TURN
      One column holds a list of message dicts, e.g.
      ``[{"role": "user", "content": "..."}, {"role": "assistant", ...}]``.
      Role/content key names and role *values* are remappable (OpenHermes uses
      ``from``/``value`` with ``human``/``gpt``).

  INSTRUCTION_RESPONSE
      Two scalar columns (instruction, response) → one user + one assistant turn.
      Optionally ``filter_on_key`` selects the best entry from a list-valued
      response by a numeric metric (AceCode's ``pass_rate``), then extracts
      ``content_key`` from it.

  INSTRUCT_COLUMN_RESPONSE
      ``instruction`` is a scalar; ``response`` is a list whose first element is
      a dict — extract ``content_key`` from ``responses[0]`` (natural_reasoning).

  INSTRUCT_MSG_RESPONSE
      ``instruction`` is a single-message list and ``response`` is a string
      (dolphin-r1-reasoning). Rows with >1 instruction message, a missing
      ``role_key``, or a null response are dropped.
  """

  SINGLE_COLUMN_MULTI_TURN = "messages"
  INSTRUCTION_RESPONSE = "instruction_response"
  INSTRUCT_COLUMN_RESPONSE = "instruct_column_response"
  INSTRUCT_MSG_RESPONSE = "instruct_msg_response"


@dataclass
class TransformAdapter:
  """Spec for turning one raw row into OpenAI messages.

  Reimplements ``marin.transform.conversation.adapters.TransformAdapter`` (only
  the row→messages transform; no executor signature/hash logic). Fields not
  relevant to a given ``dataset_format`` are ignored.
  """

  dataset_format: InputDatasetFormat = InputDatasetFormat.INSTRUCTION_RESPONSE

  # INSTRUCTION_RESPONSE / INSTRUCT_*_RESPONSE
  instruction_column: str = ""
  response_column: str = ""

  # SINGLE_COLUMN_MULTI_TURN (and role remapping for INSTRUCT_MSG_RESPONSE)
  conversation_column: str = "messages"
  role_key: str = "role"
  user_value: str = "user"
  assistant_value: str = "assistant"
  system_value: str = "system"
  content_key: str = "content"
  tool_value: str = "tool"

  # Pick the best entry from a list-valued response by this numeric key.
  filter_on_key: str = ""
  metadata_remap: dict[str, str] = field(default_factory=dict)
  replacements: dict[str, str] | None = None
  extra_metadata_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None

  def transform_conversation_to_openai_format(
      self, row: dict[str, Any]
  ) -> list[dict[str, str]] | None:
    """Convert a raw *row* into OpenAI messages, or ``None`` to drop the row.

    Faithful to marin's semantics: ``None`` means "skip this row" (missing data
    or a shape this adapter intentionally does not handle); callers must drop it
    rather than emit an empty conversation.
    """
    fmt = self.dataset_format

    if fmt == InputDatasetFormat.INSTRUCTION_RESPONSE:
      instruction = row[self.instruction_column]
      response = row[self.response_column]
      if instruction is None or response is None:
        return None
      if self.filter_on_key:
        best_completion = None
        best_metric = -float("inf")
        for completion in response:
          if completion[self.filter_on_key] > best_metric:
            best_metric = completion[self.filter_on_key]
            best_completion = completion
        assert best_completion is not None, "filter_on_key requires a non-empty response list"
        response = best_completion[self.content_key]
      return [
          {"role": "user", "content": instruction},
          {"role": "assistant", "content": response},
      ]

    if fmt == InputDatasetFormat.SINGLE_COLUMN_MULTI_TURN:
      role_to_openai_role = {
          self.user_value: "user",
          self.assistant_value: "assistant",
          self.system_value: "system",
          self.tool_value: "tool",
      }
      conversation = row[self.conversation_column]
      messages: list[dict[str, str]] = []
      for conv in conversation:
        role = role_to_openai_role[conv[self.role_key]]
        messages.append({"role": role, "content": conv[self.content_key]})
      return messages

    if fmt == InputDatasetFormat.INSTRUCT_COLUMN_RESPONSE:
      instruction = row[self.instruction_column]
      responses = row[self.response_column]
      # First (and only) response entry is a dict; pull content_key from it.
      response_dict = responses[0]
      response_content = response_dict[self.content_key]
      return [
          {"role": "user", "content": instruction},
          {"role": "assistant", "content": response_content},
      ]

    if fmt == InputDatasetFormat.INSTRUCT_MSG_RESPONSE:
      instruction = row[self.instruction_column]  # list of dicts
      responses = row[self.response_column]  # single string
      if responses is None or len(instruction) > 1 or self.role_key not in instruction[0]:
        # Drop rows with >1 instruction message or where the instruction lives
        # in a system prompt (e.g. dolphin-r1 reasoning). None, not [].
        return None
      instruction_content = instruction[0][self.content_key]
      return [
          {"role": "user", "content": instruction_content},
          {"role": "assistant", "content": responses},
      ]

    raise ValueError(f"Invalid dataset format: {fmt}")


def _apply_replacements(text: str, replacements: dict[str, str]) -> str:
  updated = text
  for old, new in replacements.items():
    updated = updated.replace(old, new)
  return updated


# ---------------------------------------------------------------------------
# Adapter constructors (mirror the helper factories in the marin original)
# ---------------------------------------------------------------------------


def multi_turn_adapter(
    conversation_column: str = "messages",
    role_key: str = "role",
    user_value: str = "user",
    assistant_value: str = "assistant",
    system_value: str = "system",
    content_key: str = "content",
    metadata_remap: dict[str, str] | None = None,
    replacements: dict[str, str] | None = None,
    extra_metadata_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> TransformAdapter:
  return TransformAdapter(
      dataset_format=InputDatasetFormat.SINGLE_COLUMN_MULTI_TURN,
      conversation_column=conversation_column,
      role_key=role_key,
      user_value=user_value,
      assistant_value=assistant_value,
      system_value=system_value,
      content_key=content_key,
      metadata_remap=metadata_remap or {},
      replacements=replacements,
      extra_metadata_fn=extra_metadata_fn,
  )


def instruction_response_adapter(
    *,
    instruction_column: str,
    response_column: str,
    content_key: str = "",
    filter_on_key: str = "",
    metadata_remap: dict[str, str] | None = None,
    replacements: dict[str, str] | None = None,
    extra_metadata_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> TransformAdapter:
  return TransformAdapter(
      dataset_format=InputDatasetFormat.INSTRUCTION_RESPONSE,
      instruction_column=instruction_column,
      response_column=response_column,
      content_key=content_key,
      filter_on_key=filter_on_key,
      metadata_remap=metadata_remap or {},
      replacements=replacements,
      extra_metadata_fn=extra_metadata_fn,
  )


def instruct_column_response_adapter(
    instruction_column: str,
    response_column: str,
    content_key: str,
    metadata_remap: dict[str, str] | None = None,
    replacements: dict[str, str] | None = None,
    extra_metadata_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> TransformAdapter:
  return TransformAdapter(
      dataset_format=InputDatasetFormat.INSTRUCT_COLUMN_RESPONSE,
      instruction_column=instruction_column,
      response_column=response_column,
      content_key=content_key,
      metadata_remap=metadata_remap or {},
      replacements=replacements,
      extra_metadata_fn=extra_metadata_fn,
  )


def instruct_msg_response_adapter(
    *,
    instruction_column: str,
    response_column: str,
    role_key: str = "role",
    user_value: str = "user",
    assistant_value: str = "assistant",
    system_value: str = "system",
    content_key: str = "content",
    metadata_remap: dict[str, str] | None = None,
    replacements: dict[str, str] | None = None,
    extra_metadata_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> TransformAdapter:
  return TransformAdapter(
      dataset_format=InputDatasetFormat.INSTRUCT_MSG_RESPONSE,
      instruction_column=instruction_column,
      response_column=response_column,
      role_key=role_key,
      user_value=user_value,
      assistant_value=assistant_value,
      system_value=system_value,
      content_key=content_key,
      metadata_remap=metadata_remap or {},
      replacements=replacements,
      extra_metadata_fn=extra_metadata_fn,
  )


# ---------------------------------------------------------------------------
# Dataset registry (name -> config). Pinned revisions preserved from marin.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InstructionDatasetConfig:
  """Where a dataset lives on HF + how to turn its rows into messages.

  Args:
    hf_dataset_id: HuggingFace repo id.
    revision: Pinned commit hash / tag (a short hash, as in the marin registry).
    adapter: Row → OpenAI-messages transform.
    metadata_columns: Source columns copied verbatim into ``metadata``.
    name: Friendly name; defaults to ``hf_dataset_id``.
    subsets: HF config name(s). The first non-default entry is passed as
      ``name=`` to ``load_dataset``; ``[]``/``["default"]`` means the default
      config.
    splits: Splits to stream; defaults to ``["train"]``.
  """

  hf_dataset_id: str
  revision: str
  adapter: TransformAdapter
  metadata_columns: list[str]
  name: str | None = None
  subsets: list[str] = field(default_factory=list)
  splits: list[str] = field(default_factory=lambda: ["train"])


# The tulu datasets are the priority for this experiment; their pinned revisions
# come straight from the marin registry.
INSTRUCTION_DATASET_NAME_TO_CONFIG: dict[str, InstructionDatasetConfig] = {
    # -- tulu (priority) -----------------------------------------------------
    "allenai/tulu-3-sft-mixture": InstructionDatasetConfig(
        hf_dataset_id="allenai/tulu-3-sft-mixture",
        revision="55e9fd6",
        adapter=multi_turn_adapter(),
        metadata_columns=["dataset", "id", "source"],
        name="allenai/tulu-3-sft-mixture",
    ),
    "allenai/tulu-v2-sft-mixture": InstructionDatasetConfig(
        hf_dataset_id="allenai/tulu-v2-sft-mixture",
        revision="6248b17",
        adapter=multi_turn_adapter(),
        metadata_columns=["dataset", "id"],
        name="allenai/tulu-v2-sft-mixture",
    ),
    "allenai/tulu-v2-sft-mixture-olmo-4096": InstructionDatasetConfig(
        hf_dataset_id="allenai/tulu-v2-sft-mixture-olmo-4096",
        revision="7a7c388",
        adapter=multi_turn_adapter(),
        metadata_columns=["dataset", "id"],
        name="allenai/tulu-v2-sft-mixture-olmo-4096",
    ),
    "sherryy/tulu-3-sft-personas-instruction-following-expanded": InstructionDatasetConfig(
        hf_dataset_id="sherryy/tulu-3-sft-personas-instruction-following-expanded",
        revision="79ab2c4",
        adapter=multi_turn_adapter(),
        metadata_columns=["dataset", "id"],
        name="sherryy/tulu-3-sft-personas-instruction-following-expanded",
    ),
    # -- a few cheap non-tulu entries (config dicts only; not the focus) ------
    "meta-math/MetaMathQA": InstructionDatasetConfig(
        hf_dataset_id="meta-math/MetaMathQA",
        revision="aa4f34d",
        adapter=instruction_response_adapter(
            instruction_column="query",
            response_column="response",
        ),
        metadata_columns=["type"],
        name="meta-math/MetaMathQA",
    ),
    "openbmb/UltraInteract_sft": InstructionDatasetConfig(
        hf_dataset_id="openbmb/UltraInteract_sft",
        revision="2b102e4",
        adapter=instruction_response_adapter(
            instruction_column="instruction",
            response_column="response",
        ),
        metadata_columns=["task", "dataset"],
        name="openbmb/UltraInteract_sft",
    ),
    "teknium/OpenHermes-2.5": InstructionDatasetConfig(
        hf_dataset_id="teknium/OpenHermes-2.5",
        revision="b820378",
        adapter=multi_turn_adapter(
            conversation_column="conversations",
            role_key="from",
            user_value="human",
            assistant_value="gpt",
            system_value="system",
            content_key="value",
        ),
        metadata_columns=["id", "category", "source"],
        name="teknium/OpenHermes-2.5",
    ),
    "HuggingFaceH4/no_robots": InstructionDatasetConfig(
        hf_dataset_id="HuggingFaceH4/no_robots",
        revision="e6f9a4a",
        adapter=multi_turn_adapter(),
        metadata_columns=["category", "prompt_id"],
        name="HuggingFaceH4/no_robots",
    ),
    "facebook/natural_reasoning": InstructionDatasetConfig(
        hf_dataset_id="facebook/natural_reasoning",
        revision="99eea5d",
        adapter=instruct_column_response_adapter(
            instruction_column="question",
            response_column="responses",
            content_key="response",
        ),
        metadata_columns=["reference_answer"],
        name="facebook/natural_reasoning",
    ),
    # -- smoltalk2 tool-calling SFT splits -----------------------------------
    # All three live in the single ``HuggingFaceTB/smoltalk2`` repo under the
    # ``SFT`` config (passed as ``name=`` to ``load_dataset``), differing only by
    # split. Each row carries a ``messages`` list of {role, content} dicts whose
    # roles include ``tool``; tool calls are pre-rendered into ``content``, so the
    # plain ``messages`` adapter (same as the tulu mixtures) handles them.
    "smoltalk2-xlam": InstructionDatasetConfig(
        hf_dataset_id="HuggingFaceTB/smoltalk2",
        revision="fc6cc21",
        adapter=multi_turn_adapter(),
        metadata_columns=[],
        name="smoltalk2-xlam",
        subsets=["SFT"],
        splits=["xlam_traces_no_think"],
    ),
    "smoltalk2-hermes-fc": InstructionDatasetConfig(
        hf_dataset_id="HuggingFaceTB/smoltalk2",
        revision="fc6cc21",
        adapter=multi_turn_adapter(),
        metadata_columns=[],
        name="smoltalk2-hermes-fc",
        subsets=["SFT"],
        splits=["hermes_function_calling_v1_no_think"],
    ),
    "smoltalk2-smolagents": InstructionDatasetConfig(
        hf_dataset_id="HuggingFaceTB/smoltalk2",
        revision="fc6cc21",
        adapter=multi_turn_adapter(),
        metadata_columns=[],
        name="smoltalk2-smolagents",
        subsets=["SFT"],
        splits=["smolagents_toolcalling_traces_think"],
    ),
}


# The tulu keys this experiment cares about, primary first.
TULU_DATASETS: list[str] = [
    "allenai/tulu-3-sft-mixture",
    "allenai/tulu-v2-sft-mixture",
    "allenai/tulu-v2-sft-mixture-olmo-4096",
    "sherryy/tulu-3-sft-personas-instruction-following-expanded",
]


# The smoltalk2 tool-calling splits, for blending a small tool-use slice into a
# conversational chat-SFT stage.
TOOLUSE_DATASETS: list[str] = [
    "smoltalk2-xlam",
    "smoltalk2-hermes-fc",
    "smoltalk2-smolagents",
]


# ---------------------------------------------------------------------------
# Row transform (adapter + replacements + metadata, per marin transform_row)
# ---------------------------------------------------------------------------


def transform_row(row: dict[str, Any], cfg: InstructionDatasetConfig) -> dict[str, Any] | None:
  """Apply ``cfg.adapter`` to *row* and attach metadata, or ``None`` to drop it.

  Mirrors ``marin.transform.conversation.transform_conversation.transform_row``
  for the message/text path: adapter → ``<think>`` replacements on string
  content → metadata columns → ``extra_metadata_fn``. We omit marin's Dolma
  envelope (id hash, ``source``/``added``/``created``) and tool-call argument
  normalization, which the raw-text SFT path here does not consume.
  """
  adapter = cfg.adapter
  messages = adapter.transform_conversation_to_openai_format(row)
  if messages is None:
    return None

  replacements = (
      adapter.replacements if adapter.replacements is not None else DEFAULT_TEXT_REPLACEMENTS
  )
  if replacements:
    for message in messages:
      content = message.get("content")
      if isinstance(content, str):
        message["content"] = _apply_replacements(content, replacements)

  metadata = {col: row.get(col, "") for col in cfg.metadata_columns}
  if adapter.extra_metadata_fn:
    extra = adapter.extra_metadata_fn(row)
    if extra:
      metadata.update(extra)

  return {"messages": messages, "metadata": metadata}


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


def _load_dataset_kwargs(cfg: InstructionDatasetConfig, split: str) -> dict[str, Any]:
  """Build ``datasets.load_dataset`` kwargs (streaming, pinned revision)."""
  kwargs: dict[str, Any] = {
      "path": cfg.hf_dataset_id,
      "split": split,
      "streaming": True,
      "revision": cfg.revision,
  }
  # First non-default subset becomes the HF config ``name`` (single-config
  # datasets like the tulu mixtures leave this unset).
  for subset in cfg.subsets:
    if subset not in (None, "default"):
      kwargs["name"] = subset
      break
  return kwargs


def load_instruction_messages(
    name: str,
    *,
    limit: int | None = None,
    split: str | None = None,
) -> Iterator[dict[str, Any]]:
  """Stream a registered instruction dataset as OpenAI chat messages.

  Args:
    name: Registry key (see :data:`INSTRUCTION_DATASET_NAME_TO_CONFIG`).
    limit: Max number of *emitted* examples (after dropping unconvertible rows).
      ``None`` streams the whole split. We pull from HF with a slack ``.take`` so
      that dropped rows do not starve the limit.
    split: Override the configured split (defaults to the first configured split,
      i.e. ``"train"``).

  Yields:
    ``{"messages": [{"role", "content"}, ...], "metadata": {...}}``. Roles are
    normalized to ``{"system", "user", "assistant", "tool"}``.

  Notes:
    Uses ``streaming=True`` so a multi-GB dataset is never downloaded whole; only
    the rows actually consumed are fetched. Requires network + (often)
    ``HF_TOKEN`` in env. Import is local so merely importing this module needs no
    ``datasets`` install.
  """
  from datasets import load_dataset

  if name not in INSTRUCTION_DATASET_NAME_TO_CONFIG:
    raise KeyError(
        f"Unknown instruction dataset: {name!r}. "
        f"Known: {sorted(INSTRUCTION_DATASET_NAME_TO_CONFIG)}"
    )
  cfg = INSTRUCTION_DATASET_NAME_TO_CONFIG[name]
  resolved_split = split if split is not None else cfg.splits[0]

  dataset = load_dataset(**_load_dataset_kwargs(cfg, resolved_split))

  # Bound the raw pull so a limit is honored even when some rows are dropped.
  # We over-pull by a safety factor and re-pull if exhausted (rare).
  if limit is not None:
    dataset = dataset.take(max(limit * 4, limit + 64))

  emitted = 0
  for row in dataset:
    out = transform_row(row, cfg)
    if out is None:
      continue
    yield out
    emitted += 1
    if limit is not None and emitted >= limit:
      return


def load_tulu_sft(limit: int | None = None, *, split: str | None = None) -> Iterator[dict[str, Any]]:
  """Convenience: stream ``allenai/tulu-3-sft-mixture`` (the primary tulu mix)."""
  return load_instruction_messages("allenai/tulu-3-sft-mixture", limit=limit, split=split)


# Default mixture: mostly tulu-3 chat with a small deliberate tool-use slice
# (~85% chat / ~15% tool-use, the latter split evenly across the three smoltalk2
# function-calling splits). Weights are relative; they need not sum to 1.
_UP_TO_SHAPE_DEFAULT_WEIGHTS: dict[str, float] = {
    "allenai/tulu-3-sft-mixture": 0.85,
    "smoltalk2-xlam": 0.05,
    "smoltalk2-hermes-fc": 0.05,
    "smoltalk2-smolagents": 0.05,
}


def load_up_to_shape_mixture(
    *,
    limit: int | None = None,
    seed: int = 0,
    weights: dict[str, float] | None = None,
) -> Iterator[dict[str, Any]]:
  """Stream a weighted interleave of tulu-3 chat + smoltalk2 tool-use.

  Blends ``allenai/tulu-3-sft-mixture`` with the three :data:`TOOLUSE_DATASETS`
  splits so a base model can be post-trained on conversational **and** tool-use
  data in one stream. The default proportions are ~85% tulu-3 chat and ~15%
  tool-use (split evenly across the three smoltalk2 splits); pass *weights* to
  override (keys are registry names, values are relative weights that need not
  sum to 1).

  Each underlying dataset is consumed lazily via :func:`load_instruction_messages`
  (``streaming=True``), so nothing is materialized up front. On each step we draw
  a source weighted by *weights* and yield its next row; exhausted sources are
  dropped and the remaining weights renormalize, so the stream ends only when all
  sources are exhausted (or *limit* is reached).

  Args:
    limit: Max number of *emitted* examples across all sources. ``None`` runs
      until every source is exhausted.
    seed: Seed for the source-selection RNG (deterministic interleave).
    weights: Optional ``{registry_name: relative_weight}`` override. Defaults to
      :data:`_UP_TO_SHAPE_DEFAULT_WEIGHTS`.

  Yields:
    ``{"messages": [...], "metadata": {...}}`` rows, the same shape as
    :func:`load_instruction_messages`.
  """
  import random

  weights = dict(weights if weights is not None else _UP_TO_SHAPE_DEFAULT_WEIGHTS)
  if not weights:
    raise ValueError("load_up_to_shape_mixture requires at least one weighted source")

  rng = random.Random(seed)
  # name -> (iterator, weight); we never materialize a source, only pull next().
  streams: dict[str, Iterator[dict[str, Any]]] = {
      name: load_instruction_messages(name) for name in weights
  }

  emitted = 0
  while streams:
    names = list(streams)
    pick = rng.choices(names, weights=[weights[n] for n in names], k=1)[0]
    try:
      row = next(streams[pick])
    except StopIteration:
      # Source exhausted: drop it and let the remaining weights renormalize.
      del streams[pick]
      continue
    yield row
    emitted += 1
    if limit is not None and emitted >= limit:
      return
