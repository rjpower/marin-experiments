"""Shared agentic-GRPO glue for porting the Delphi harness onto tunix's agentic stack.

Delphi (``marin-community/delphi-3e18-447Mparams-1.2Btokens``) is a 447M dense
Qwen3 **base** LM with NO chat template (Llama-3 tokenizer, vocab 128256,
``EOS=128001``, ``pad=eos``). The stock ``QwenChatTemplateParser`` emits Qwen
control strings (``<|im_start|>``/``<|im_end|>``) that are out-of-vocab for the
Llama-3 tokenizer, so it cannot be used here. The agentic learner instead lets
you inject a custom ``chat_parser`` whose ``parse`` renders messages to raw text;
:class:`DelphiRawTextChatParser` is that raw-text renderer (modeled on the
``VTCRawTextParser`` in ``examples/agentic/qwen3_grpo_gsm8k_demo.py``).

The parser is called in two places by the engine (both with the same
``parse(messages, add_generation_prompt=..., is_first_msg=...) -> str`` contract):

  * :meth:`AgenticRLLearner._model_call` calls ``parse(messages, ...)`` on the
    full chat list to build the rollout prompt string.
  * ``tunix.rl.agentic.utils.convert_single_message`` calls ``parse([msg], ...)``
    per message when re-tokenizing the rollout to build the assistant loss mask
    (assistant tokens -> mask 1, all other roles -> mask 0). For a single-turn
    rollout the whole completion is assistant text, so the mask is all ones.

A ``role == "tool"`` branch is included now (rendered as ``"Tool result: ..."``)
even though milestone M-port is single-turn with no tools: later tool stages
(T0+) feed tool outputs back as ``tool`` messages, and rendering them as plain
text is harmless for the single-turn case (no tool messages are produced).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import optax

# Temporary diagnostic: when DELPHI_T0_DEBUG=1, print the rendered MULTI-TURN
# rollout prompt (the one containing a tool result) so we can see exactly what
# the model is asked to continue on turn 2.
_DEBUG = os.environ.get("DELPHI_T0_DEBUG", "0") == "1"


def clipped_adamw(learning_rate: float) -> optax.GradientTransformation:
  """Builds the global-norm-clipped AdamW used by every Delphi training phase.

  Gradient clipping is LOAD-BEARING for the multi-turn tool stages, not just a
  stability nicety: an occasional exploding update (more frequent at higher lr)
  produced ``inf``/``NaN`` gradients that crashed the TPU run with a libtpu
  ``SIGSEGV`` mid-training (e.g. lr=2e-5 crashed ~step 3, lr=1e-5 ~step 99),
  losing all progress. Clipping the global norm to 1.0 bounds the update, keeps
  the run alive long enough to converge, and is harmless for the single-turn
  M-port. Both the SFT warm-up (:func:`agentic_sft.run_sft_warmup`) and the RL
  phase (:func:`train_agentic._train_agentic_calc`) use this same optimizer.

  Args:
    learning_rate: the AdamW learning rate.

  Returns:
    ``optax.chain(clip_by_global_norm(1.0), adamw(lr, b1=0.9, b2=0.99, wd=0.0))``.
  """
  return optax.chain(
      optax.clip_by_global_norm(1.0),
      optax.adamw(learning_rate=learning_rate, b1=0.9, b2=0.99, weight_decay=0.0),
  )


class DelphiRawTextChatParser:
  """Raw-text chat parser for the Delphi base LM (no chat template).

  Renders a list of chat messages into a single plain-text string by
  concatenating the per-role contents with newlines. This mirrors how the
  non-agentic Delphi harness feeds raw few-shot prompts to the base model: the
  arithmetic ``prompts`` column already ends exactly where the model should emit
  its answer (e.g. ``"... Q: 3 + 4 = A:"``), so the rendered user turn IS the
  raw prompt and the model continues it directly.

  The ``is_first_msg`` flag is intentionally ignored: a base LM has no BOS to
  inject here (the rollout tokenizer adds BOS itself). Deliberately NOT exposing
  an ``assistant_token`` attribute keeps ``convert_single_message`` from trying
  to strip a generation prefix off assistant turns (there is none).

  ``add_generation_prompt`` is honored only through the optional
  :attr:`generation_suffix` (default ``""``, so single-turn M-port is unchanged:
  its prompts already end exactly where the model should emit, e.g. ``"...A:"``).
  Multi-turn TOOL stages (T0+) set ``generation_suffix="\\n"`` so the rollout
  prompt ends with a trailing newline and the model begins its turn on a fresh
  line, emitting ``<tool_call>`` (or the final number) DIRECTLY instead of a
  leading ``"\\n<tool_call>"``. The leading-newline matters because the tool
  stages add the newline token to ``eos_tokens`` (single-line turns): if the
  model's first generated token were the newline, the completion would be empty
  and the agentic engine would discard the turn (the M-port degenerate failure).
  """

  def __init__(self, generation_suffix: str = ""):
    """Builds the raw-text parser.

    Args:
      generation_suffix: text appended to the rendered prompt when
        ``add_generation_prompt=True`` (the live rollout prompt). Default ``""``
        preserves the M-port single-turn behavior; tool stages pass ``"\\n"``.
    """
    self.generation_suffix = generation_suffix

  def parse(
      self,
      messages: List[Dict[str, Any]],
      add_generation_prompt: bool = False,
      is_first_msg: bool = False,
  ) -> str:
    """Renders chat messages to a single raw-text string.

    Args:
      messages: a list of ``{"role": str, "content": str}`` dicts. Supported
        roles are ``system``, ``user``, ``assistant`` and ``tool``; an empty
        ``system``/``assistant`` content contributes nothing.
      add_generation_prompt: when True (the live rollout prompt), appends
        :attr:`generation_suffix` to the result.
      is_first_msg: ignored (the rollout tokenizer injects BOS itself).

    Returns:
      The messages' contents joined by newlines, with ``tool`` messages prefixed
      by ``"Tool result: "``, plus :attr:`generation_suffix` when
      ``add_generation_prompt`` is set.
    """
    del is_first_msg
    parts: List[str] = []
    for message in messages:
      role = message.get("role")
      content = message.get("content", "")
      if content is None:
        content = ""
      content = str(content)
      if role == "system":
        if content:
          parts.append(content)
      elif role == "user":
        parts.append(content)
      elif role == "assistant":
        if content:
          parts.append(content)
      elif role == "tool":
        # Single-turn M-port never emits tool messages; rendered here so the
        # later tool stages (T0+) can feed tool outputs back as plain text.
        parts.append(f"Tool result: {content}")
    rendered = "\n".join(parts)
    if add_generation_prompt and self.generation_suffix:
      rendered += self.generation_suffix
    if (
        _DEBUG
        and add_generation_prompt
        and any(m.get("role") == "tool" for m in messages)
    ):
      # Turn-2 rollout prompt: show the tail (the tool result + where the model
      # must continue) so we can confirm the result is present and well-rendered.
      print(f"[t0-dbg] turn2-prompt-tail={rendered[-160:]!r}", flush=True)
    return rendered
