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

from typing import Any, Dict, List


class DelphiRawTextChatParser:
  """Raw-text chat parser for the Delphi base LM (no chat template).

  Renders a list of chat messages into a single plain-text string by
  concatenating the per-role contents with newlines. This mirrors how the
  non-agentic Delphi harness feeds raw few-shot prompts to the base model: the
  arithmetic ``prompts`` column already ends exactly where the model should emit
  its answer (e.g. ``"... Q: 3 + 4 = A:"``), so the rendered user turn IS the
  raw prompt and the model continues it directly.

  The ``add_generation_prompt`` and ``is_first_msg`` flags are intentionally
  ignored: a base LM has no assistant/generation marker to append and no BOS to
  inject here (the rollout tokenizer adds BOS itself). Deliberately NOT exposing
  an ``assistant_token`` attribute keeps ``convert_single_message`` from trying
  to strip a generation prefix off assistant turns (there is none).
  """

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
      add_generation_prompt: ignored (a base LM has no generation marker).
      is_first_msg: ignored (the rollout tokenizer injects BOS itself).

    Returns:
      The messages' contents joined by newlines, with ``tool`` messages prefixed
      by ``"Tool result: "``.
    """
    del add_generation_prompt, is_first_msg
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
    return "\n".join(parts)
