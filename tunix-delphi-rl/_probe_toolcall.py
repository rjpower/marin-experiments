"""M-format probe: can the Delphi base LM emit a PARSEABLE tool call from few-shot?

Compares candidate tool-call surface syntaxes (Qwen <tool_call> JSON vs a simple
CALC(a op b) fallback) by greedy-decoding Delphi on held-out 2-digit multiplies
and measuring (a) tool_call_rate = parseable call emitted, (b) arg_acc = correct
operands. Greedy is the best case: if greedy can't produce the format, temp>0 RL
rollouts won't reliably either. Decides the T0 tool surface + whether we need a
custom parser (CALC) or can use tunix's stock QwenToolParser.
"""
import json
import re
import jax.numpy as jnp
from _probe_arith_format import greedy
from delphi_qwen3 import load_delphi, load_tokenizer

DELPHI_DIR = "/home/power/code/_tunix_lab/delphi"

# Few-shot demos per format. Prompt ends right where the model should emit the
# call for the final problem.
QWEN = (
    'Q: 12 * 13\n<tool_call>{{"name": "calculator", "arguments": {{"a": 12, "b": 13, "op": "*"}}}}</tool_call>\n'
    'Q: 24 * 31\n<tool_call>{{"name": "calculator", "arguments": {{"a": 24, "b": 31, "op": "*"}}}}</tool_call>\n'
    'Q: 58 * 46\n<tool_call>{{"name": "calculator", "arguments": {{"a": 58, "b": 46, "op": "*"}}}}</tool_call>\n'
    'Q: {a} * {b}\n'
)
CALC = (
    "Q: 12 * 13\nCALC(12 * 13)\n"
    "Q: 24 * 31\nCALC(24 * 31)\n"
    "Q: 58 * 46\nCALC(58 * 46)\n"
    "Q: {a} * {b}\n"
)

_QWEN_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_CALC_RE = re.compile(r"CALC\(\s*(\d+)\s*\*\s*(\d+)\s*\)")


def parse_qwen(text, a, b):
  m = _QWEN_RE.search(text)
  if not m:
    return False, False
  try:
    obj = json.loads(m.group(1))
    args = obj.get("arguments", {})
    ok_args = (obj.get("name") == "calculator"
               and {int(args.get("a")), int(args.get("b"))} == {a, b}
               and args.get("op") == "*")
    return True, ok_args
  except Exception:
    return True, False  # parseable tags but bad json/args


def parse_calc(text, a, b):
  m = _CALC_RE.search(text)
  if not m:
    return False, False
  return True, {int(m.group(1)), int(m.group(2))} == {a, b}


def main():
  tokenizer = load_tokenizer(DELPHI_DIR)
  model = load_delphi(DELPHI_DIR, dtype=jnp.bfloat16)
  cases = [(47, 53), (62, 18), (33, 27), (84, 19), (56, 44),
           (71, 23), (39, 48), (95, 12), (28, 67), (43, 51)]
  for name, tmpl, parser, maxnew in [
      ("QWEN_JSON", QWEN, parse_qwen, 48),
      ("CALC", CALC, parse_calc, 16),
  ]:
    n_call = n_arg = 0
    print(f"\n===== format: {name} =====")
    for a, b in cases:
      cont = greedy(model, tokenizer, tmpl.format(a=a, b=b), max_new=maxnew)
      called, arg_ok = parser(cont, a, b)
      n_call += called
      n_arg += arg_ok
      print(f"  {a}*{b}: call={int(called)} arg_ok={int(arg_ok)} -> {cont[:70]!r}")
    n = len(cases)
    print(f"  >>> {name}: tool_call_rate={n_call}/{n}={n_call/n:.2f}  arg_acc={n_arg}/{n}={n_arg/n:.2f}")


if __name__ == "__main__":
  main()
