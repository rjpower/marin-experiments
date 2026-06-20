# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the :mod:`environments.micropython` sandboxed interpreter.

Run with ``uv run pytest tests/test_micropython.py -q``.

The tests cover the language subset (literals, arithmetic, precedence, control
flow, recursion, comprehensions, list/string methods, f-strings, ``print``
kwargs), the error/safety surface (NameError, ZeroDivisionError, IndexError,
step/output limits, blocked ``import`` and dunder access), and determinism.
"""

from __future__ import annotations

import dataclasses

from environments.micropython import ExecResult, run


# --- helpers -------------------------------------------------------------------


def _ok(source: str, **kw) -> str:
  """Run ``source``, assert it succeeded, and return its stdout."""
  result = run(source, **kw)
  assert result.ok, f"expected ok, got error={result.error!r}"
  assert result.error is None
  return result.stdout


def _err(source: str, **kw) -> str:
  """Run ``source``, assert it failed, and return the error string."""
  result = run(source, **kw)
  assert not result.ok, f"expected failure, got stdout={result.stdout!r}"
  assert result.error is not None
  return result.error


# --- literals & arithmetic -----------------------------------------------------


def test_print_int():
  assert _ok("print(2)") == "2\n"


def test_print_string_and_bool_and_none():
  assert _ok("print('hi')") == "hi\n"
  assert _ok("print(True, False, None)") == "True False None\n"


def test_int_arithmetic():
  assert _ok("print(1 + 2 * 3)") == "7\n"
  assert _ok("print((1 + 2) * 3)") == "9\n"
  assert _ok("print(7 - 3 - 1)") == "3\n"


def test_floor_div_mod_pow():
  assert _ok("print(7 // 2)") == "3\n"
  assert _ok("print(7 % 3)") == "1\n"
  assert _ok("print(2 ** 10)") == "1024\n"
  assert _ok("print(-7 // 2)") == "-4\n"  # Python floor semantics


def test_true_division_is_float():
  assert _ok("print(7 / 2)") == "3.5\n"
  assert _ok("print(6 / 3)") == "2.0\n"


def test_float_arithmetic():
  assert _ok("print(0.1 + 0.2)") == f"{0.1 + 0.2}\n"
  assert _ok("print(3.0 * 2)") == "6.0\n"


def test_precedence_with_unary_and_pow():
  assert _ok("print(-2 ** 2)") == "-4\n"  # -(2**2)
  assert _ok("print(2 ** 3 ** 2)") == "512\n"  # right-assoc


def test_unary_ops():
  assert _ok("print(-5)") == "-5\n"
  assert _ok("print(+5)") == "5\n"
  assert _ok("print(not True)") == "False\n"
  assert _ok("print(not 0)") == "True\n"


# --- variables, assignment, unpacking ------------------------------------------


def test_variables():
  assert _ok("x = 5\ny = x + 1\nprint(y)") == "6\n"


def test_augmented_assign():
  assert _ok("x = 1\nx += 4\nx *= 2\nprint(x)") == "10\n"
  assert _ok("x = 10\nx -= 3\nx //= 2\nprint(x)") == "3\n"


def test_tuple_unpacking():
  assert _ok("a, b = 1, 2\nprint(a, b)") == "1 2\n"
  # Multiple targets assign left-to-right, like CPython.
  assert _ok("x = y = 5\nprint(x, y)") == "5 5\n"


def test_swap_via_unpacking():
  assert _ok("a, b = 1, 2\na, b = b, a\nprint(a, b)") == "2 1\n"


def test_subscript_assignment():
  assert _ok("xs = [1, 2, 3]\nxs[0] = 9\nprint(xs)") == "[9, 2, 3]\n"


# --- comparisons & boolean -----------------------------------------------------


def test_comparisons():
  assert _ok("print(1 < 2, 2 <= 2, 3 > 2, 2 >= 3)") == "True True True False\n"
  assert _ok("print(1 == 1, 1 != 2)") == "True True\n"


def test_chained_comparison():
  assert _ok("print(1 < 2 < 3)") == "True\n"
  assert _ok("print(1 < 2 > 5)") == "False\n"
  assert _ok("x = 5\nprint(0 <= x <= 10)") == "True\n"


def test_boolean_short_circuit():
  assert _ok("print(True and 'yes')") == "yes\n"
  assert _ok("print(False or 'fallback')") == "fallback\n"
  assert _ok("print(0 and 1, 1 or 2)") == "0 1\n"


def test_membership():
  assert _ok("print(3 in [1, 2, 3])") == "True\n"
  assert _ok("print('a' not in 'xyz')") == "True\n"


def test_ternary():
  assert _ok("x = 5\nprint('big' if x > 3 else 'small')") == "big\n"


# --- control flow --------------------------------------------------------------


def test_if_elif_else():
  src = "x = 2\nif x == 1:\n  print('a')\nelif x == 2:\n  print('b')\nelse:\n  print('c')"
  assert _ok(src) == "b\n"


def test_while_loop():
  src = "i = 0\nwhile i < 3:\n  print(i)\n  i += 1"
  assert _ok(src) == "0\n1\n2\n"


def test_while_break():
  src = "i = 0\nwhile True:\n  if i == 2:\n    break\n  print(i)\n  i += 1"
  assert _ok(src) == "0\n1\n"


def test_while_continue():
  src = "i = 0\nout = []\nwhile i < 5:\n  i += 1\n  if i % 2 == 0:\n    continue\n  print(i)"
  assert _ok(src) == "1\n3\n5\n"


def test_for_range():
  assert _ok("for i in range(3):\n  print(i)") == "0\n1\n2\n"
  assert _ok("for i in range(1, 6, 2):\n  print(i)") == "1\n3\n5\n"


def test_for_over_list_and_string():
  assert _ok("for c in 'ab':\n  print(c)") == "a\nb\n"
  assert _ok("for x in [10, 20]:\n  print(x)") == "10\n20\n"


def test_for_break_continue():
  src = "for i in range(10):\n  if i == 3:\n    break\n  if i == 1:\n    continue\n  print(i)"
  assert _ok(src) == "0\n2\n"


def test_for_else():
  src = "for i in range(3):\n  print(i)\nelse:\n  print('done')"
  assert _ok(src) == "0\n1\n2\ndone\n"


def test_for_else_skipped_on_break():
  src = "for i in range(3):\n  if i == 1:\n    break\nelse:\n  print('done')\nprint('after')"
  assert _ok(src) == "after\n"


# --- functions & recursion -----------------------------------------------------


def test_simple_function():
  assert _ok("def add(a, b):\n  return a + b\nprint(add(2, 3))") == "5\n"


def test_function_default_args():
  src = "def greet(name, greeting='hi'):\n  return greeting + ' ' + name\nprint(greet('x'))\nprint(greet('y', 'yo'))"
  assert _ok(src) == "hi x\nyo y\n"


def test_function_keyword_args():
  src = "def f(a, b):\n  return a - b\nprint(f(b=1, a=10))"
  assert _ok(src) == "9\n"


def test_local_scope_does_not_leak():
  src = "def f():\n  x = 99\n  return x\nx = 1\nprint(f())\nprint(x)"
  assert _ok(src) == "99\n1\n"


def test_function_reads_globals():
  src = "G = 7\ndef f():\n  return G + 1\nprint(f())"
  assert _ok(src) == "8\n"


def test_fibonacci_recursion():
  src = (
      "def fib(n):\n"
      "  if n < 2:\n"
      "    return n\n"
      "  return fib(n - 1) + fib(n - 2)\n"
      "print(fib(10))"
  )
  assert _ok(src) == "55\n"


def test_factorial_recursion():
  src = (
      "def fact(n):\n"
      "  if n <= 1:\n"
      "    return 1\n"
      "  return n * fact(n - 1)\n"
      "print(fact(5))"
  )
  assert _ok(src) == "120\n"


def test_return_none_default():
  src = "def f():\n  pass\nprint(f())"
  assert _ok(src) == "None\n"


# --- comprehensions ------------------------------------------------------------


def test_list_comprehension():
  assert _ok("print([x * x for x in range(5)])") == "[0, 1, 4, 9, 16]\n"


def test_list_comprehension_with_filter():
  assert _ok("print([x for x in range(10) if x % 2 == 0])") == "[0, 2, 4, 6, 8]\n"


def test_comprehension_var_does_not_leak():
  assert _ok("ys = [i for i in range(3)]\nprint(ys)") == "[0, 1, 2]\n"
  # ``i`` must not be defined after the comprehension.
  assert "NameError" in _err("ys = [i for i in range(3)]\nprint(i)")


# --- list operations -----------------------------------------------------------


def test_list_indexing_and_slicing():
  assert _ok("xs = [1, 2, 3, 4]\nprint(xs[0], xs[-1])") == "1 4\n"
  assert _ok("xs = [1, 2, 3, 4]\nprint(xs[1:3])") == "[2, 3]\n"
  assert _ok("xs = [1, 2, 3, 4]\nprint(xs[::-1])") == "[4, 3, 2, 1]\n"


def test_list_concat_and_repeat():
  assert _ok("print([1, 2] + [3])") == "[1, 2, 3]\n"
  assert _ok("print([0] * 3)") == "[0, 0, 0]\n"


def test_list_methods():
  src = "xs = [1, 2]\nxs.append(3)\nxs.insert(0, 0)\nprint(xs)\nprint(xs.pop())\nprint(xs.index(2), xs.count(1))"
  assert _ok(src) == "[0, 1, 2, 3]\n3\n2 1\n"


def test_builtins_over_lists():
  assert _ok("print(len([1, 2, 3]))") == "3\n"
  assert _ok("print(sum([1, 2, 3]))") == "6\n"
  assert _ok("print(min([3, 1, 2]), max([3, 1, 2]))") == "1 3\n"
  assert _ok("print(sorted([3, 1, 2]))") == "[1, 2, 3]\n"
  assert _ok("print(reversed([1, 2, 3]))") == "[3, 2, 1]\n"
  assert _ok("print(list(range(3)))") == "[0, 1, 2]\n"


def test_enumerate_and_zip():
  assert _ok("print(enumerate(['a', 'b']))") == "[(0, 'a'), (1, 'b')]\n"
  assert _ok("print(zip([1, 2], [3, 4]))") == "[(1, 3), (2, 4)]\n"


# --- string operations ---------------------------------------------------------


def test_string_indexing_slicing():
  assert _ok("s = 'hello'\nprint(s[0], s[-1])") == "h o\n"
  assert _ok("s = 'hello'\nprint(s[1:4])") == "ell\n"


def test_string_concat_repeat_membership():
  assert _ok("print('ab' + 'cd')") == "abcd\n"
  assert _ok("print('ab' * 3)") == "ababab\n"
  assert _ok("print('b' in 'abc')") == "True\n"


def test_string_methods():
  assert _ok("print('Hi'.upper(), 'Hi'.lower())") == "HI hi\n"
  assert _ok("print('  x '.strip())") == "x\n"
  assert _ok("print('a,b,c'.split(','))") == "['a', 'b', 'c']\n"
  assert _ok("print('-'.join(['a', 'b']))") == "a-b\n"
  assert _ok("print('abc'.replace('b', 'X'))") == "aXc\n"
  assert _ok("print('abc'.find('c'))") == "2\n"


# --- f-strings & formatting ----------------------------------------------------


def test_fstring_basic():
  assert _ok("x = 7\nprint(f'x is {x}')") == "x is 7\n"


def test_fstring_expression_and_format_spec():
  assert _ok("print(f'{2 + 3}')") == "5\n"
  assert _ok("print(f'{3.14159:.2f}')") == "3.14\n"
  assert _ok("print(f'{42:05d}')") == "00042\n"


def test_percent_formatting():
  assert _ok("print('%d-%s' % (3, 'x'))") == "3-x\n"
  assert _ok("print('%.1f' % 2.5)") == "2.5\n"


# --- print kwargs --------------------------------------------------------------


def test_print_sep():
  assert _ok("print(1, 2, 3, sep='-')") == "1-2-3\n"


def test_print_end():
  assert _ok("print('a', end='')\nprint('b')") == "ab\n"


def test_print_sep_and_end():
  assert _ok("print('a', 'b', sep='|', end='!')") == "a|b!"


def test_print_empty():
  assert _ok("print()") == "\n"


# --- multi-statement programs --------------------------------------------------


def test_multi_statement_program():
  src = (
      "total = 0\n"
      "for i in range(1, 5):\n"
      "  total += i\n"
      "def double(x):\n"
      "  return x * 2\n"
      "print(double(total))"
  )
  assert _ok(src) == "20\n"


def test_fizzbuzz():
  src = (
      "for i in range(1, 6):\n"
      "  if i % 15 == 0:\n"
      "    print('FizzBuzz')\n"
      "  elif i % 3 == 0:\n"
      "    print('Fizz')\n"
      "  elif i % 5 == 0:\n"
      "    print('Buzz')\n"
      "  else:\n"
      "    print(i)"
  )
  assert _ok(src) == "1\n2\nFizz\n4\nBuzz\n"


# --- error & safety cases ------------------------------------------------------


def test_syntax_error():
  err = _err("print(")
  assert err.startswith("SyntaxError:")


def test_name_error():
  assert _err("print(undefined_var)").startswith("NameError:")


def test_zero_division():
  assert _err("print(1 / 0)").startswith("ZeroDivisionError:")
  assert _err("print(1 // 0)").startswith("ZeroDivisionError:")
  assert _err("print(1 % 0)").startswith("ZeroDivisionError:")


def test_index_error():
  assert _err("xs = [1, 2]\nprint(xs[5])").startswith("IndexError:")
  assert _err("s = 'ab'\nprint(s[9])").startswith("IndexError:")


def test_type_error():
  assert _err("print('a' + 1)").startswith("TypeError:")


def test_not_callable():
  assert _err("x = 5\nprint(x())").startswith("TypeError:")


def test_unpack_mismatch():
  assert _err("a, b = 1, 2, 3").startswith("ValueError:")


def test_step_limit_infinite_while():
  result = run("while True:\n  pass", max_steps=1000)
  assert not result.ok
  assert result.error == "StepLimit: exceeded max_steps"
  assert result.steps >= 1000


def test_step_limit_infinite_recursion():
  src = "def f():\n  return f()\nf()"
  result = run(src, max_steps=5000)
  assert not result.ok
  # Either our step limit or the host RecursionError guard; both are clean.
  assert result.error in (
      "StepLimit: exceeded max_steps",
      "RecursionError: maximum recursion depth exceeded",
  )


def test_output_limit():
  src = "while True:\n  print('x')"
  result = run(src, max_output=10)
  assert not result.ok
  assert result.error == "OutputLimit: exceeded max_output"
  assert len(result.stdout) == 10  # truncated prefix preserved


def test_output_limit_partial_preserved():
  result = run("print('hello world')", max_output=5)
  assert not result.ok
  assert result.error == "OutputLimit: exceeded max_output"
  assert result.stdout == "hello"


# --- disallowed constructs -----------------------------------------------------


def test_import_disallowed():
  assert _err("import os").startswith("UnsupportedSyntax:")
  assert _err("import os\nprint(os.getcwd())").startswith("UnsupportedSyntax:")


def test_import_from_disallowed():
  assert _err("from os import getcwd").startswith("UnsupportedSyntax:")


def test_class_disallowed():
  assert _err("class Foo:\n  pass").startswith("UnsupportedSyntax: ClassDef")


def test_with_disallowed():
  assert _err("with open('x') as f:\n  pass").startswith("UnsupportedSyntax: With")


def test_try_disallowed():
  assert _err("try:\n  pass\nexcept:\n  pass").startswith("UnsupportedSyntax: Try")


def test_lambda_disallowed():
  assert _err("f = lambda x: x").startswith("UnsupportedSyntax: Lambda")


def test_global_disallowed():
  assert _err("def f():\n  global x\n  x = 1\nf()").startswith(
      "UnsupportedSyntax: Global"
  )


def test_generator_disallowed():
  # A generator expression is an unsupported expression node.
  assert _err("print(sum(x for x in range(3)))").startswith("UnsupportedSyntax:")


def test_decorator_disallowed():
  assert _err("@deco\ndef f():\n  pass").startswith("UnsupportedSyntax: decorators")


def test_dict_and_set_literals_disallowed():
  assert _err("print({'a': 1})").startswith("UnsupportedSyntax: Dict")
  assert _err("print({1, 2})").startswith("UnsupportedSyntax: Set")


def test_assert_del_yield_disallowed():
  assert _err("assert False").startswith("UnsupportedSyntax: Assert")
  assert _err("x = 1\ndel x").startswith("UnsupportedSyntax: Delete")
  assert _err("yield 1").startswith("UnsupportedSyntax: Yield")


def test_starargs_disallowed():
  assert _err("print(*[1, 2])").startswith("UnsupportedSyntax:")


def test_eval_and_exec_not_available():
  assert _err("eval('1+1')").startswith("NameError:")
  assert _err("exec('x=1')").startswith("NameError:")


def test_open_and_input_not_available():
  assert _err("open('f')").startswith("NameError:")
  assert _err("input()").startswith("NameError:")


def test_dunder_attribute_access_blocked():
  assert _err("print((1).__class__)").startswith("AttributeError:")
  assert _err("print(''.__class__)").startswith("AttributeError:")
  assert _err("print([].__class__.__bases__)").startswith("AttributeError:")


def test_arbitrary_attribute_access_blocked():
  # Even non-dunder attribute access is disallowed (no data attributes exist).
  assert _err("x = 5\nprint(x.real)").startswith("AttributeError:")


def test_dunder_method_call_blocked():
  assert _err("print('abc'.__len__())").startswith("AttributeError:")


def test_unknown_method_blocked():
  assert _err("[].sort()").startswith("AttributeError:")  # not whitelisted
  assert _err("'x'.format()").startswith("AttributeError:")


def test_builtins_not_leaked():
  for name in ("__import__", "globals", "locals", "vars", "getattr", "setattr"):
    assert _err(f"{name}()").startswith("NameError:"), name


# --- determinism ---------------------------------------------------------------


def test_determinism_success():
  src = "def fib(n):\n  return n if n < 2 else fib(n-1) + fib(n-2)\nprint(fib(10))"
  first = run(src)
  second = run(src)
  assert first == second
  assert isinstance(first, ExecResult)
  assert first.stdout == "55\n"
  assert first.steps == second.steps and first.steps > 0


def test_determinism_error():
  src = "print(1 / 0)"
  assert run(src) == run(src)


def test_determinism_step_limit():
  src = "while True:\n  pass"
  assert run(src, max_steps=2000) == run(src, max_steps=2000)


# --- result shape / API --------------------------------------------------------


def test_result_is_frozen_dataclass():
  result = run("print(1)")
  assert isinstance(result, ExecResult)
  try:
    result.ok = False  # type: ignore[misc]
    raised = False
  except dataclasses.FrozenInstanceError:  # type: ignore[name-defined]
    raised = True
  assert raised


def test_steps_counted_on_success():
  result = run("print(1)")
  assert result.ok and result.steps > 0
