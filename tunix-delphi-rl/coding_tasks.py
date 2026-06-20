# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Graded coding tasks for the micropython agentic-coding RL experiment.

Each :class:`Task` pairs a natural-language ``prompt`` with a reference
``solution`` (a program valid under :mod:`micropython`) and the exact gold
``answer`` that ``micropython.run(solution).stdout`` produces. The RL setup
shows a small (447M) model the ``prompt``, asks it to emit a tiny Python
program, executes that program with :func:`micropython.run`, and rewards an
*exact* stdout match against ``answer``. The same reference solutions seed SFT
by way of replayed execution traces.

Because the reward is an exact string match, every prompt is written to fully
determine its stdout -- the value, the formatting, the separators, and the
implicit trailing newline that ``print`` appends. Every solution stays strictly
inside the interpreter's supported subset (no imports, classes, dicts/sets,
``with``/``try``, lambdas, or non-whitelisted builtins/methods).

The 50 tasks form a five-tier difficulty curriculum (~10 tasks per tier):

  * **Tier 0 -- constant output.** Print a literal (int, string, float, bool).
  * **Tier 1 -- one-step arithmetic.** Print one expression over constants,
    covering ``+ - * // % **`` plus parenthesized / float division.
  * **Tier 2 -- variables & conditionals.** Assign and combine variables;
    simple ``if``/``else`` and ternaries; small fixed computations.
  * **Tier 3 -- loops.** ``for``/``while`` over ``range``, accumulation, and
    sequence/string manipulation, with single- and multi-line outputs.
  * **Tier 4 -- functions & recursion.** ``def`` + ``return``, recursion, and
    small classic algorithms (fib, factorial, gcd, primality, ...).

Run ``uv run python coding_tasks.py`` to validate every task against the
interpreter.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Task:
  """One graded coding task.

  Attributes:
    id: Stable unique slug, e.g. ``"t1_03_six_times_seven"``.
    tier: Curriculum/difficulty tier in ``0..4``.
    prompt: The natural-language instruction shown to the model.
    solution: A correct micropython program (the reference solution).
    answer: The gold stdout, equal to ``micropython.run(solution).stdout``.
    concepts: Tags drawn from a small consistent vocabulary.
  """

  id: str
  tier: int
  prompt: str
  solution: str
  answer: str
  concepts: tuple[str, ...]


TASKS: list[Task] = [
    # --- Tier 0: constant output ----------------------------------------------
    Task(
        id="t0_01_print_two",
        tier=0,
        prompt="Print the number 2.",
        solution="print(2)",
        answer="2\n",
        concepts=("print",),
    ),
    Task(
        id="t0_02_print_zero",
        tier=0,
        prompt="Print the number 0.",
        solution="print(0)",
        answer="0\n",
        concepts=("print",),
    ),
    Task(
        id="t0_03_print_hello",
        tier=0,
        prompt="Print the word hello (lowercase, no quotes).",
        solution="print('hello')",
        answer="hello\n",
        concepts=("print", "string"),
    ),
    Task(
        id="t0_04_print_hello_world",
        tier=0,
        prompt="Print exactly: Hello, World!",
        solution="print('Hello, World!')",
        answer="Hello, World!\n",
        concepts=("print", "string"),
    ),
    Task(
        id="t0_05_print_negative",
        tier=0,
        prompt="Print the number -7.",
        solution="print(-7)",
        answer="-7\n",
        concepts=("print",),
    ),
    Task(
        id="t0_06_print_big_int",
        tier=0,
        prompt="Print the number 1000000.",
        solution="print(1000000)",
        answer="1000000\n",
        concepts=("print",),
    ),
    Task(
        id="t0_07_print_float",
        tier=0,
        prompt="Print the number 3.14.",
        solution="print(3.14)",
        answer="3.14\n",
        concepts=("print",),
    ),
    Task(
        id="t0_08_print_true",
        tier=0,
        prompt="Print the boolean value True.",
        solution="print(True)",
        answer="True\n",
        concepts=("print",),
    ),
    Task(
        id="t0_09_print_letter",
        tier=0,
        prompt="Print the single letter A (uppercase, no quotes).",
        solution="print('A')",
        answer="A\n",
        concepts=("print", "string"),
    ),
    Task(
        id="t0_10_print_phrase",
        tier=0,
        prompt="Print exactly: the quick brown fox",
        solution="print('the quick brown fox')",
        answer="the quick brown fox\n",
        concepts=("print", "string"),
    ),
    # --- Tier 1: one-step arithmetic ------------------------------------------
    Task(
        id="t1_01_add",
        tier=1,
        prompt="Print the result of 3 plus 4.",
        solution="print(3 + 4)",
        answer="7\n",
        concepts=("arithmetic",),
    ),
    Task(
        id="t1_02_subtract",
        tier=1,
        prompt="Print the result of 10 minus 25.",
        solution="print(10 - 25)",
        answer="-15\n",
        concepts=("arithmetic",),
    ),
    Task(
        id="t1_03_six_times_seven",
        tier=1,
        prompt="Print the result of 6 times 7.",
        solution="print(6 * 7)",
        answer="42\n",
        concepts=("arithmetic",),
    ),
    Task(
        id="t1_04_floor_div",
        tier=1,
        prompt="Print the integer (floor) division of 17 by 5.",
        solution="print(17 // 5)",
        answer="3\n",
        concepts=("intdiv",),
    ),
    Task(
        id="t1_05_modulo",
        tier=1,
        prompt="Print the remainder when 17 is divided by 5.",
        solution="print(17 % 5)",
        answer="2\n",
        concepts=("modulo",),
    ),
    Task(
        id="t1_06_power",
        tier=1,
        prompt="Print 2 raised to the power 10.",
        solution="print(2 ** 10)",
        answer="1024\n",
        concepts=("power",),
    ),
    Task(
        id="t1_07_parens",
        tier=1,
        prompt="Print the result of (3 plus 4) times 5.",
        solution="print((3 + 4) * 5)",
        answer="35\n",
        concepts=("arithmetic",),
    ),
    Task(
        id="t1_08_true_div",
        tier=1,
        prompt="Print the result of 7 divided by 2 using true division (a decimal).",
        solution="print(7 / 2)",
        answer="3.5\n",
        concepts=("arithmetic",),
    ),
    Task(
        id="t1_09_precedence",
        tier=1,
        prompt="Print the result of 2 plus 3 times 4 (using normal operator precedence).",
        solution="print(2 + 3 * 4)",
        answer="14\n",
        concepts=("arithmetic",),
    ),
    Task(
        id="t1_10_negative_pow",
        tier=1,
        prompt="Print the result of 10 minus 3 raised to the power 2 (the exponent binds first).",
        solution="print(10 - 3 ** 2)",
        answer="1\n",
        concepts=("power", "arithmetic"),
    ),
    # --- Tier 2: variables & conditionals -------------------------------------
    Task(
        id="t2_01_xy_product_plus_x",
        tier=2,
        prompt="Set x to 5 and y to 3, then print x times y plus x.",
        solution="x = 5\ny = 3\nprint(x * y + x)",
        answer="20\n",
        concepts=("variable", "arithmetic"),
    ),
    Task(
        id="t2_02_sum_three",
        tier=2,
        prompt="Set a to 10, b to 20, and c to 30, then print their sum.",
        solution="a = 10\nb = 20\nc = 30\nprint(a + b + c)",
        answer="60\n",
        concepts=("variable", "arithmetic"),
    ),
    Task(
        id="t2_03_even_or_odd_18",
        tier=2,
        prompt="Print the word even if 18 is even, otherwise print odd. (18 is even.)",
        solution="n = 18\nif n % 2 == 0:\n  print('even')\nelse:\n  print('odd')",
        answer="even\n",
        concepts=("conditional", "modulo"),
    ),
    Task(
        id="t2_04_even_or_odd_7",
        tier=2,
        prompt="Print the word even if 7 is even, otherwise print odd. (7 is odd.)",
        solution="n = 7\nif n % 2 == 0:\n  print('even')\nelse:\n  print('odd')",
        answer="odd\n",
        concepts=("conditional", "modulo"),
    ),
    Task(
        id="t2_05_max_two",
        tier=2,
        prompt="Set a to 14 and b to 9, then print the larger of the two.",
        solution="a = 14\nb = 9\nif a > b:\n  print(a)\nelse:\n  print(b)",
        answer="14\n",
        concepts=("variable", "conditional"),
    ),
    Task(
        id="t2_06_swap",
        tier=2,
        prompt="Set a to 1 and b to 2, swap them using tuple assignment, then print a and b on one line separated by a single space.",
        solution="a = 1\nb = 2\na, b = b, a\nprint(a, b)",
        answer="2 1\n",
        concepts=("variable",),
    ),
    Task(
        id="t2_07_ternary_sign",
        tier=2,
        prompt="Set n to -4. Print the word positive if n is greater than 0, otherwise print nonpositive. (n is -4.)",
        solution="n = -4\nprint('positive' if n > 0 else 'nonpositive')",
        answer="nonpositive\n",
        concepts=("variable", "conditional"),
    ),
    Task(
        id="t2_08_fizzbuzz_single",
        tier=2,
        prompt="Set n to 15. If n is divisible by both 3 and 5 print FizzBuzz, else if divisible by 3 print Fizz, else if divisible by 5 print Buzz, else print n. (n is 15.)",
        solution=(
            "n = 15\n"
            "if n % 3 == 0 and n % 5 == 0:\n"
            "  print('FizzBuzz')\n"
            "elif n % 3 == 0:\n"
            "  print('Fizz')\n"
            "elif n % 5 == 0:\n"
            "  print('Buzz')\n"
            "else:\n"
            "  print(n)"
        ),
        answer="FizzBuzz\n",
        concepts=("conditional", "modulo"),
    ),
    Task(
        id="t2_09_abs_diff",
        tier=2,
        prompt="Set a to 3 and b to 8, then print the absolute value of a minus b.",
        solution="a = 3\nb = 8\nprint(abs(a - b))",
        answer="5\n",
        concepts=("variable", "arithmetic"),
    ),
    Task(
        id="t2_10_grade",
        tier=2,
        prompt="Set score to 72. Print A if score is at least 90, B if at least 80, C if at least 70, otherwise F. (score is 72, so the answer is C.)",
        solution=(
            "score = 72\n"
            "if score >= 90:\n"
            "  print('A')\n"
            "elif score >= 80:\n"
            "  print('B')\n"
            "elif score >= 70:\n"
            "  print('C')\n"
            "else:\n"
            "  print('F')"
        ),
        answer="C\n",
        concepts=("variable", "conditional"),
    ),
    # --- Tier 3: loops --------------------------------------------------------
    Task(
        id="t3_01_sum_1_to_100",
        tier=3,
        prompt="Print the sum of the integers from 1 to 100 (inclusive).",
        solution="total = 0\nfor i in range(1, 101):\n  total += i\nprint(total)",
        answer="5050\n",
        concepts=("loop", "arithmetic"),
    ),
    Task(
        id="t3_02_count_1_to_5",
        tier=3,
        prompt="Print the numbers 1 through 5 (inclusive), one per line.",
        solution="for i in range(1, 6):\n  print(i)",
        answer="1\n2\n3\n4\n5\n",
        concepts=("loop",),
    ),
    Task(
        id="t3_03_reverse_hello",
        tier=3,
        prompt="Print the string hello reversed (that is, olleh).",
        solution="print('hello'[::-1])",
        answer="olleh\n",
        concepts=("string",),
    ),
    Task(
        id="t3_04_factorial_loop",
        tier=3,
        prompt="Print the product of the integers from 1 to 5 (that is, 5 factorial).",
        solution="p = 1\nfor i in range(1, 6):\n  p *= i\nprint(p)",
        answer="120\n",
        concepts=("loop", "arithmetic"),
    ),
    Task(
        id="t3_05_sum_evens",
        tier=3,
        prompt="Print the sum of all even integers from 1 to 20 (inclusive).",
        solution="total = 0\nfor i in range(1, 21):\n  if i % 2 == 0:\n    total += i\nprint(total)",
        answer="110\n",
        concepts=("loop", "modulo"),
    ),
    Task(
        id="t3_06_countdown",
        tier=3,
        prompt="Using a while loop, print the numbers 5, 4, 3, 2, 1 each on its own line (counting down).",
        solution="n = 5\nwhile n >= 1:\n  print(n)\n  n -= 1",
        answer="5\n4\n3\n2\n1\n",
        concepts=("while", "loop"),
    ),
    Task(
        id="t3_07_fizzbuzz_1_to_15",
        tier=3,
        prompt=(
            "For each integer i from 1 to 15 (inclusive), print one line: Fizz if i "
            "is divisible by 3 and not 5, Buzz if divisible by 5 and not 3, FizzBuzz "
            "if divisible by both, otherwise the number i itself."
        ),
        solution=(
            "for i in range(1, 16):\n"
            "  if i % 3 == 0 and i % 5 == 0:\n"
            "    print('FizzBuzz')\n"
            "  elif i % 3 == 0:\n"
            "    print('Fizz')\n"
            "  elif i % 5 == 0:\n"
            "    print('Buzz')\n"
            "  else:\n"
            "    print(i)"
        ),
        answer="1\n2\nFizz\n4\nBuzz\nFizz\n7\n8\nFizz\nBuzz\n11\nFizz\n13\n14\nFizzBuzz\n",
        concepts=("loop", "conditional", "modulo"),
    ),
    Task(
        id="t3_08_join_range",
        tier=3,
        prompt="Print the numbers 1 through 5 (inclusive) on a single line separated by commas, with no spaces (that is, 1,2,3,4,5).",
        solution="print(','.join([str(i) for i in range(1, 6)]))",
        answer="1,2,3,4,5\n",
        concepts=("loop", "string", "list"),
    ),
    Task(
        id="t3_09_count_vowels",
        tier=3,
        prompt="Count how many vowels (a, e, i, o, u) are in the string education and print that count. (The answer is 5.)",
        solution=(
            "word = 'education'\n"
            "count = 0\n"
            "for ch in word:\n"
            "  if ch in 'aeiou':\n"
            "    count += 1\n"
            "print(count)"
        ),
        answer="5\n",
        concepts=("loop", "string"),
    ),
    Task(
        id="t3_10_max_in_list",
        tier=3,
        prompt="Given the list [3, 7, 2, 9, 4], print its largest element using a loop (do not use the built-in max).",
        solution=(
            "nums = [3, 7, 2, 9, 4]\n"
            "best = nums[0]\n"
            "for x in nums:\n"
            "  if x > best:\n"
            "    best = x\n"
            "print(best)"
        ),
        answer="9\n",
        concepts=("loop", "list"),
    ),
    # --- Tier 4: functions & recursion ----------------------------------------
    Task(
        id="t4_01_fib_10",
        tier=4,
        prompt=(
            "Define a function fib with fib(0)=0, fib(1)=1, and "
            "fib(n)=fib(n-1)+fib(n-2) for n>=2. Print fib(10). (The answer is 55.)"
        ),
        solution=(
            "def fib(n):\n"
            "  if n < 2:\n"
            "    return n\n"
            "  return fib(n - 1) + fib(n - 2)\n"
            "print(fib(10))"
        ),
        answer="55\n",
        concepts=("function", "recursion"),
    ),
    Task(
        id="t4_02_factorial_5",
        tier=4,
        prompt="Define a recursive factorial function (with factorial(0)=1) and print factorial(5). (The answer is 120.)",
        solution=(
            "def factorial(n):\n"
            "  if n == 0:\n"
            "    return 1\n"
            "  return n * factorial(n - 1)\n"
            "print(factorial(5))"
        ),
        answer="120\n",
        concepts=("function", "recursion"),
    ),
    Task(
        id="t4_03_is_prime_true",
        tier=4,
        prompt=(
            "Define a function is_prime(n) that returns True if n is prime and False "
            "otherwise, then print is_prime(13). (13 is prime, so print True.)"
        ),
        solution=(
            "def is_prime(n):\n"
            "  if n < 2:\n"
            "    return False\n"
            "  d = 2\n"
            "  while d * d <= n:\n"
            "    if n % d == 0:\n"
            "      return False\n"
            "    d += 1\n"
            "  return True\n"
            "print(is_prime(13))"
        ),
        answer="True\n",
        concepts=("function", "while", "modulo"),
    ),
    Task(
        id="t4_04_is_prime_false",
        tier=4,
        prompt=(
            "Define a function is_prime(n) that returns True if n is prime and False "
            "otherwise, then print is_prime(21). (21 = 3 times 7, so print False.)"
        ),
        solution=(
            "def is_prime(n):\n"
            "  if n < 2:\n"
            "    return False\n"
            "  d = 2\n"
            "  while d * d <= n:\n"
            "    if n % d == 0:\n"
            "      return False\n"
            "    d += 1\n"
            "  return True\n"
            "print(is_prime(21))"
        ),
        answer="False\n",
        concepts=("function", "while", "modulo"),
    ),
    Task(
        id="t4_05_gcd",
        tier=4,
        prompt=(
            "Define a function gcd(a, b) using the Euclidean algorithm and print "
            "gcd(48, 36). (The greatest common divisor is 12.)"
        ),
        solution=(
            "def gcd(a, b):\n"
            "  while b != 0:\n"
            "    a, b = b, a % b\n"
            "  return a\n"
            "print(gcd(48, 36))"
        ),
        answer="12\n",
        concepts=("function", "while", "modulo"),
    ),
    Task(
        id="t4_06_sum_digits",
        tier=4,
        prompt=(
            "Define a function sum_digits(n) that returns the sum of the decimal "
            "digits of a non-negative integer n, then print sum_digits(12345). "
            "(1+2+3+4+5 = 15.)"
        ),
        solution=(
            "def sum_digits(n):\n"
            "  total = 0\n"
            "  while n > 0:\n"
            "    total += n % 10\n"
            "    n //= 10\n"
            "  return total\n"
            "print(sum_digits(12345))"
        ),
        answer="15\n",
        concepts=("function", "while", "modulo"),
    ),
    Task(
        id="t4_07_triangular",
        tier=4,
        prompt=(
            "Define a function triangular(n) that returns the nth triangular number "
            "(the sum 1+2+...+n), then print triangular(10). (The answer is 55.)"
        ),
        solution=(
            "def triangular(n):\n"
            "  return n * (n + 1) // 2\n"
            "print(triangular(10))"
        ),
        answer="55\n",
        concepts=("function", "arithmetic"),
    ),
    Task(
        id="t4_08_power_recursive",
        tier=4,
        prompt=(
            "Define a recursive function power(base, exp) that computes base raised to "
            "exp for a non-negative integer exp (with power(base, 0)=1), then print "
            "power(3, 4). (The answer is 81.)"
        ),
        solution=(
            "def power(base, exp):\n"
            "  if exp == 0:\n"
            "    return 1\n"
            "  return base * power(base, exp - 1)\n"
            "print(power(3, 4))"
        ),
        answer="81\n",
        concepts=("function", "recursion"),
    ),
    Task(
        id="t4_09_count_primes",
        tier=4,
        prompt=(
            "Define a function is_prime(n) and use it to count how many integers from "
            "2 to 20 (inclusive) are prime, then print that count. (The answer is 8.)"
        ),
        solution=(
            "def is_prime(n):\n"
            "  if n < 2:\n"
            "    return False\n"
            "  d = 2\n"
            "  while d * d <= n:\n"
            "    if n % d == 0:\n"
            "      return False\n"
            "    d += 1\n"
            "  return True\n"
            "count = 0\n"
            "for k in range(2, 21):\n"
            "  if is_prime(k):\n"
            "    count += 1\n"
            "print(count)"
        ),
        answer="8\n",
        concepts=("function", "loop", "modulo"),
    ),
    Task(
        id="t4_10_collatz_steps",
        tier=4,
        prompt=(
            "Define a function collatz_steps(n) that counts how many steps it takes to "
            "reach 1 from n, where each step replaces n with n//2 if n is even or 3*n+1 "
            "if n is odd. Print collatz_steps(6). (6 -> 3 -> 10 -> 5 -> 16 -> 8 -> 4 -> "
            "2 -> 1 is 8 steps.)"
        ),
        solution=(
            "def collatz_steps(n):\n"
            "  steps = 0\n"
            "  while n != 1:\n"
            "    if n % 2 == 0:\n"
            "      n //= 2\n"
            "    else:\n"
            "      n = 3 * n + 1\n"
            "    steps += 1\n"
            "  return steps\n"
            "print(collatz_steps(6))"
        ),
        answer="8\n",
        concepts=("function", "while", "modulo"),
    ),
]


def load_tasks() -> list[Task]:
  """Return the 50 tasks."""
  return TASKS


if __name__ == "__main__":
  import micropython

  tasks = load_tasks()

  # Structural invariants.
  assert len(tasks) == 50, f"expected 50 tasks, got {len(tasks)}"
  ids = [t.id for t in tasks]
  assert len(set(ids)) == len(ids), "task ids are not unique"

  # Every solution must run cleanly and reproduce its gold answer exactly.
  failures = 0
  for t in tasks:
    r = micropython.run(t.solution)
    if not r.ok:
      print(f"[FAIL] {t.id}: solution errored: {r.error}")
      failures += 1
      continue
    if r.stdout != t.answer:
      print(
          f"[FAIL] {t.id}: stdout mismatch\n"
          f"  expected: {t.answer!r}\n"
          f"  actual:   {r.stdout!r}"
      )
      failures += 1
  assert failures == 0, f"{failures} task(s) failed validation"

  # Per-tier histogram.
  hist: dict[int, int] = {}
  for t in tasks:
    hist[t.tier] = hist.get(t.tier, 0) + 1
  print("Per-tier histogram:")
  for tier in sorted(hist):
    print(f"  tier {tier}: {hist[tier]} tasks")

  # A few sample (prompt -> answer) lines, one per tier.
  print("\nSample tasks (one per tier):")
  seen: set[int] = set()
  for t in tasks:
    if t.tier not in seen:
      seen.add(t.tier)
      preview = t.answer.replace("\n", "\\n")
      print(f"  [{t.id}] {t.prompt}  ->  {preview!r}")

  print(f"\nAll {len(tasks)} tasks valid.")
