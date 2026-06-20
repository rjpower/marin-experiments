# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A purely-functional, sandboxed interpreter for a small subset of Python.

This module is the execution environment + verifier for an RL "agentic coding"
task: a small language model writes a tiny Python program (e.g. ``print(2)`` or
"print the 10th Fibonacci number") and we grade the program's captured stdout
against a gold answer. To make grading safe and reproducible the program is NOT
run with the host ``exec``/``eval``. Instead :func:`run` parses the source with
:mod:`ast` and walks the tree with an explicit interpreter, so:

  * **Deterministic.** The same source always yields an identical
    :class:`ExecResult` (no clock, no randomness, no host state).
  * **Sandboxed.** No imports, no filesystem/network/os access, no dunder
    attribute access, and only an explicit whitelist of builtins and methods.
  * **Bounded.** A step counter and an output cap guarantee termination, so an
    infinite loop or runaway recursion stops cleanly with ``ok=False`` rather
    than hanging the host.

:func:`run` never raises: every failure (syntax error, runtime error,
unsupported construct, step-limit, output-limit) is returned as ``ok=False`` with
a short ``"ErrorType: message"`` string.

Example:
  >>> run("print('hi', 2, sep='-')").stdout
  'hi-2\\n'
"""

from __future__ import annotations

import ast
import dataclasses
from typing import Any, Callable


@dataclasses.dataclass(frozen=True)
class ExecResult:
  """The outcome of executing a micropython program.

  Attributes:
    stdout: Everything printed; each ``print()`` appends per its ``end=`` kwarg
      (default ``'\\n'``). Truncated to ``max_output`` characters on overflow.
    ok: True iff the program ran to completion with no error.
    error: A short ``"ErrorType: message"`` string when ``ok`` is False, else
      None.
    steps: Number of AST nodes evaluated. A diagnostic and a determinism
      witness (same source => same step count).
  """

  stdout: str
  ok: bool
  error: str | None
  steps: int


# --- Internal control-flow / error signals -------------------------------------
#
# These are host exceptions used purely to unwind the interpreter's own Python
# call stack; they never escape run(). Program-level errors (NameError, etc.) are
# represented by MicroError, which carries the "ErrorType: message" we report.


class _MicroError(Exception):
  """A program-level error to surface to the caller as ``ok=False``."""

  def __init__(self, kind: str, message: str):
    super().__init__(f"{kind}: {message}")
    self.kind = kind
    self.message = message


class _StepLimit(Exception):
  """Raised when the step budget is exhausted (terminates infinite loops)."""


class _OutputLimit(Exception):
  """Raised when the captured output exceeds ``max_output``."""


class _BreakSignal(Exception):
  """Unwinds the host stack to the enclosing loop for a ``break``."""


class _ContinueSignal(Exception):
  """Unwinds the host stack to the enclosing loop for a ``continue``."""


class _ReturnSignal(Exception):
  """Unwinds the host stack out of a function body for a ``return``."""

  def __init__(self, value: Any):
    super().__init__("return")
    self.value = value


@dataclasses.dataclass
class _Function:
  """A user-defined function (the value bound by a ``def`` statement)."""

  name: str
  params: list[str]
  defaults: list[Any]  # defaults for the trailing ``len(defaults)`` params
  body: list[ast.stmt]
  # Globals captured at definition time so a call can read module-level names.
  globals_env: dict[str, Any]


# --- Builtins whitelist --------------------------------------------------------
#
# Only these names are visible to a program. They are plain references to safe
# Python builtins (or thin wrappers). Anything not here resolves to NameError.

_SAFE_BUILTINS: dict[str, Any] = {
    "range": range,
    "len": len,
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "sorted": sorted,
    "reversed": lambda x: list(reversed(x)),
    "enumerate": lambda *a, **k: list(enumerate(*a, **k)),
    "int": int,
    "str": str,
    "float": float,
    "bool": bool,
    "list": list,
    "tuple": tuple,
    "round": round,
    "map": lambda f, *its: list(map(f, *its)),
    "filter": lambda f, it: list(filter(f, it)),
    "zip": lambda *its: list(zip(*its)),
    # ``print`` is installed per-run by the interpreter (it needs the output
    # buffer), so it is intentionally absent here.
}

# Methods we allow on list/tuple/str values, by receiver type. Each maps a
# method name to a callable that takes (receiver, *args) -> result. Keeping this
# explicit (rather than getattr) is what blocks dunder / introspection access.
_LIST_METHODS: dict[str, Callable[..., Any]] = {
    # Mutating methods return None and mutate the list in place, matching Python.
    "append": lambda lst, x: lst.append(x),
    "pop": lambda lst, *a: lst.pop(*a),
    "insert": lambda lst, i, x: lst.insert(i, x),
    "index": lambda lst, *a: lst.index(*a),
    "count": lambda lst, x: lst.count(x),
}

_STR_METHODS: dict[str, Callable[..., Any]] = {
    "upper": lambda s: s.upper(),
    "lower": lambda s: s.lower(),
    "strip": lambda s, *a: s.strip(*a),
    "split": lambda s, *a: s.split(*a),
    "join": lambda s, it: s.join(it),
    "replace": lambda s, *a: s.replace(*a),
    "find": lambda s, *a: s.find(*a),
}

_TUPLE_METHODS: dict[str, Callable[..., Any]] = {
    "index": lambda t, *a: t.index(*a),
    "count": lambda t, x: t.count(x),
}


class _Interpreter:
  """A tree-walking interpreter for one program execution.

  One instance handles a single :func:`run`. It owns the shared output buffer,
  the step counter and the step/output budgets. Evaluation methods raise the
  internal signal exceptions above for control flow and :class:`_MicroError` for
  program errors.
  """

  def __init__(self, max_steps: int, max_output: int):
    self.max_steps = max_steps
    self.max_output = max_output
    self.steps = 0
    self._out: list[str] = []
    self._out_len = 0

  # -- budgeting --------------------------------------------------------------

  def _tick(self) -> None:
    """Charge one step; raise :class:`_StepLimit` when the budget is gone."""
    self.steps += 1
    if self.steps > self.max_steps:
      raise _StepLimit()

  def _emit(self, text: str) -> None:
    """Append to stdout, raising :class:`_OutputLimit` past the cap.

    On overflow we still record the truncated prefix so the caller sees the
    partial output produced before the limit was hit.
    """
    if self._out_len + len(text) > self.max_output:
      remaining = self.max_output - self._out_len
      if remaining > 0:
        self._out.append(text[:remaining])
        self._out_len += remaining
      raise _OutputLimit()
    self._out.append(text)
    self._out_len += len(text)

  @property
  def stdout(self) -> str:
    return "".join(self._out)

  # -- the program-visible print ----------------------------------------------

  def _print(self, *args: Any, sep: str = " ", end: str = "\n") -> None:
    """The only output primitive; mirrors ``print`` with ``sep``/``end``."""
    if not isinstance(sep, str):
      raise _MicroError("TypeError", "sep must be a string")
    if not isinstance(end, str):
      raise _MicroError("TypeError", "end must be a string")
    self._emit(sep.join(_py_str(a) for a in args) + end)

  # -- statement execution ----------------------------------------------------

  def exec_block(self, body: list[ast.stmt], env: dict[str, Any]) -> None:
    for stmt in body:
      self.exec_stmt(stmt, env)

  def exec_stmt(self, node: ast.stmt, env: dict[str, Any]) -> None:
    self._tick()
    method = getattr(self, "_st_" + type(node).__name__, None)
    if method is None:
      raise _MicroError("UnsupportedSyntax", type(node).__name__)
    method(node, env)

  def _st_Expr(self, node: ast.Expr, env: dict[str, Any]) -> None:
    self.eval_expr(node.value, env)

  def _st_Pass(self, node: ast.Pass, env: dict[str, Any]) -> None:
    pass

  def _st_Assign(self, node: ast.Assign, env: dict[str, Any]) -> None:
    value = self.eval_expr(node.value, env)
    for target in node.targets:
      self._assign(target, value, env)

  def _st_AugAssign(self, node: ast.AugAssign, env: dict[str, Any]) -> None:
    if not isinstance(node.target, ast.Name):
      raise _MicroError(
          "UnsupportedSyntax", "augmented assignment to non-name target"
      )
    current = self._load_name(node.target.id, env)
    result = self._binop(node.op, current, self.eval_expr(node.value, env))
    env[node.target.id] = result

  def _st_If(self, node: ast.If, env: dict[str, Any]) -> None:
    if _truthy(self.eval_expr(node.test, env)):
      self.exec_block(node.body, env)
    else:
      self.exec_block(node.orelse, env)

  def _st_While(self, node: ast.While, env: dict[str, Any]) -> None:
    while _truthy(self.eval_expr(node.test, env)):
      self._tick()  # charge per iteration so infinite loops hit the step limit
      try:
        self.exec_block(node.body, env)
      except _BreakSignal:
        break
      except _ContinueSignal:
        continue
    else:
      self.exec_block(node.orelse, env)

  def _st_For(self, node: ast.For, env: dict[str, Any]) -> None:
    iterable = self.eval_expr(node.iter, env)
    broke = False
    for item in _as_iter(iterable):
      self._tick()  # charge per iteration to bound large iterables
      self._assign(node.target, item, env)
      try:
        self.exec_block(node.body, env)
      except _BreakSignal:
        broke = True
        break
      except _ContinueSignal:
        continue
    if not broke:
      self.exec_block(node.orelse, env)

  def _st_Break(self, node: ast.Break, env: dict[str, Any]) -> None:
    raise _BreakSignal()

  def _st_Continue(self, node: ast.Continue, env: dict[str, Any]) -> None:
    raise _ContinueSignal()

  def _st_FunctionDef(self, node: ast.FunctionDef, env: dict[str, Any]) -> None:
    if getattr(node, "decorator_list", None):
      raise _MicroError("UnsupportedSyntax", "decorators")
    args = node.args
    if args.vararg or args.kwarg or args.kwonlyargs or args.posonlyargs:
      raise _MicroError(
          "UnsupportedSyntax", "only positional params with defaults are allowed"
      )
    defaults = [self.eval_expr(d, env) for d in args.defaults]
    env[node.name] = _Function(
        name=node.name,
        params=[a.arg for a in args.args],
        defaults=defaults,
        body=node.body,
        globals_env=env,
    )

  def _st_Return(self, node: ast.Return, env: dict[str, Any]) -> None:
    value = None if node.value is None else self.eval_expr(node.value, env)
    raise _ReturnSignal(value)

  # -- assignment targets -----------------------------------------------------

  def _assign(self, target: ast.expr, value: Any, env: dict[str, Any]) -> None:
    if isinstance(target, ast.Name):
      env[target.id] = value
    elif isinstance(target, (ast.Tuple, ast.List)):
      items = list(_as_iter(value))
      if len(items) != len(target.elts):
        raise _MicroError(
            "ValueError",
            f"expected {len(target.elts)} values to unpack, got {len(items)}",
        )
      for sub, item in zip(target.elts, items):
        self._assign(sub, item, env)
    elif isinstance(target, ast.Subscript):
      container = self.eval_expr(target.value, env)
      key = self._subscript_key(target.slice, env)
      try:
        container[key] = value
      except (TypeError, IndexError, KeyError) as exc:
        raise _MicroError(type(exc).__name__, str(exc)) from exc
    else:
      raise _MicroError(
          "UnsupportedSyntax", f"assignment target {type(target).__name__}"
      )

  # -- expression evaluation --------------------------------------------------

  def eval_expr(self, node: ast.expr, env: dict[str, Any]) -> Any:
    self._tick()
    method = getattr(self, "_ex_" + type(node).__name__, None)
    if method is None:
      raise _MicroError("UnsupportedSyntax", type(node).__name__)
    return method(node, env)

  def _ex_Constant(self, node: ast.Constant, env: dict[str, Any]) -> Any:
    value = node.value
    # Reject exotic constants (Ellipsis, bytes, complex) we don't model.
    if not isinstance(value, (int, float, str, bool, type(None))):
      raise _MicroError("UnsupportedSyntax", f"constant {type(value).__name__}")
    return value

  def _ex_Name(self, node: ast.Name, env: dict[str, Any]) -> Any:
    return self._load_name(node.id, env)

  def _ex_List(self, node: ast.List, env: dict[str, Any]) -> Any:
    return [self.eval_expr(e, env) for e in node.elts]

  def _ex_Tuple(self, node: ast.Tuple, env: dict[str, Any]) -> Any:
    return tuple(self.eval_expr(e, env) for e in node.elts)

  def _ex_UnaryOp(self, node: ast.UnaryOp, env: dict[str, Any]) -> Any:
    operand = self.eval_expr(node.operand, env)
    op = node.op
    try:
      if isinstance(op, ast.UAdd):
        return +operand
      if isinstance(op, ast.USub):
        return -operand
      if isinstance(op, ast.Not):
        return not _truthy(operand)
      if isinstance(op, ast.Invert):
        return ~operand
    except TypeError as exc:
      raise _MicroError("TypeError", str(exc)) from exc
    raise _MicroError("UnsupportedSyntax", type(op).__name__)

  def _ex_BinOp(self, node: ast.BinOp, env: dict[str, Any]) -> Any:
    left = self.eval_expr(node.left, env)
    right = self.eval_expr(node.right, env)
    return self._binop(node.op, left, right)

  def _ex_BoolOp(self, node: ast.BoolOp, env: dict[str, Any]) -> Any:
    if isinstance(node.op, ast.And):
      result: Any = True
      for value_node in node.values:
        result = self.eval_expr(value_node, env)
        if not _truthy(result):
          return result  # short-circuit, returning the falsy operand
      return result
    # Or
    result = False
    for value_node in node.values:
      result = self.eval_expr(value_node, env)
      if _truthy(result):
        return result
    return result

  def _ex_Compare(self, node: ast.Compare, env: dict[str, Any]) -> Any:
    left = self.eval_expr(node.left, env)
    for op, right_node in zip(node.ops, node.comparators):
      right = self.eval_expr(right_node, env)
      if not self._compare(op, left, right):
        return False  # chained compare short-circuits, like Python
      left = right
    return True

  def _ex_IfExp(self, node: ast.IfExp, env: dict[str, Any]) -> Any:
    if _truthy(self.eval_expr(node.test, env)):
      return self.eval_expr(node.body, env)
    return self.eval_expr(node.orelse, env)

  def _ex_Subscript(self, node: ast.Subscript, env: dict[str, Any]) -> Any:
    container = self.eval_expr(node.value, env)
    key = self._subscript_key(node.slice, env)
    try:
      return container[key]
    except IndexError as exc:
      raise _MicroError("IndexError", str(exc)) from exc
    except KeyError as exc:
      raise _MicroError("KeyError", str(exc)) from exc
    except TypeError as exc:
      raise _MicroError("TypeError", str(exc)) from exc

  def _subscript_key(self, slice_node: ast.expr, env: dict[str, Any]) -> Any:
    if isinstance(slice_node, ast.Slice):
      lower = (
          None if slice_node.lower is None else self.eval_expr(slice_node.lower, env)
      )
      upper = (
          None if slice_node.upper is None else self.eval_expr(slice_node.upper, env)
      )
      step = (
          None if slice_node.step is None else self.eval_expr(slice_node.step, env)
      )
      return slice(lower, upper, step)
    return self.eval_expr(slice_node, env)

  def _ex_ListComp(self, node: ast.ListComp, env: dict[str, Any]) -> Any:
    if len(node.generators) != 1:
      raise _MicroError(
          "UnsupportedSyntax", "only single-generator comprehensions are allowed"
      )
    gen = node.generators[0]
    if gen.is_async:
      raise _MicroError("UnsupportedSyntax", "async comprehension")
    result = []
    for item in _as_iter(self.eval_expr(gen.iter, env)):
      self._tick()
      # Comprehensions get a child scope so the loop var does not leak, but they
      # can still read outer names (Python's comprehension scoping).
      local = dict(env)
      self._assign(gen.target, item, local)
      if all(_truthy(self.eval_expr(c, local)) for c in gen.ifs):
        result.append(self.eval_expr(node.elt, local))
    return result

  def _ex_JoinedStr(self, node: ast.JoinedStr, env: dict[str, Any]) -> Any:
    """Evaluate an f-string into a plain ``str``."""
    parts: list[str] = []
    for piece in node.values:
      if isinstance(piece, ast.Constant):
        parts.append(_py_str(piece.value))
      elif isinstance(piece, ast.FormattedValue):
        parts.append(self._format_value(piece, env))
      else:
        raise _MicroError("UnsupportedSyntax", type(piece).__name__)
    return "".join(parts)

  def _format_value(self, node: ast.FormattedValue, env: dict[str, Any]) -> str:
    value = self.eval_expr(node.value, env)
    # conversion: -1 none, 115 !s, 114 !r, 97 !a
    if node.conversion == 115:
      value = _py_str(value)
    elif node.conversion == 114:
      value = repr(value)
    elif node.conversion == 97:
      value = ascii(value)
    spec = ""
    if node.format_spec is not None:
      spec = self.eval_expr(node.format_spec, env)
    try:
      return format(value, spec)
    except (ValueError, TypeError) as exc:
      raise _MicroError(type(exc).__name__, str(exc)) from exc

  def _ex_Call(self, node: ast.Call, env: dict[str, Any]) -> Any:
    if node.keywords and any(k.arg is None for k in node.keywords):
      raise _MicroError("UnsupportedSyntax", "**kwargs call")
    if any(isinstance(a, ast.Starred) for a in node.args):
      raise _MicroError("UnsupportedSyntax", "*args call")

    # Method call: evaluate as a whitelisted method on the receiver.
    if isinstance(node.func, ast.Attribute):
      return self._call_method(node, env)

    func = self.eval_expr(node.func, env)
    args = [self.eval_expr(a, env) for a in node.args]
    kwargs = {k.arg: self.eval_expr(k.value, env) for k in node.keywords}
    return self._apply(func, args, kwargs)

  def _apply(self, func: Any, args: list[Any], kwargs: dict[str, Any]) -> Any:
    if isinstance(func, _Function):
      return self._call_user_function(func, args, kwargs)
    if callable(func):  # a whitelisted builtin or our print/method wrapper
      try:
        return func(*args, **kwargs)
      except _MicroError:
        raise
      except (
          _StepLimit,
          _OutputLimit,
          _ReturnSignal,
          _BreakSignal,
          _ContinueSignal,
      ):
        raise
      except Exception as exc:  # surface builtin errors as program errors
        raise _MicroError(type(exc).__name__, str(exc)) from exc
    raise _MicroError("TypeError", f"{_py_str(func)} is not callable")

  def _call_user_function(
      self, func: _Function, args: list[Any], kwargs: dict[str, Any]
  ) -> Any:
    self._tick()
    params = func.params
    if len(args) > len(params):
      raise _MicroError(
          "TypeError",
          f"{func.name}() takes {len(params)} positional arguments "
          f"but {len(args)} were given",
      )
    # Build the fresh local scope: positionals, then kwargs, then defaults.
    local: dict[str, Any] = {}
    for name, value in zip(params, args):
      local[name] = value
    bound = set(params[: len(args)])
    for key, value in kwargs.items():
      if key not in params:
        raise _MicroError(
            "TypeError", f"{func.name}() got an unexpected keyword argument '{key}'"
        )
      if key in bound:
        raise _MicroError(
            "TypeError", f"{func.name}() got multiple values for argument '{key}'"
        )
      local[key] = value
      bound.add(key)
    # Apply defaults for any still-unbound trailing params.
    default_start = len(params) - len(func.defaults)
    for index, name in enumerate(params):
      if name not in local:
        if index >= default_start:
          local[name] = func.defaults[index - default_start]
        else:
          raise _MicroError(
              "TypeError",
              f"{func.name}() missing required argument '{name}'",
          )
    # The call sees its captured globals for reads (chained via a layered dict),
    # but assigns into ``local`` only. We model this with a ChainMap-like view.
    scope = _Scope(local, func.globals_env)
    try:
      self.exec_block(func.body, scope)
    except _ReturnSignal as signal:
      return signal.value
    return None

  def _call_method(self, node: ast.Call, env: dict[str, Any]) -> Any:
    attr = node.func
    assert isinstance(attr, ast.Attribute)
    if attr.attr.startswith("__"):
      raise _MicroError("AttributeError", f"access to '{attr.attr}' is not allowed")
    receiver = self.eval_expr(attr.value, env)
    args = [self.eval_expr(a, env) for a in node.args]
    if node.keywords:
      raise _MicroError("UnsupportedSyntax", "keyword arguments to methods")

    if isinstance(receiver, list):
      table = _LIST_METHODS
    elif isinstance(receiver, str):
      table = _STR_METHODS
    elif isinstance(receiver, tuple):
      table = _TUPLE_METHODS
    else:
      raise _MicroError(
          "AttributeError",
          f"'{type(receiver).__name__}' object has no method '{attr.attr}'",
      )
    impl = table.get(attr.attr)
    if impl is None:
      raise _MicroError(
          "AttributeError",
          f"'{type(receiver).__name__}' object has no method '{attr.attr}'",
      )
    try:
      return impl(receiver, *args)
    except _MicroError:
      raise
    except Exception as exc:
      raise _MicroError(type(exc).__name__, str(exc)) from exc

  def _ex_Attribute(self, node: ast.Attribute, env: dict[str, Any]) -> Any:
    # Bare attribute access (not a method call) is never allowed: there are no
    # whitelisted data attributes, and this is where dunder access would leak.
    raise _MicroError(
        "AttributeError", f"access to attribute '{node.attr}' is not allowed"
    )

  # -- shared operator helpers ------------------------------------------------

  def _binop(self, op: ast.operator, left: Any, right: Any) -> Any:
    try:
      if isinstance(op, ast.Add):
        return left + right
      if isinstance(op, ast.Sub):
        return left - right
      if isinstance(op, ast.Mult):
        return left * right
      if isinstance(op, ast.Div):
        return left / right
      if isinstance(op, ast.FloorDiv):
        return left // right
      if isinstance(op, ast.Mod):
        return left % right
      if isinstance(op, ast.Pow):
        # Guard against giant exponentiations that would burn host CPU/memory
        # despite cheap step counts (e.g. 10 ** 10**9).
        _guard_pow(left, right)
        return left ** right
    except ZeroDivisionError as exc:
      raise _MicroError("ZeroDivisionError", str(exc)) from exc
    except _MicroError:
      raise
    except (TypeError, ValueError, OverflowError) as exc:
      raise _MicroError(type(exc).__name__, str(exc)) from exc
    raise _MicroError("UnsupportedSyntax", type(op).__name__)

  def _compare(self, op: ast.cmpop, left: Any, right: Any) -> bool:
    try:
      if isinstance(op, ast.Eq):
        return bool(left == right)
      if isinstance(op, ast.NotEq):
        return bool(left != right)
      if isinstance(op, ast.Lt):
        return bool(left < right)
      if isinstance(op, ast.LtE):
        return bool(left <= right)
      if isinstance(op, ast.Gt):
        return bool(left > right)
      if isinstance(op, ast.GtE):
        return bool(left >= right)
      if isinstance(op, ast.In):
        return bool(left in right)
      if isinstance(op, ast.NotIn):
        return bool(left not in right)
    except TypeError as exc:
      raise _MicroError("TypeError", str(exc)) from exc
    raise _MicroError("UnsupportedSyntax", type(op).__name__)

  # -- name resolution --------------------------------------------------------

  def _load_name(self, name: str, env: dict[str, Any]) -> Any:
    if name in env:
      return env[name]
    if name == "print":
      return self._print
    if name in _SAFE_BUILTINS:
      return _SAFE_BUILTINS[name]
    raise _MicroError("NameError", f"name '{name}' is not defined")


class _Scope(dict):
  """A function-local scope that falls back to captured globals on read.

  Writes (``__setitem__``) land in the local dict only, so assignment inside a
  function never mutates module globals -- giving the required "read globals,
  write locals" semantics. Reads (``__contains__`` / ``__getitem__``) see locals
  first, then globals.
  """

  def __init__(self, local: dict[str, Any], globals_env: dict[str, Any]):
    super().__init__(local)
    self._globals = globals_env

  def __contains__(self, key: object) -> bool:
    return super().__contains__(key) or key in self._globals

  def __getitem__(self, key: Any) -> Any:
    if super().__contains__(key):
      return super().__getitem__(key)
    return self._globals[key]


# --- Module-level helpers ------------------------------------------------------


def _truthy(value: Any) -> bool:
  """Python truthiness over the values our interpreter can produce."""
  return bool(value)


def _as_iter(value: Any):
  """Iterate over a value, mapping unsupported iteration to a program error."""
  if isinstance(value, (list, tuple, str, range, enumerate, zip)):
    return iter(value)
  try:
    return iter(value)
  except TypeError as exc:
    raise _MicroError("TypeError", str(exc)) from exc


def _py_str(value: Any) -> str:
  """``str()`` for printing, with ``True``/``False``/``None`` like Python."""
  return str(value)


def _guard_pow(base: Any, exp: Any) -> None:
  """Reject exponentiations whose result would be absurdly large.

  Step counting cannot see the cost of ``10 ** 10**9`` (one BinOp node), so we
  bound the bit-length of integer powers to keep execution cheap and bounded.
  """
  if isinstance(base, int) and isinstance(exp, int) and exp > 0 and base != 0:
    # Approximate result bit-length = exp * log2(|base|).
    abs_base = abs(base)
    if abs_base > 1 and exp * abs_base.bit_length() > 100_000:
      raise _MicroError("OverflowError", "exponentiation result too large")


def run(source: str, *, max_steps: int = 100_000, max_output: int = 10_000) -> ExecResult:
  """Execute a micropython program purely-functionally and capture its stdout.

  The source is parsed with :mod:`ast` and walked by an explicit interpreter; it
  is never handed to the host ``exec``/``eval``. Execution is deterministic,
  sandboxed and bounded, so the same source always returns an identical result
  and infinite loops / runaway output terminate cleanly.

  Args:
    source: The program text.
    max_steps: Maximum number of AST nodes to evaluate before aborting with a
      step-limit error (bounds infinite loops / runaway recursion).
    max_output: Maximum number of characters to capture; further output aborts
      with an output-limit error (the truncated prefix is still returned).

  Returns:
    An :class:`ExecResult`. ``ok`` is True only on a clean run to completion;
    otherwise ``ok`` is False and ``error`` is a short ``"ErrorType: message"``.
    This function never raises.
  """
  interp = _Interpreter(max_steps=max_steps, max_output=max_output)
  try:
    tree = ast.parse(source, mode="exec")
  except SyntaxError as exc:
    msg = exc.msg if exc.msg else "invalid syntax"
    return ExecResult(stdout="", ok=False, error=f"SyntaxError: {msg}", steps=0)
  except ValueError as exc:  # e.g. null bytes in source
    return ExecResult(stdout="", ok=False, error=f"SyntaxError: {exc}", steps=0)

  env: dict[str, Any] = {}
  try:
    interp.exec_block(tree.body, env)
  except _MicroError as exc:
    return ExecResult(
        stdout=interp.stdout, ok=False, error=str(exc), steps=interp.steps
    )
  except _StepLimit:
    return ExecResult(
        stdout=interp.stdout,
        ok=False,
        error="StepLimit: exceeded max_steps",
        steps=interp.steps,
    )
  except _OutputLimit:
    return ExecResult(
        stdout=interp.stdout,
        ok=False,
        error="OutputLimit: exceeded max_output",
        steps=interp.steps,
    )
  except _ReturnSignal:
    return ExecResult(
        stdout=interp.stdout,
        ok=False,
        error="SyntaxError: 'return' outside function",
        steps=interp.steps,
    )
  except (_BreakSignal, _ContinueSignal) as exc:
    kind = "break" if isinstance(exc, _BreakSignal) else "continue"
    return ExecResult(
        stdout=interp.stdout,
        ok=False,
        error=f"SyntaxError: '{kind}' outside loop",
        steps=interp.steps,
    )
  except RecursionError:
    # Deep user recursion can exhaust the host stack before max_steps; report it
    # as a clean program error rather than letting it escape.
    return ExecResult(
        stdout=interp.stdout,
        ok=False,
        error="RecursionError: maximum recursion depth exceeded",
        steps=interp.steps,
    )
  except Exception as exc:  # pragma: no cover - defensive catch-all
    return ExecResult(
        stdout=interp.stdout,
        ok=False,
        error=f"{type(exc).__name__}: {exc}",
        steps=interp.steps,
    )
  return ExecResult(stdout=interp.stdout, ok=True, error=None, steps=interp.steps)
