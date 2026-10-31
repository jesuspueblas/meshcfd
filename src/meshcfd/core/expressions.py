"""Safe mathematical expressions for initial-condition fields and regions."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any


class ExpressionError(ValueError):
    """Raised when a mathematical expression is invalid or cannot be evaluated."""


def _safe_power(base: float, exponent: float) -> float:
    if abs(exponent) > 1024:
        raise ValueError("Exponent magnitude must not exceed 1024.")
    return operator.pow(base, exponent)


_FUNCTIONS: dict[str, Callable[..., float]] = {
    name: getattr(math, name)
    for name in (
        "acos",
        "acosh",
        "asin",
        "asinh",
        "atan",
        "atan2",
        "atanh",
        "ceil",
        "cos",
        "cosh",
        "degrees",
        "exp",
        "fabs",
        "floor",
        "hypot",
        "log",
        "log10",
        "radians",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
    )
}
_FUNCTIONS.update({"abs": abs, "max": max, "min": min, "pow": _safe_power})
_CONSTANTS = {"e": math.e, "pi": math.pi}
_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: _safe_power,
    ast.Mod: operator.mod,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_COMPARISONS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_MAX_EXPRESSION_LENGTH = 512
_MAX_AST_NODES = 128


class MathExpression:
    def __init__(self, source: str, tree: ast.Expression, boolean: bool) -> None:
        self.source = source
        self._tree = tree
        self._boolean = boolean

    def evaluate(self, x: float, y: float) -> float | bool:
        try:
            result = self._evaluate_node(self._tree.body, {"x": x, "y": y})
        except (ArithmeticError, TypeError, ValueError) as error:
            raise ExpressionError(
                f"Could not evaluate expression '{self.source}': {error}"
            ) from error
        if self._boolean:
            if not isinstance(result, bool):
                raise ExpressionError(
                    f"Domain expression '{self.source}' must evaluate to true or false."
                )
        elif isinstance(result, bool) or not isinstance(result, (int, float)):
            raise ExpressionError(
                f"Field expression '{self.source}' must evaluate to a number."
            )
        else:
            try:
                finite = math.isfinite(result)
            except OverflowError:
                finite = False
            if not finite:
                raise ExpressionError(
                    f"Field expression '{self.source}' must evaluate to a finite number."
                )
        return result

    @classmethod
    def _evaluate_node(cls, node: ast.AST, variables: dict[str, float]) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in variables:
                return variables[node.id]
            if node.id in _CONSTANTS:
                return _CONSTANTS[node.id]
            raise ExpressionError(f"Unknown expression name '{node.id}'.")
        if isinstance(node, ast.BinOp):
            operation = _BINARY_OPERATORS[type(node.op)]
            return operation(
                cls._evaluate_node(node.left, variables),
                cls._evaluate_node(node.right, variables),
            )
        if isinstance(node, ast.UnaryOp):
            operand = cls._evaluate_node(node.operand, variables)
            if isinstance(node.op, ast.Not):
                if not isinstance(operand, bool):
                    raise ExpressionError("The 'not' operator needs a boolean value.")
                return not operand
            return _UNARY_OPERATORS[type(node.op)](operand)
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                for value in node.values:
                    result = cls._evaluate_node(value, variables)
                    if not isinstance(result, bool):
                        raise ExpressionError("The 'and' operator needs boolean values.")
                    if not result:
                        return False
                return True
            for value in node.values:
                result = cls._evaluate_node(value, variables)
                if not isinstance(result, bool):
                    raise ExpressionError("The 'or' operator needs boolean values.")
                if result:
                    return True
            return False
        if isinstance(node, ast.Compare):
            left = cls._evaluate_node(node.left, variables)
            for operation, comparator in zip(node.ops, node.comparators):
                right = cls._evaluate_node(comparator, variables)
                if not _COMPARISONS[type(operation)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.Call):
            function = _FUNCTIONS[node.func.id]
            return function(
                *(cls._evaluate_node(argument, variables) for argument in node.args)
            )
        raise ExpressionError("Unsupported expression syntax.")


def compile_expression(source: str, *, boolean: bool = False) -> MathExpression:
    if not isinstance(source, str) or not source.strip():
        raise ExpressionError("Expression must not be empty.")
    if len(source) > _MAX_EXPRESSION_LENGTH:
        raise ExpressionError("Expression is too long.")
    try:
        tree = ast.parse(source.replace("^", "**"), mode="eval")
    except SyntaxError as error:
        raise ExpressionError(f"Invalid expression '{source}': {error.msg}.") from error
    nodes = list(ast.walk(tree))
    if len(nodes) > _MAX_AST_NODES:
        raise ExpressionError("Expression is too complex.")
    if not boolean and any(
        isinstance(node, ast.Constant) and isinstance(node.value, bool)
        for node in nodes
    ):
        raise ExpressionError("Field expression must be numeric.")
    if not boolean and any(
        isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)
        for node in nodes
    ):
        raise ExpressionError("Field expression must be numeric.")
    _validate_node(tree.body)
    if boolean and not isinstance(
        tree.body, (ast.Compare, ast.BoolOp)
    ) and not (
        isinstance(tree.body, ast.UnaryOp) and isinstance(tree.body.op, ast.Not)
    ) and not (isinstance(tree.body, ast.Constant) and isinstance(tree.body.value, bool)):
        raise ExpressionError("Domain expression must be a boolean predicate.")
    if not boolean and isinstance(
        tree.body, (ast.Compare, ast.BoolOp)
    ):
        raise ExpressionError("Field expression must be numeric.")
    if (
        not boolean
        and isinstance(tree.body, ast.Constant)
        and isinstance(tree.body.value, bool)
    ):
        raise ExpressionError("Field expression must be numeric.")
    return MathExpression(source, tree, boolean)


def _validate_node(node: ast.AST) -> None:
    if isinstance(node, ast.Expression):
        _validate_node(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return
        if not isinstance(node.value, (int, float)) or (
            isinstance(node.value, float) and not math.isfinite(node.value)
        ):
            raise ExpressionError("Only finite numeric constants are supported.")
    elif isinstance(node, ast.Name):
        if node.id not in {"x", "y", *_CONSTANTS}:
            raise ExpressionError(f"Unknown expression name '{node.id}'.")
    elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        _validate_node(node.left)
        _validate_node(node.right)
    elif isinstance(node, ast.UnaryOp) and (
        type(node.op) in _UNARY_OPERATORS or isinstance(node.op, ast.Not)
    ):
        _validate_node(node.operand)
    elif isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        for value in node.values:
            _validate_node(value)
    elif isinstance(node, ast.Compare) and all(
        type(operation) in _COMPARISONS for operation in node.ops
    ):
        _validate_node(node.left)
        for comparator in node.comparators:
            _validate_node(comparator)
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        for argument in node.args:
            _validate_node(argument)
    else:
        raise ExpressionError("Unsupported expression syntax.")
